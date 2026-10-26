# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the reward shaping helpers in rpl_reward_manager."""

import unittest

from rpl.llm_agent.rpl_reward_manager import (
    _answer_grounded,
    _answer_in_text,
    _billable_calls,
    _coverage_bonus,
    _efficiency_bonus,
    _grounding_bonus,
    _recall_precision,
    _union_recall_precision,
    _validate_efficiency_params,
    _validate_reward_mode,
)
from verl.utils.reward_score.search_r1_like_qa_em import well_formed

# Distinct sentinels per metric so a mis-wired mode is caught. scale=0.5.
_METRICS = dict(answer_recall=0.7, answer_precision=0.3, union_recall=1.0, union_precision=0.4)

B, P = 0.2, 0.02  # min_call_bonus, per_call_price


def _coverage(mode, em=0.0):
    return _coverage_bonus(mode, em, 0.5, **_METRICS)


def _efficiency(em, calls, is_val=False, b=B, p=P):
    return _efficiency_bonus(is_val, em, b, p, calls)


class RewardModeTest(unittest.TestCase):
    def test_accepts_every_handled_mode(self):
        for mode in ("em", "conditional", "conditional_multi", "grounded", "autorefine"):
            _validate_reward_mode(mode)
        for union in ("", "union_"):
            for metric in ("hit", "prec"):
                for cond in ("", "_cond"):
                    _validate_reward_mode(f"coverage_{union}{metric}{cond}")

    def test_rejects_modes_that_would_silently_train_as_em(self):
        for mode in ("additive", "coverage_hits", "coverage_cond", "EM", ""):
            with self.assertRaises(ValueError):
                _validate_reward_mode(mode)


class AnswerInTextTest(unittest.TestCase):
    def test_case_insensitive(self):
        self.assertTrue(_answer_in_text("Paris is the capital of France", ["paris"]))
        self.assertTrue(_answer_in_text("it was PARIS", ["Paris"]))
        self.assertFalse(_answer_in_text("Berlin is in Germany", ["Paris", "Lyon"]))
        self.assertFalse(_answer_in_text("anything", []))  # no targets -> never a hit


class RecallPrecisionTest(unittest.TestCase):
    def test_basic(self):
        # call 1: 1 of 2 docs contains gold -> hit, precision 0.5 ; call 2: 0 of 1 -> miss, precision 0.0
        call_docs = [["Paris is the capital", "Berlin is in Germany"], ["London bridge"]]
        recall, precision, n = _recall_precision(call_docs, ["Paris"])
        self.assertEqual(n, 2)
        self.assertEqual(recall, 0.5)       # 1 of 2 calls hit
        self.assertEqual(precision, 0.25)   # (0.5 + 0.0) / 2

    def test_all_hit_multi_target(self):
        call_docs = [["doc about Röntgen", "Wilhelm Conrad Röntgen won"], ["Röntgen physics"]]
        recall, precision, _ = _recall_precision(call_docs, ["Wilhelm Conrad Röntgen", "Röntgen"])
        self.assertEqual(recall, 1.0)       # both calls surface a gold string
        self.assertEqual(precision, 1.0)    # (2/2 + 1/1) / 2

    def test_no_calls(self):
        self.assertEqual(_recall_precision([], ["x"]), (0.0, 0.0, 0))
        # no targets -> zero, but n_calls counted
        self.assertEqual(_recall_precision([["Paris"]], []), (0.0, 0.0, 1))

    def test_skips_empty_call(self):
        # an empty doc list (a call that returned nothing) counts toward n_calls but adds no precision (no div/0)
        recall, precision, n = _recall_precision([[], ["has Paris here"]], ["Paris"])
        self.assertEqual(n, 2)
        self.assertEqual(recall, 0.5)       # only the second call hits
        self.assertEqual(precision, 0.5)    # (0 skipped + 1.0) / 2


class UnionRecallPrecisionTest(unittest.TestCase):
    def test_basic(self):
        # union over 3 distinct docs, 1 contains gold -> recall binary 1.0, precision 1/3
        call_docs = [["Paris is the capital", "Berlin is in Germany"], ["London bridge"]]
        recall, precision = _union_recall_precision(call_docs, ["Paris"])
        self.assertEqual(recall, 1.0)
        self.assertEqual(precision, 1 / 3)

    def test_dedup(self):
        # the same doc retrieved by two calls is counted once (deduped on text)
        recall, precision = _union_recall_precision(
            [["has Paris", "distractor"], ["has Paris", "distractor"]], ["Paris"])
        self.assertEqual(recall, 1.0)
        self.assertEqual(precision, 0.5)    # {has Paris, distractor} -> 1 of 2 distinct docs

    def test_is_call_count_neutral(self):
        # adding calls that miss never lowers union recall (the whole point vs per-call recall)
        one_call = _union_recall_precision([["has Paris"]], ["Paris"])[0]
        three_calls = _union_recall_precision([["has Paris"], ["nope one"], ["nope two"]], ["Paris"])[0]
        self.assertEqual(one_call, 1.0)
        self.assertEqual(three_calls, 1.0)
        # per-call recall, by contrast, drops from 1.0 to 1/3 for the same trajectory
        self.assertEqual(_recall_precision([["has Paris"], ["nope one"], ["nope two"]], ["Paris"])[0], 1 / 3)

    def test_no_hit_or_empty(self):
        self.assertEqual(_union_recall_precision([["Berlin"], ["London"]], ["Paris"]), (0.0, 0.0))
        self.assertEqual(_union_recall_precision([], ["Paris"]), (0.0, 0.0))
        self.assertEqual(_union_recall_precision([["Paris"]], []), (0.0, 0.0))  # no targets


class CoverageBonusTest(unittest.TestCase):
    def test_non_coverage_modes_are_zero(self):
        for mode in ("em", "conditional", "conditional_multi"):
            self.assertEqual(_coverage(mode), 0.0)

    def test_unconditional_metric_selection(self):
        self.assertEqual(_coverage("coverage_hit"), 0.5 * 0.7)          # per-call recall
        self.assertEqual(_coverage("coverage_prec"), 0.5 * 0.3)         # per-call precision
        self.assertEqual(_coverage("coverage_union_hit"), 0.5 * 1.0)    # union recall
        self.assertEqual(_coverage("coverage_union_prec"), 0.5 * 0.4)   # union precision

    def test_unconditional_ignores_em(self):
        # without `_cond`, coverage is paid on a correct answer too -- it stacks on top of EM
        self.assertEqual(_coverage("coverage_hit", em=1.0), 0.5 * 0.7)
        self.assertEqual(_coverage("coverage_union_prec", em=1.0), 0.5 * 0.4)

    def test_conditional_gates_on_wrong_answer(self):
        # _cond: full metric when wrong, zero when correct (can never out-reward a correct answer)
        self.assertEqual(_coverage("coverage_hit_cond", em=0.0), 0.5 * 0.7)
        self.assertEqual(_coverage("coverage_hit_cond", em=1.0), 0.0)
        self.assertEqual(_coverage("coverage_union_prec_cond", em=0.0), 0.5 * 0.4)
        self.assertEqual(_coverage("coverage_union_prec_cond", em=1.0), 0.0)
        self.assertEqual(_coverage("coverage_union_hit_cond", em=1.0), 0.0)


class GroundingTest(unittest.TestCase):
    def test_answer_grounded(self):
        docs = [["Paris is the capital of France", "Berlin is in Germany"], ["London bridge"]]
        self.assertTrue(_answer_grounded("Paris", docs))      # answer appears in a retrieved doc
        self.assertTrue(_answer_grounded("london", docs))     # case-insensitive
        self.assertFalse(_answer_grounded("Madrid", docs))    # answer not retrieved
        self.assertFalse(_answer_grounded("", docs))          # empty answer -> not grounded
        self.assertFalse(_answer_grounded("Paris", []))       # no docs

    def test_bonus_only_for_grounded_correct(self):
        # fires only when mode==grounded AND correct AND grounded
        self.assertEqual(_grounding_bonus("grounded", em=1.0, grounded_scale=0.1, answer_grounded=True), 0.1)
        # parametric guess
        self.assertEqual(_grounding_bonus("grounded", em=1.0, grounded_scale=0.1, answer_grounded=False), 0.0)
        # gated on correct
        self.assertEqual(_grounding_bonus("grounded", em=0.0, grounded_scale=0.1, answer_grounded=True), 0.0)
        # wrong mode
        self.assertEqual(_grounding_bonus("em", em=1.0, grounded_scale=0.1, answer_grounded=True), 0.0)
        self.assertEqual(_grounding_bonus("coverage_hit", em=1.0, grounded_scale=0.1, answer_grounded=True), 0.0)


class FormatTest(unittest.TestCase):
    def test_rpl_and_sat_trajectories(self):
        rpl = "<think>a</think><code>x</code><output>y</output><think>b</think><answer>Paris</answer>"
        sat = "<think>a</think><search>q</search><information>d</information><answer>Paris</answer>"
        self.assertTrue(well_formed(rpl))
        self.assertTrue(well_formed(sat))

    def test_rejects_missing_think_or_answer(self):
        self.assertFalse(well_formed("<answer>Paris</answer>"))
        self.assertFalse(well_formed("<think>a</think>"))

    def test_rejects_an_answer_only_inside_think(self):
        self.assertFalse(well_formed("<think>maybe <answer>Paris</answer></think>"))

    def test_rejects_text_between_blocks(self):
        self.assertFalse(well_formed("<think>a</think> stray prose <answer>Paris</answer>"))


class EfficiencyBonusTest(unittest.TestCase):
    def test_off_by_default(self):
        self.assertEqual(_efficiency_bonus(False, 1.0, 0.0, P, 3), 0.0)

    def test_wrong_answers_get_nothing(self):
        # Efficiency must never be purchasable with accuracy.
        self.assertEqual(_efficiency(em=0.0, calls=1), 0.0)
        self.assertEqual(_efficiency(em=0.0, calls=99), 0.0)

    def test_validation_is_unshaped(self):
        self.assertEqual(_efficiency(em=1.0, calls=2, is_val=True), 0.0)

    def test_searching_once_beats_answering_blind(self):
        # The whole point of the min-call step: a correct closed-book guess must not win.
        self.assertEqual(1.0 + _efficiency(em=1.0, calls=0), 1.0)
        self.assertGreater(1.0 + _efficiency(em=1.0, calls=1), 1.0)

    def test_cheaper_correct_answers_score_higher(self):
        scores = [1.0 + _efficiency(em=1.0, calls=c) for c in (1, 2, 3, 5)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(scores[0] - scores[1], P)  # each extra call costs exactly the price

    def test_bonus_vanishes_at_the_implied_budget(self):
        # b/p calls is where searching stops being better than not searching.
        self.assertAlmostEqual(_efficiency(em=1.0, calls=int(B / P)), 0.0)

    def test_rejects_a_price_that_would_punish_the_first_call(self):
        with self.assertRaises(ValueError):
            _validate_efficiency_params(min_call_bonus=0.01, per_call_price=0.05)

    def test_accepts_sane_and_disabled_configs(self):
        _validate_efficiency_params(min_call_bonus=B, per_call_price=P)
        _validate_efficiency_params(min_call_bonus=0.0, per_call_price=P)  # disabled: price is irrelevant


class BillableCallsTest(unittest.TestCase):
    def test_sat_pays_for_each_search(self):
        # SaT has no sandbox counters; billing it 0 calls made a priced SaT run train on plain EM.
        self.assertEqual(_billable_calls(0, 0, 0, search_calls=2), 2)
        self.assertGreater(_efficiency(em=1.0, calls=_billable_calls(0, 0, 0, search_calls=1)), 0.0)

    def test_sandbox_counters_win_over_search_actions(self):
        self.assertEqual(_billable_calls(2, 1, 1, search_calls=1), 4)

    def test_no_search_and_no_counters_is_free(self):
        self.assertEqual(_billable_calls(0, 0, 0, search_calls=0), 0)
        self.assertEqual(_billable_calls(0, 0, 0, search_calls=None), 0)
