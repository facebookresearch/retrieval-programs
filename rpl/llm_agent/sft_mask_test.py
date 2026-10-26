# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The SFT loss mask must line up with the token stream RL actually rolls out."""

import pathlib
import re
import unittest

from rpl.llm_agent.sft_mask import (
    INJECTED_RE, SPAN_CODE, SPAN_INJECTED, SPAN_PROSE, assemble,
    classify_spans, span_mask, split_completion,
    stitch_invalid_actions,
)

# prompt = 3 tokens; response = code(2, generated) + observation(3, injected) + answer(2, generated)
PROMPT = [11, 12, 13]
RESPONSE = [21, 22, 31, 32, 33, 41, 42]
MASK = [1, 1, 0, 0, 0, 1, 1]


class AssembleTest(unittest.TestCase):
    def test_input_ids_are_prompt_then_response(self):
        ids, _ = assemble(PROMPT, RESPONSE, MASK)
        self.assertEqual(ids, PROMPT + RESPONSE)

    def test_mask_length_matches_input_ids(self):
        ids, mask = assemble(PROMPT, RESPONSE, MASK)
        self.assertEqual(len(ids), len(mask))

    def test_the_prompt_is_never_supervised(self):
        _, mask = assemble(PROMPT, RESPONSE, MASK)
        self.assertEqual(mask[: len(PROMPT)], [0, 0, 0])

    def test_the_mask_is_not_pre_shifted(self):
        """Selecting by the mask must return exactly the generated tokens. A one-off shift returns
        neighbours -- [13, 21, 33, 41] instead -- which no runtime check would catch."""
        ids, mask = assemble(PROMPT, RESPONSE, MASK)
        self.assertEqual([t for t, m in zip(ids, mask) if m], [21, 22, 41, 42])

    def test_the_program_is_supervised_and_the_observation_is_not(self):
        ids, mask = assemble(PROMPT, RESPONSE, MASK)
        by_id = dict(zip(ids, mask))
        for tok in (21, 22):
            self.assertEqual(by_id[tok], 1, "the repaired <code> block must carry gradient")
        for tok in (31, 32, 33):
            self.assertEqual(by_id[tok], 0, "an injected observation must not be supervised")

    def test_answer_only_supervision_is_distinguishable(self):
        """Guards the regression directly: had the builder masked the program off, the supervised span
        would be the answer alone. This test would then fail rather than the run looking fine."""
        _, mask = assemble(PROMPT, RESPONSE, [0, 0, 0, 0, 0, 1, 1])
        ids, good = assemble(PROMPT, RESPONSE, MASK)
        self.assertNotEqual(mask, good)
        self.assertEqual(sum(mask), 2)
        self.assertEqual(sum(good), 4)

    def test_a_mismatched_mask_length_raises(self):
        with self.assertRaisesRegex(ValueError, "response_mask"):
            assemble(PROMPT, RESPONSE, [1, 1])

    def test_an_all_zero_mask_raises(self):
        with self.assertRaisesRegex(ValueError, "all zeros"):
            assemble(PROMPT, RESPONSE, [0] * len(RESPONSE))

    def test_an_empty_prompt_is_allowed(self):
        """A record whose prefix was entirely the repaired turn has no earlier-turn head."""
        ids, mask = assemble([], RESPONSE, MASK)
        self.assertEqual(ids, RESPONSE)
        self.assertEqual(mask, MASK)

    def test_the_inputs_are_not_mutated(self):
        p, r, m = list(PROMPT), list(RESPONSE), list(MASK)
        assemble(p, r, m)
        self.assertEqual((p, r, m), (PROMPT, RESPONSE, MASK))


class SplitCompletionTest(unittest.TestCase):
    def test_the_answer_starts_at_the_tag(self):
        reasoning, answer = split_completion("I can see X.\n\n<answer>tempo</answer>")
        self.assertEqual(reasoning, "I can see X.\n\n")
        self.assertEqual(answer, "<answer>tempo</answer>")

    def test_a_completion_that_never_answered_is_all_reasoning(self):
        reasoning, answer = split_completion("I should search again.\n<code>x</code>")
        self.assertEqual(reasoning, "I should search again.\n<code>x</code>")
        self.assertEqual(answer, "")

    def test_the_last_answer_wins(self):
        _, answer = split_completion("<answer>first</answer> no wait <answer>second</answer>")
        self.assertEqual(answer, "<answer>second</answer>")


# An earlier turn: 2 generated tokens (its reasoning + <code>), then 3 injected observation tokens.
# Then the repaired program (3), its observation (4), the final reasoning (5) and the answer (2).
HEAD = [(2, SPAN_CODE), (3, SPAN_INJECTED)]
SPANS = dict(head=HEAD, code=3, obs=4, reasoning=5, answer=2)
HEAD_LEN = 5


class SpanMaskTest(unittest.TestCase):
    def test_rl_supervises_the_policys_earlier_turns_but_not_the_injected_observation(self):
        """The fix: `response_mask` semantics -- 1 for everything generated, 0 only for injected text."""
        m = span_mask(**SPANS, mode="rl", answer_trusted=True)
        self.assertEqual(m, [1, 1] + [0, 0, 0] + [1, 1, 1] + [0, 0, 0, 0] + [1] * 5 + [1, 1])

    def test_rl_on_an_untrusted_row_degrades_to_program_only(self):
        """Nothing in a failed trajectory is validated except the harvest program, so nothing else is taught."""
        m = span_mask(**SPANS, mode="rl", answer_trusted=False)
        self.assertEqual(m, span_mask(**SPANS, mode="program", answer_trusted=False))
        self.assertEqual(sum(m), 3)

    def test_program_mode_drops_the_reasoning_but_keeps_a_trusted_answer(self):
        m = span_mask(**SPANS, mode="program", answer_trusted=True)
        self.assertEqual(m, [0, 0] + [0, 0, 0] + [1, 1, 1] + [0, 0, 0, 0] + [0] * 5 + [1, 1])

    def test_program_mode_on_an_untrusted_row_supervises_only_the_program(self):
        """The point of the relaxation: a harvest row still teaches its program without asserting its answer."""
        m = span_mask(**SPANS, mode="program", answer_trusted=False)
        self.assertEqual(m, [0] * 5 + [1, 1, 1] + [0, 0, 0, 0] + [0] * 5 + [0, 0])
        self.assertEqual(sum(m), 3)

    def test_the_injected_observation_is_never_supervised_in_any_mode(self):
        for mode, trusted in (("rl", True), ("rl", False), ("program", True), ("program", False)):
            m = span_mask(**SPANS, mode=mode, answer_trusted=trusted)
            self.assertEqual(m[HEAD_LEN + 3:HEAD_LEN + 7], [0] * 4, f"{mode=} supervised injected evidence")
            self.assertEqual(m[2:5], [0, 0, 0], f"{mode=} supervised an injected head observation")

    def test_the_repaired_program_is_supervised_in_every_mode(self):
        for mode, trusted in (("rl", True), ("rl", False), ("program", True), ("program", False)):
            m = span_mask(**SPANS, mode=mode, answer_trusted=trusted)
            self.assertEqual(m[HEAD_LEN:HEAD_LEN + 3], [1, 1, 1],
                             f"{mode=} dropped the program, which is the objective")

    def test_the_dropped_ablation_modes_are_refused(self):
        for mode in ("full", "actions", "narrate"):
            with self.assertRaisesRegex(ValueError, "mode must be one of"):
                span_mask(**SPANS, mode=mode, answer_trusted=True)

    def test_an_unknown_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            span_mask(**SPANS, mode="everything", answer_trusted=True)

    def test_the_mask_length_is_the_total_span_length(self):
        m = span_mask(**SPANS, mode="program", answer_trusted=True)
        self.assertEqual(len(m), HEAD_LEN + 3 + 4 + 5 + 2)

    def test_a_missing_answer_span_contributes_nothing(self):
        m = span_mask(head=[], code=2, obs=1, reasoning=3, answer=0, mode="program", answer_trusted=True)
        self.assertEqual(m, [1, 1, 0, 0, 0, 0])


class ClassifySpansTest(unittest.TestCase):
    def test_prose_code_and_injected_are_told_apart(self):
        spans = classify_spans("<think>a</think>\n<code>c</code>\n\n<output>Doc 1(Title: T) x</output>\n\nnext")
        self.assertEqual([k for _, k in spans], [SPAN_PROSE, SPAN_CODE, SPAN_INJECTED, SPAN_PROSE])
        self.assertEqual(spans[1][0], "<code>c</code>")
        self.assertEqual(spans[2][0], "\n\n<output>Doc 1(Title: T) x</output>\n\n")

    def test_the_spans_reconstruct_the_input_exactly(self):
        text = "a\n\n<output>one</output>\n\nb<code>p</code>\n\n<output>two</output>\n\nc"
        self.assertEqual("".join(s for s, _ in classify_spans(text)), text)

    def test_surrounding_newlines_belong_to_the_injected_span(self):
        """The loop injects `"\\n\\n<output>...</output>\\n\\n"` as ONE unit, so RL zeroes those newlines."""
        spans = classify_spans("x\n\n<output>o</output>\n\ny")
        self.assertEqual(spans, [("x", SPAN_PROSE), ("\n\n<output>o</output>\n\n", SPAN_INJECTED),
                                 ("y", SPAN_PROSE)])

    def test_the_invalid_action_nudge_is_injected_despite_looking_like_reasoning(self):
        """It is `<think>`-wrapped but written by the LOOP; supervising it teaches the model to emit it."""
        text = "ok<think>My previous action is invalid. Let me try again.</think>\nmore"
        self.assertEqual([k for _, k in classify_spans(text)], [SPAN_PROSE, SPAN_INJECTED, SPAN_PROSE])

    def test_a_genuine_think_block_is_prose_not_injected(self):
        spans = classify_spans("<think>I should search for the author.</think><code>c</code>")
        self.assertEqual(spans, [("<think>I should search for the author.</think>", SPAN_PROSE),
                                 ("<code>c</code>", SPAN_CODE)])

    def test_a_code_block_inside_an_observation_is_not_mistaken_for_an_action(self):
        """Injection is resolved first, so a retrieved document quoting `<code>` cannot become supervised."""
        spans = classify_spans("a\n\n<output>Doc 1(Title: T) see <code>x</code> here</output>\n\nb")
        self.assertEqual([k for _, k in spans], [SPAN_PROSE, SPAN_INJECTED, SPAN_PROSE])

    def test_text_with_no_injection_is_one_prose_span(self):
        self.assertEqual(classify_spans("just reasoning"), [("just reasoning", SPAN_PROSE)])

    def test_empty_text_yields_no_spans(self):
        self.assertEqual(classify_spans(""), [])


# A fuller head: prose, the policy's own earlier <code> block, then its injected observation.
HEAD3 = [(2, SPAN_PROSE), (3, SPAN_CODE), (4, SPAN_INJECTED)]
SPANS3 = dict(head=HEAD3, code=3, obs=2, reasoning=5, answer=2)


class FullerHeadTest(unittest.TestCase):
    def test_rl_supervises_earlier_prose_and_code_but_not_the_injected_observation(self):
        m = span_mask(**SPANS3, mode="rl", answer_trusted=True)
        self.assertEqual(m, [1, 1] + [1, 1, 1] + [0, 0, 0, 0] + [1, 1, 1] + [0, 0] + [1] * 5 + [1, 1])

    def test_program_leaves_the_whole_head_unsupervised(self):
        m = span_mask(**SPANS3, mode="program", answer_trusted=True)
        self.assertEqual(m, [0] * 9 + [1, 1, 1] + [0, 0] + [0] * 5 + [1, 1])

    def test_rl_on_an_untrusted_row_degrades_to_program_only(self):
        m = span_mask(**SPANS3, mode="rl", answer_trusted=False)
        self.assertEqual(m, span_mask(**SPANS3, mode="program", answer_trusted=False))

    def test_an_unknown_head_kind_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown head span kind"):
            span_mask(head=[(2, "mystery")], code=1, obs=1, reasoning=1, answer=1,
                      mode="rl", answer_trusted=True)


class StitchInvalidActionsTest(unittest.TestCase):
    NUDGE = ("<think>My previous action is invalid. If I want to search, I should write Python between "
             "<code> and </code> that uses the search SDK and prints the results. If I want to give the "
             "final answer, I should put it between <answer> and </answer>. Let me try again.</think>")

    def test_both_generated_halves_survive(self):
        """The rejected generation is usually real reasoning; dropping it teaches answer-immediately."""
        text = f"From the documents, the Missoulian is third largest.\n{self.NUDGE}\n</think>\n\n<answer>Lee</answer>"
        out, n = stitch_invalid_actions(text)
        self.assertEqual(n, 1)
        self.assertIn("Missoulian is third largest", out)
        self.assertIn("<answer>Lee</answer>", out)

    def test_the_nudge_and_the_stray_close_tag_are_both_gone(self):
        text = f"reasoning\n{self.NUDGE}\n</think>\n\n<answer>x</answer>"
        out, _ = stitch_invalid_actions(text)
        self.assertNotIn("previous action is invalid", out)
        self.assertEqual(out.count("</think>"), 0, "the retry's stray close-tag must go with the nudge")

    def test_a_retry_without_a_stray_tag_is_untouched_otherwise(self):
        text = f"reasoning\n{self.NUDGE}\n<code>c</code>"
        out, n = stitch_invalid_actions(text)
        self.assertEqual((n, out), (1, "reasoning\n\n<code>c</code>"))

    def test_an_empty_rejected_generation_leaves_observation_then_retry(self):
        """29.4% of nudges follow an EMPTY generation; those are pure excision."""
        text = f"<output>docs</output>{self.NUDGE}\n</think>\n\n<code>c</code>"
        out, _ = stitch_invalid_actions(text)
        self.assertEqual(out, "<output>docs</output>\n\n<code>c</code>")

    def test_the_premature_answer_nudge_is_also_stitched(self):
        text = ("thinking\n<think>I don't have enough evidence to answer yet. Let me search again before "
                "giving the final answer.</think>\n<code>c</code>")
        out, n = stitch_invalid_actions(text)
        self.assertEqual(n, 1)
        self.assertNotIn("enough evidence", out)

    def test_multiple_nudges_in_one_trajectory(self):
        text = f"a\n{self.NUDGE}\n</think>\nb\n{self.NUDGE}\n</think>\nc"
        out, n = stitch_invalid_actions(text)
        self.assertEqual(n, 2)
        self.assertEqual(out, "a\n\nb\n\nc")

    def test_a_genuine_think_block_is_not_touched(self):
        text = "<think>I should search for the author.</think>\n<code>c</code>"
        self.assertEqual(stitch_invalid_actions(text), (text, 0))

    def test_text_with_no_nudge_is_returned_unchanged(self):
        self.assertEqual(stitch_invalid_actions("plain"), ("plain", 0))

    def test_the_nudge_body_quoting_code_tags_does_not_end_the_match_early(self):
        """The nudge quotes `</code>` and `</answer>`; the match must stop at its own `</think>`."""
        out, n = stitch_invalid_actions(f"x{self.NUDGE}y")
        self.assertEqual(n, 1)
        self.assertNotIn("search SDK", out)


class AgentLoopCouplingTest(unittest.TestCase):
    """`INJECTED_RE` duplicates the loop's injected literals; this fails if the two drift apart.

    Importing the loop is not an option (torch/verl), and a missed injected form does not error -- it trains
    the model on text the loop wrote, which is invisible in the loss.
    """

    def setUp(self):
        self.src = (pathlib.Path(__file__).parent / "rpl_agent_loop.py").read_text()

    def test_every_injected_literal_in_the_loop_is_matched(self):
        # The loop builds observations with f-strings and defines the nudges as string constants; pull the
        # opening marker out of each and check INJECTED_RE claims a span that starts there.
        samples = {
            "observation": '\n\n<output>Doc 1(Title: T) body</output>\n\n',
            "information": '\n\n<information>Doc 1(Title: T) body</information>\n\n',
        }
        for name, literal in samples.items():
            self.assertTrue(INJECTED_RE.fullmatch(literal), f"{name} is no longer recognised as injected")
        for const in ("_INVALID_SAT", "_INVALID_SAC", "_INVALID_MARAG", "_PREMATURE_ANSWER"):
            m = re.search(rf'{const} = \(\s*(.*?)\)\n', self.src, re.DOTALL)
            self.assertIsNotNone(m, f"{const} no longer parses out of the loop source")
            text = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))).encode().decode("unicode_escape")
            self.assertTrue(INJECTED_RE.search(text),
                            f"{const} is injected by the loop but INJECTED_RE would mark it GENERATED")

    def test_the_loop_still_zeroes_observations_and_ones_generations(self):
        """If the loop's own contract flips, an 'RL-matched' mask stops matching RL."""
        self.assertIn("response_mask += [0] * len(obs_ids)", self.src)
        self.assertIn("response_mask += [1] * len(gen)", self.src)


if __name__ == "__main__":
    unittest.main()
