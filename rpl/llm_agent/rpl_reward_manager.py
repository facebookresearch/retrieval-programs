# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reward manager for search agents.

Computes an exact-match (EM) base reward with a configurable train-time reward shape, so validation
`val-core/.../acc` stays pure EM. All reward logic lives here in the reward layer; `verl/trainer/`
is untouched and `verl/` needs no registration edit -- this class is loaded by path:

    reward.reward_manager.source=importlib
    reward.reward_manager.module.path=pkg://rpl.llm_agent.rpl_reward_manager
    reward.reward_manager.name=RPLRewardManager

Reward modes (config `+rpl.reward_mode`, default `em`):
  - `em`          : pure exact-match.
  - `conditional` : 1.0 if correct, else `reward_b` (default 0.1) if a real search was made, else 0
                    ("search as a means" -- no reward for searching when already correct).
  - `conditional_multi` : like `conditional` but the wrong-answer bonus scales with the number of
                    retrieval calls -- `reward_b * min(#searches, reward_cap)` (default cap 3) -- to
                    encourage multi-hop search. Capped to limit search-spam.
  - `coverage_hit`  : EM + `coverage_scale` * answer-RECALL (fraction of retrieval calls whose docs contain a
                    gold answer string) -- makes retrieval recall reward-visible independently of answer
                    conversion. RPL only; train-only; val stays pure EM.
  - `coverage_prec` : like `coverage_hit` but answer-PRECISION (gold-containing docs / docs, averaged per call).
  - `coverage_union_hit` / `coverage_union_prec` : like the two above, but coverage is measured over the
                    DEDUPED UNION of all docs retrieved across the trajectory rather than averaged per call.
                    Union recall is binary (1.0 if any retrieved doc contains a gold) and union precision is
                    global (gold-containing / total distinct docs). Because the metric is call-count neutral --
                    an extra call can only add docs, never lower the score -- it does not push the policy toward
                    fewer/safer calls (covhit) or precision-farming over-search (covprec) the way the per-call
                    metrics do.
  - any `coverage_*` mode may carry a `_cond` suffix (e.g. `coverage_hit_cond`, `coverage_union_prec_cond`):
                    the coverage bonus is applied ONLY when the answer is wrong (em < 1) -- a graded "search as
                    a stepping stone" that anneals as accuracy rises and can never out-reward a correct answer
                    (bonus <= coverage_scale < 1.0). This is the graded analogue of `conditional`.
  - `grounded`   : EM + `grounded_scale` bonus (default 0.1) when the answer is CORRECT and appears verbatim in
                    a retrieved doc (grounded, not a parametric guess). Gated on em==1 -> tie-break within the
                    correct set (grounded-correct > ungrounded-correct > wrong); cannot lower accuracy.
  - `autorefine` : AutoRefine (arXiv:2505.11277), reproducing their `cmd/train.sh` (`reward_style=F1`,
                    `refine_score=0.1`, `format_score=0.0`): token F1 on the answer, cascading to a flat
                    `autorefine_refine_score` payout when F1 is 0 but the gold answer survives into the
                    concatenated `<refine>` blocks.
"""

from __future__ import annotations

import re

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score.search_r1_like_qa_em import compute_score as _em_score
from verl.utils.reward_score.search_r1_like_qa_em import (
    cover_em_check,
    extract_solution,
    f1_check,
    globalqa_em_check,
    globalqa_f1_check,
    searched,
    well_formed,
)


# The invalid action nudges injected by the agent loop contain the literal substring `<answer> and </answer>`
# ("...I should put it between <answer> and </answer>"). Any answer extractor scanning the response therefore
# finds a phantom answer of " and " -- and when a rollout ends without ever answering, that phantom is the
# LAST match and becomes the scored prediction.
#
# Sanitized here, in the reward layer, rather than by rewording the nudge.
# Keep this list in sync with the loop's `_INVALID_*` / `_PREMATURE_ANSWER` constants.
_PHANTOM_ANSWER = "<answer> and </answer>"


def _strip_injected_nudges(text: str) -> str:
    """Remove the phantom `<answer> and </answer>` the invalid action nudges contribute."""
    return text.replace(_PHANTOM_ANSWER, "<answer_tags>") if text else text


def _answer_in_text(text: str, targets: list) -> bool:
    """True if any gold answer string appears (case-insensitive) in `text` (standard answer-in-passage recall)."""
    t = text.lower()
    return any(g and str(g).lower() in t for g in targets)


def _recall_precision(call_docs: list, targets: list) -> tuple[float, float, int]:
    """Answer recall/precision over per-call retrieved doc texts vs gold `targets`.

    Returns (recall, precision, n_calls): recall = fraction of calls whose docs contain a gold answer;
    precision = mean over calls of (gold-containing docs / docs in that call). (0, 0, 0) when there were no calls.
    """
    n_calls = len(call_docs)
    if n_calls == 0 or not targets:
        return 0.0, 0.0, n_calls
    hit_calls = 0
    prec_sum = 0.0
    for docs in call_docs:
        if not docs:
            continue
        hits = sum(1 for d in docs if _answer_in_text(d, targets))
        if hits:
            hit_calls += 1
        prec_sum += hits / len(docs)
    return hit_calls / n_calls, prec_sum / n_calls, n_calls


def _set_f1(retrieved_ids, gold_ids) -> float:
    """F1 between a retrieved and a gold document-id set."""
    pred, gold = set(retrieved_ids), set(gold_ids)
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gold))


def _union_recall_precision(call_docs: list, targets: list) -> tuple[float, float]:
    """Answer recall/precision over the DEDUPED UNION of all retrieved docs (call-count independent).

    Unlike `_recall_precision` (which averages per call, so extra calls can only dilute the score), this
    flattens every call's docs into one deduped set: recall = 1.0 if any distinct doc contains a gold string
    else 0.0 (binary union coverage -- adding calls never lowers it); precision = (distinct gold-containing
    docs) / (distinct docs). (0.0, 0.0) when there were no docs or no targets. Docs are deduped on their text.
    """
    seen = {d for docs in call_docs or [] for d in (docs or [])}
    if not seen or not targets:
        return 0.0, 0.0
    hits = sum(1 for d in seen if _answer_in_text(d, targets))
    return (1.0 if hits else 0.0), hits / len(seen)


_PLAIN_MODES = ("em", "conditional", "conditional_multi", "grounded", "autorefine")
_COVERAGE_MODE = re.compile(r"coverage_(union_)?(hit|prec)(_cond)?")


def _validate_reward_mode(mode: str) -> None:
    """Reject a mode no code path handles. `_base_reward` scores anything it does not recognise as pure EM,
    so without this a typo or a retired mode trains as EM and looks like a legitimate run."""
    if mode not in _PLAIN_MODES and not _COVERAGE_MODE.fullmatch(mode):
        raise ValueError(f"unknown reward_mode {mode!r}: expected one of {', '.join(_PLAIN_MODES)}, "
                         "or coverage_[union_]{hit|prec}[_cond]")


def _coverage_bonus(mode: str, em: float, coverage_scale: float, answer_recall: float,
                    answer_precision: float, union_recall: float, union_precision: float) -> float:
    """Coverage-shaping bonus for a `coverage_*` train mode (0.0 for any non-coverage mode).

    Mode grammar: `coverage_[union_]{hit|prec}[_cond]`.
    * `union` -> deduped-union metric (call-count neutral); otherwise the per-call metric.
    * `prec` -> precision; otherwise recall ("hit").
    * `_cond` -> credited only when the answer is wrong (em < 1) -- a graded stepping stone that anneals.

    The bonus is `coverage_scale * metric` and is always <= coverage_scale.
    """
    if not mode.startswith("coverage_"):
        return 0.0
    if mode.endswith("_cond") and em >= 1.0:
        return 0.0
    if "union" in mode:
        metric = union_precision if "prec" in mode else union_recall
    else:
        metric = answer_precision if "prec" in mode else answer_recall
    return coverage_scale * metric


_REFINE_RE = re.compile(r"<refine>(.*?)</refine>", re.DOTALL)


def _refine_text(response_str: str) -> str:
    """Concatenation of every `<refine>` block, matching AutoRefine's `compute_refine_score_subem`.

    All blocks are joined before the containment check, so evidence distilled across several turns counts
    together -- the gold answer only has to survive somewhere in the refined text.
    """
    return " ".join(m.group(1) for m in _REFINE_RE.finditer(response_str)).strip()


def _answer_grounded(answer, call_docs: list) -> bool:
    """True if the emitted answer string appears verbatim in any retrieved doc (grounded, not parametric)."""
    if not answer:
        return False
    return any(_answer_in_text(d, [answer]) for docs in call_docs or [] for d in (docs or []))


def _grounding_bonus(mode: str, em: float, grounded_scale: float, answer_grounded: bool) -> float:
    """Grounding-conditioned-correctness bonus (0.0 unless mode == `grounded`).

    Rewards a CORRECT answer that is grounded in retrieved text (the answer appears in a retrieved doc) over
    a correct answer produced from parametric memory. Gated on `em >= 1` -- a pure tie-break WITHIN the
    correct set (grounded-correct > ungrounded-correct > wrong), so it can never lower accuracy; it only
    biases the policy toward evidence-grounded (and verbatim-extractive) answers.
    """
    if mode != "grounded":
        return 0.0
    return grounded_scale if (em >= 1.0 and answer_grounded) else 0.0


def _validate_efficiency_params(min_call_bonus: float, per_call_price: float) -> None:
    """Reject a price that would make the first call score below making none (see `_efficiency_bonus`)."""
    if 0.0 < min_call_bonus < per_call_price:
        raise ValueError(f"efficiency_min_call_bonus ({min_call_bonus}) must be >= efficiency_per_call_price "
                         f"({per_call_price}), else one call scores below zero calls and the policy is pushed "
                         "to answer without searching")


def _billable_calls(dense_calls: int, sparse_calls: int, rerank_calls: int, search_calls: int | None) -> int:
    """Outgoing API calls one rollout made: the sandbox's retrievals plus reranks (`fuse`/`dedup` are local).

    SaT has no sandbox, so those counters stay 0 and its cost is one retriever POST per `<search>`. Billing it 0
    made `_efficiency_bonus` pay nothing (it requires >= 1 call), so a priced SaT run silently trained on plain EM.
    """
    sandbox_calls = dense_calls + sparse_calls + rerank_calls
    return sandbox_calls if sandbox_calls else int(search_calls or 0)


def _efficiency_bonus(is_validation: bool, em: float, min_call_bonus: float, per_call_price: float,
                      api_calls: int) -> float:
    """Pay a CORRECT answer for using fewer outgoing API calls (0.0 unless `min_call_bonus > 0`).

    Cost is the count of calls that leave the sandbox -- `dense_retrieval`, `sparse_retrieval` and `rerank`, or
    one per `<search>` for SaT (`_billable_calls`). `fuse`/`dedup` are pure-python and free. The bonus is
    `min_call_bonus - per_call_price * api_calls`, and it applies only when `em >= 1` and at least one call was
    made, which gives two distinct knobs:

    * `min_call_bonus` -- the step for making ANY call, so a correct closed-book guess (1.0) never beats a
      correct searched answer (1 + min_call_bonus - per_call_price). Requires `min_call_bonus >= per_call_price`.
    * `per_call_price` -- the slope, i.e. the probability of flipping correctness that a marginal call must
      buy to be worth making. Values around 0.01-0.05 are sane; the bonus reaches 0 at
      `min_call_bonus / per_call_price` calls, beyond which searching is no better than not searching.

    Gating on correctness is what makes this safe: on a question where the extra call was needed, dropping it
    makes the answer wrong and EM dominates, so the bonus prunes effort that was unnecessary on that question.
    """
    if is_validation or min_call_bonus <= 0.0 or em < 1.0 or api_calls < 1:
        return 0.0
    return min_call_bonus - per_call_price * api_calls


class RPLRewardManager(RewardManagerBase):
    """EM reward with a configurable train-time shape.

    Usage: `reward.reward_manager.{source=importlib,name=RPLRewardManager,module.path=pkg://...}`
    plus `+rpl.reward_mode=<mode>` (default `em`). Validation always scores pure EM.
    """

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        # `compute_score` is intentionally unused: this manager is Search-R1-specific and calls the
        # QA-EM scorer + shaping directly. It stays in the signature for the reward-loop constructor.
        super().__init__(config, tokenizer, compute_score)
        # Reward config lives at the config top level (`+rpl.*`), read the same way the AgentLoop does.
        sr = self.config.get("rpl", None)
        self.reward_mode = str(sr.get("reward_mode", "em") if sr is not None else "em")
        _validate_reward_mode(self.reward_mode)
        self.reward_b = float(sr.get("reward_b", 0.1) if sr is not None else 0.1)
        # Cap for `conditional_multi`: max number of searches that earn the per-search bonus.
        self.reward_cap = int(sr.get("reward_cap", 3) if sr is not None else 3)
        # Optional train-only bonus for a well-formed response (see `well_formed`), added on top of every mode
        # except `autorefine`. Default 0 = off.
        self.format_bonus = float(sr.get("format_bonus", 0.0) if sr is not None else 0.0)
        # Optional complementary-search bonus (RPL routing arm, GRASP-style, 2607.10463): reward a trajectory
        # that used BOTH retrievers (>=1 dense AND >=1 sparse sandbox call). Default 0 = off.
        self.complementary_bonus = float(sr.get("complementary_bonus", 0.0) if sr is not None else 0.0)
        # Efficiency shaping (see `_efficiency_bonus`): pay a CORRECT answer for using fewer outgoing API
        # calls. `efficiency_min_call_bonus` is the step for making at least one call (0 = off, the default);
        # `efficiency_per_call_price` is the per-call slope.
        self.efficiency_min_call_bonus = float(sr.get("efficiency_min_call_bonus", 0.0) if sr is not None else 0.0)
        self.efficiency_per_call_price = float(sr.get("efficiency_per_call_price", 0.02) if sr is not None else 0.02)
        _validate_efficiency_params(self.efficiency_min_call_bonus, self.efficiency_per_call_price)
        # Weight for coverage shaping (`coverage_hit`/`coverage_prec`): reward = em + coverage_scale * (answer
        # recall | precision), making retrieval recall reward-visible independently of answer conversion. It also
        # keeps search alive (no recall without searching). Default 0.5.
        self.coverage_scale = float(sr.get("coverage_scale", 0.5) if sr is not None else 0.5)
        # GlobalQA only: weight on document-coverage F1 (retrieved vs gold doc ids) added to the train reward.
        self.globalqa_doc_reward = float(sr.get("globalqa_doc_reward", 0.0) if sr is not None else 0.0)
        # Weight for the `grounded` mode: bonus for a CORRECT answer that appears verbatim in a retrieved
        # doc (grounded, not a parametric guess). Default 0.1.
        self.grounded_scale = float(sr.get("grounded_scale", 0.1) if sr is not None else 0.1)
        # AutoRefine's `refine_score` (their cmd/train.sh passes 0.1).
        self.autorefine_refine_score = float(sr.get("autorefine_refine_score", 0.1) if sr is not None else 0.1)

    def _base_reward(self, em: float, response_str: str, search_calls, mode: str,
                     f1: float = 0.0, refine_covers: bool = False) -> float:
        """The mode's base reward for (EM, trajectory), before any bonus is added."""
        sc = int(search_calls) if search_calls is not None else None
        if mode == "autorefine":
            # Cascade, NOT a weighted sum: the refine term is consolation for a zero-scoring answer, so it
            # can never out-rank answering correctly.
            if f1 > 0.0:
                return f1
            return self.autorefine_refine_score if refine_covers else 0.0
        if mode == "conditional":
            # Full credit only for a correct answer; partial credit for making a real search keeps the
            # search behaviour alive early and anneals away as accuracy rises (search as a means).
            return 1.0 if em >= 1.0 else (self.reward_b if searched(response_str, sc) else 0.0)
        if mode == "conditional_multi":
            # Like `conditional`, but the wrong-answer bonus scales with #retrieval calls (capped) to
            # encourage multi-hop search. Uses the sandbox's ground-truth count `sc` when available.
            if em >= 1.0:
                return 1.0
            n = min(sc, self.reward_cap) if sc is not None else (1 if searched(response_str, None) else 0)
            return self.reward_b * n
        # `em`, and the coverage_* / `grounded` modes, whose shaping is added as a bonus in `run_single`.
        return em

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]  # last sequence only, matching the naive manager
        data_item = data[0]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        # Per-sample fields from the AgentLoop arrive nested under `tool_extra_fields` (the streaming
        # reward path packs each loop's `extra_fields` there -- see NaiveRewardManager), NOT as
        # top-level non_tensor_batch keys.
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None) or {}
        # RPL ground-truth retrieval count (per-sample, from the sandbox); None => SaT text fallback.
        search_calls = tool_extra_fields.get("code_search_calls", None)
        # Per-method retrieval counts for the complementary-search reward + routing analytics.
        dense_calls = int(tool_extra_fields.get("code_dense_calls", 0) or 0)
        sparse_calls = int(tool_extra_fields.get("code_sparse_calls", 0) or 0)
        # Validation is always pure EM.
        is_validation = bool(tool_extra_fields.get("is_validation", data.meta_info.get("validate", False)))
        # Complementary-search bonus: used BOTH retrievers this trajectory (train-only, gated; 0 = off).
        comp_bonus = (
            self.complementary_bonus
            if (not is_validation and self.complementary_bonus > 0 and dense_calls > 0 and sparse_calls > 0)
            else 0.0
        )

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        # Before any extraction: the injected nudges carry a phantom answer tag pair.
        response_str = _strip_injected_nudges(response_str)

        # GlobalQA (corpus-aggregation): the answer is a count/id-set aggregation of the gold docs, not a
        # free-text span -- score by integer-set F1. The wiki-18 EM path below does not apply. Gated on
        # data_source so every other dataset is unchanged.
        data_source = data_item.non_tensor_batch.get("data_source", "")
        # Echoed so a verify dump can be joined back PER REPAIR rather than per question. Numeric because
        # verl averages every reward_extra_info key into `val-aux/<ds>/<key>/mean@1`; the mean of an id is
        # meaningless but harmless, and a string would break the aggregation.
        _rid = (data_item.non_tensor_batch.get("extra_info") or {}).get("repair_id")
        rid = {"repair_id": float(_rid)} if _rid is not None else {}
        if data_source == "globalqa":
            answer = extract_solution(response_str)
            targets = ground_truth["target"] if isinstance(ground_truth, dict) else ground_truth
            gq_em = globalqa_em_check(answer, targets) if answer is not None else 0.0
            gq_f1 = globalqa_f1_check(answer, targets) if answer is not None else 0.0
            gold_docs = ground_truth.get("golden_doc_ids") if isinstance(ground_truth, dict) else None
            ids = [i for call in (tool_extra_fields.get("retrieved_doc_ids") or []) for i in call]
            extra = {
                **rid, "acc": gq_em, "em": gq_em, "f1": gq_f1, "format_bonus": 0.0,
                "token_f1": f1_check(answer, targets) if answer is not None else 0.0,
                "n_retrieved_docs": float(len(set(ids))),
                "dense_calls": dense_calls, "sparse_calls": sparse_calls,
                "complementary_bonus": comp_bonus,
            }
            if gold_docs:
                extra["doc_f1"] = _set_f1(ids, gold_docs)
                extra["doc_f1@20"] = _set_f1(ids[:20], gold_docs)
            doc_bonus = self.globalqa_doc_reward * extra["doc_f1"] if (
                self.globalqa_doc_reward > 0.0 and "doc_f1" in extra) else 0.0
            extra["doc_coverage_bonus"] = doc_bonus
            reward = gq_em if is_validation else gq_f1 + doc_bonus + comp_bonus
            return {"reward_score": reward, "reward_extra_info": extra}

        em = await self.loop.run_in_executor(None, lambda: _em_score(response_str, ground_truth))
        answer = extract_solution(response_str)
        targets = ground_truth["target"] if isinstance(ground_truth, dict) else ground_truth
        cover_em = cover_em_check(answer, targets) if answer is not None else 0
        f1 = f1_check(answer, targets) if answer is not None else 0.0

        # Answer recall/precision over the retrieved doc sets.
        retrieved_docs = tool_extra_fields.get("retrieved_docs") or []
        reranked_docs = tool_extra_fields.get("reranked_docs") or []
        answer_recall, answer_precision, _ = _recall_precision(retrieved_docs, targets)
        rerank_recall, rerank_precision, _ = _recall_precision(reranked_docs, targets)
        # Call-count-independent coverage over the deduped union of retrieved docs (see _union_recall_precision).
        # Eval metric for every mode; consumed by the reward gate only under the coverage_union_* modes.
        union_recall, union_precision = _union_recall_precision(retrieved_docs, targets)

        # Is the emitted answer grounded in retrieved text?
        answer_grounded = _answer_grounded(answer, retrieved_docs)

        # AutoRefine's distilled-evidence signal.
        refine_str = _refine_text(response_str)
        refine_present = 1.0 if refine_str else 0.0
        refine_covers = bool(refine_str) and bool(cover_em_check(refine_str, targets))

        mode = "em" if is_validation else self.reward_mode
        reward = self._base_reward(em, response_str, search_calls, mode, f1=f1, refine_covers=refine_covers)
        # Format bonus.
        format_bonus = (self.format_bonus if (not is_validation and mode != "autorefine" and self.format_bonus > 0
                                              and well_formed(response_str)) else 0.0)
        # Coverage shaping.
        coverage_bonus = _coverage_bonus(mode, em, self.coverage_scale, answer_recall, answer_precision,
                                                union_recall, union_precision)
        # Grounding-conditioned-correctness bonus (gated on em==1 so it never lowers accuracy).
        grounding_bonus = _grounding_bonus(mode, em, self.grounded_scale, answer_grounded)
        rerank_calls = int(tool_extra_fields.get("code_rerank_calls", 0) or 0)
        api_calls = _billable_calls(dense_calls, sparse_calls, rerank_calls, search_calls)
        efficiency_bonus = _efficiency_bonus(is_validation, em, self.efficiency_min_call_bonus,
                                                    self.efficiency_per_call_price, api_calls)
        reward += format_bonus + coverage_bonus + comp_bonus + grounding_bonus + efficiency_bonus

        return {
            "reward_score": reward,
            "reward_extra_info": {
                **rid, "acc": em, "em": em, "cover_em": cover_em, "f1": f1,
                "format_bonus": format_bonus,
                "dense_calls": dense_calls, "sparse_calls": sparse_calls,
                "complementary_bonus": comp_bonus, "coverage_bonus": coverage_bonus,
                "grounding_bonus": grounding_bonus, "answer_grounded": float(answer_grounded),
                "answer_recall": answer_recall, "answer_precision": answer_precision,
                "union_recall": union_recall, "union_precision": union_precision,
                "rerank_recall": rerank_recall, "rerank_precision": rerank_precision,
                "refine_present": refine_present, "refine_covers": float(refine_covers),
                "rerank_calls": float(rerank_calls), "api_calls": float(api_calls),
                "api_calls_correct": float(api_calls) * em, "efficiency_bonus": efficiency_bonus,
            },
        }
