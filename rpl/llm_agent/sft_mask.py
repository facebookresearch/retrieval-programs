# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Assemble one SFT record's token stream and loss mask.

This is the one piece of the SFT path that fails silently when wrong: a misaligned or mis-scoped mask
trains on the wrong tokens and still produces a plausible loss curve. It lives here, importing nothing, so
it is testable with no third-party dependencies.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

ANSWER_OPEN = "<answer>"

# What each supervision mode puts gradient on. The split exists because harvest and verify license different
# spans: harvest validates the program (it retrieved the gold), verify validates the answer (the policy
# reached it unaided). `program` is program mutation's mask; `rl` is textbook RFT, for the RFT control.
SUPERVISION_MODES = ("program", "rl")

# What an earlier-turn span is.
SPAN_CODE, SPAN_PROSE, SPAN_INJECTED = "code", "prose", "injected"

# Everything `rpl_agent_loop` injects into the response stream.
INJECTED_RE = re.compile(
    r"\n*(?:"
    r"<output>.*?</output>"
    r"|<information>.*?</information>"
    r"|<think>My previous action is invalid\..*?</think>"
    r"|<think>I don't have enough evidence to answer yet\..*?</think>"
    r")\n*",
    re.DOTALL,
)


CODE_BLOCK_RE = re.compile(r"<code>.*?</code>", re.DOTALL)

# One injected nudge plus the stray `</think>` the retry opens with. Surrounding whitespace is consumed so
# the two generated halves join cleanly. `.*?</think>` stops at the nudge's own closing tag -- the nudge body
# quotes `<code>` / `</code>` / `<answer>`, so a character-class bound would not work.
NUDGE_STITCH_RE = re.compile(
    r"\s*<think>(?:My previous action is invalid|I don't have enough evidence to answer yet)\."
    r".*?</think>\s*(?:</think>)?\s*",
    re.DOTALL,
)


def stitch_invalid_actions(text: str) -> tuple[str, int]:
    """Drop the loop's injected nudges, returning `(stitched, n_removed)`."""
    stitched, n = NUDGE_STITCH_RE.subn("\n\n", text or "")
    return stitched, n


def classify_spans(text: str) -> list[tuple[str, str]]:
    """`text` as `(span, kind)` with kind one of `SPAN_CODE` / `SPAN_PROSE` / `SPAN_INJECTED`.

    Concatenating the spans reproduces `text` exactly; empty spans are dropped so token counts stay honest.
    Injection is resolved first, because the invalid-action nudge is `<think>`-wrapped and would otherwise be
    classified as the policy's own prose.
    """
    out: list[tuple[str, str]] = []

    def _generated(chunk: str) -> None:
        pos = 0
        for m in CODE_BLOCK_RE.finditer(chunk):
            if m.start() > pos:
                out.append((chunk[pos:m.start()], SPAN_PROSE))
            out.append((m.group(0), SPAN_CODE))
            pos = m.end()
        if pos < len(chunk):
            out.append((chunk[pos:], SPAN_PROSE))

    pos = 0
    for m in INJECTED_RE.finditer(text or ""):
        if m.start() > pos:
            _generated(text[pos:m.start()])
        out.append((m.group(0), SPAN_INJECTED))
        pos = m.end()
    if pos < len(text or ""):
        _generated(text[pos:])
    return out


def split_completion(completion: str) -> tuple[str, str]:
    """Split a generated completion into (reasoning, answer), the answer starting at the last `<answer>`."""
    i = (completion or "").rfind(ANSWER_OPEN)
    return ((completion, "") if i < 0 else (completion[:i], completion[i:]))


def span_mask(head: Sequence[tuple[int, bool]], code: int, obs: int, reasoning: int, answer: int, mode: str,
              answer_trusted: bool) -> list[int]:
    """The response-half loss mask for one record, from its span lengths.

    `head` is the rollout's earlier turns as `(n_tokens, kind)` pairs from `classify_spans` -- the policy's
    `<code>` blocks and prose interleaved with the observations the loop injected. Two spans are fixed
    across every mode: an injected observation is never supervised (neither `obs` nor an injected head span
    was produced by the policy), and `code` -- the repaired program -- always is.

    Modes:
    * `program` -- supervise the program everywhere, and the answer only where verify passed. Reasoning and
      earlier turns never carry gradient. This is what lets every harvest row contribute.
    * `rl` -- the RL contract. Every span the policy generated carries gradient and only injected text is
      zeroed, which is exactly `AgentLoopOutput.response_mask`, plus the repaired program. On a row whose answer
      is untrusted only the harvest-validated program is licensed, so it degrades to `program`.

    Raises:
        ValueError: on an unknown mode.
    """
    if mode not in SUPERVISION_MODES:
        raise ValueError(f"mode must be one of {SUPERVISION_MODES}, got {mode!r}")
    whole = mode == "rl" and answer_trusted     # supervise the whole generated trajectory
    sup_reasoning, sup_answer = int(whole), int(answer_trusted)
    # An injected span never carries gradient, in either mode.
    sup_by_kind = {SPAN_INJECTED: 0, SPAN_CODE: int(whole), SPAN_PROSE: int(whole)}
    out: list[int] = []
    for n_tokens, kind in head:
        if kind not in sup_by_kind:
            raise ValueError(f"unknown head span kind {kind!r}; expected one of {sorted(sup_by_kind)}")
        out += [sup_by_kind[kind]] * n_tokens
    return out + [1] * code + [0] * obs + [sup_reasoning] * reasoning + [sup_answer] * answer


def assemble(prompt_ids: list[int], response_ids: list[int],
             response_mask: list[int]) -> tuple[list[int], list[int]]:
    """Concatenate a record into (input_ids, loss_mask).

    `loss_mask` is in `input_ids` coordinates and marks the tokens to be predicted. It is NOT pre-shifted:
    verl's `sft_loss` applies `torch.roll(loss_mask, -1)` to align it with `log_prob`, so shifting here as
    well would supervise each token using the position before it.

    The prompt is always 0 -- it is given at inference and never generated. The response half carries the
    rollout's own mask (1 = the policy generated it, 0 = an injected observation), which is the same
    `response_mask` contract as `AgentLoopOutput`.

    Raises:
        ValueError: if the mask length disagrees with the response, or nothing is supervised. Both are
            build errors upstream, and both would otherwise show up as a quietly diluted loss.
    """
    if len(response_ids) != len(response_mask):
        raise ValueError(f"response_ids is {len(response_ids)} tokens but response_mask is {len(response_mask)}")
    if not any(response_mask):
        raise ValueError("response_mask is all zeros, so this record has no supervised tokens")
    return list(prompt_ids) + list(response_ids), [0] * len(prompt_ids) + list(response_mask)
