# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Assemble the program-mutation SFT set from verify rollouts, and emit it in the RL token-stream format.

Which rows, and which spans (`--supervise`):
* `program` -- the default, and program mutation's mask. Reasoning and earlier turns are never supervised and the
  answer only on verify-passing rows, so every harvest row contributes its program. Sound because harvest already
  validated the program (it retrieved the gold) independently of whether the policy then converted it.
* `rl` -- rejection-sampling fine-tuning, for the RFT control: the `AgentLoopOutput.response_mask` contract, 1
  for everything the policy generated and 0 only for injected text.

Usage:
    python scripts/program_mutation/mutation_sft.py --verify_parquet verify.parquet \
        --verify_rollouts <eval>/val_rollouts/<step>.jsonl --out sft.jsonl
    python scripts/program_mutation/mutation_sft.py --rft_dump merged_window.jsonl --train_parquet train.parquet \
        --rft_rows 461 --out rft.jsonl
    python scripts/program_mutation/mutation_sft.py --records sft.jsonl --tokenize --model <hf> \
        --supervise program --prompt_format apply --out sft.parquet
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

from mutation_screen import qkey

# Python puts this script's own directory on sys.path, not the repo root, and the tokenize step runs as a
# bare `python scripts/program_mutation/mutation_sft.py` with no PYTHONPATH. Add the root so the mask logic
# can be shared with the dataset class instead of duplicated.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rpl.llm_agent.sft_mask import (  # noqa: E402
    SUPERVISION_MODES,
    classify_spans,
    span_mask,
    split_completion,
    stitch_invalid_actions,
)

# The prefix ends with the repaired program's observation; everything after it is what the policy generated.
OBS_CLOSE = "</output>"


def continuation(output: str) -> str:
    """The generated span, split on the last `</output>`."""
    i = (output or "").rfind(OBS_CLOSE)
    return (output[i + len(OBS_CLOSE):] if i >= 0 else (output or "")).strip()


def split_prefix(prefix: str) -> tuple[str, str, str]:
    """Split a verify prefix into (earlier turns, the repaired `<code>` block, its observation).

    The repaired program is the last code block: verify replaces the rollout's final retrieval turn and
    appends the evidence that program retrieved. Returning it separately is what lets the SFT mask supervise
    the program rather than only the answer.
    """
    i = (prefix or "").rfind("<code>")
    if i < 0:
        return prefix or "", "", ""
    j = prefix.find("</code>", i)
    if j < 0:
        return prefix, "", ""
    j += len("</code>")
    return prefix[:i], prefix[i:j], prefix[j:]


def build_records(verify_parquet: str, verify_rollouts: str, out: str, keep_all: bool = True,
                  correct_metric: str = "cover_em", stitch: bool = True, min_acc: float = 0.5) -> None:
    """Join verify rollouts back to their prefixes and emit one record per usable trajectory.

    `correct_metric` decides which rows count as verify passes, i.e. whose answer may be supervised.
    `cover_em` is the default and roughly doubles the set because strict EM rejects answers that are right:
    gold "during the 1980s" against the model's "1980s", where the reasoning was correct.

    With `keep_all`, rows that did not pass are kept too, tagged `correct: false`. Their answer is never
    supervised; they exist so `program` can learn their harvest-validated program.

    `stitch` removes the loop's invalid-action nudges and the stray `</think>` the retry opens with (see
    `stitch_invalid_actions`). It is a no-op on a late-training harvest, but an early-training window can be
    ~10% nudges, and any head-supervising mode would otherwise train the policy's rejected generation.
    """
    df = pd.read_parquet(verify_parquet)
    # Keyed by `repair_id` when verify stamped one, else by question. A question may carry several valid
    # fixes, and keying those by question keeps only the last.
    prefix_by_q, prompt_by_q = {}, {}
    for i in range(len(df)):
        row = df.iloc[i]
        content = [dict(m) for m in list(row["prompt"])][-1]["content"]
        ei = row["extra_info"] or {}
        k = ei.get("repair_id")
        k = float(k) if k is not None else qkey(content)
        prefix_by_q[k] = ei.get("response_prefix")
        prompt_by_q[k] = content
    n_ids = sum(1 for k in prefix_by_q if isinstance(k, float))
    if n_ids:
        print(f"  joining on repair_id ({n_ids} of {len(df)} rows carry one)")

    counts = {"unjoined": 0, "no_prefix": 0, "incorrect": 0, "empty_completion": 0,
              "kept": 0, "kept_incorrect": 0}
    alt_pass = {"em": 0, "cover_em": 0}
    stitched = stitched_rows = 0
    records = []
    for line in open(verify_rollouts):
        if not line.strip():
            continue
        r = json.loads(line)
        q = qkey(r.get("input"))
        # The dump echoes `repair_id` when verify stamped one (see rpl_reward_manager); fall back to
        # the question so single-fix runs are byte-identical to before.
        k = float(r["repair_id"]) if r.get("repair_id") is not None else q
        prefix, prompt = prefix_by_q.get(k), prompt_by_q.get(k)
        if prefix is None or prompt is None:
            counts["unjoined"] += 1
            continue
        if not prefix:
            counts["no_prefix"] += 1
            continue
        if correct_metric not in r:
            sys.exit(f"ABORT: {verify_rollouts} has no {correct_metric!r} field (has: {sorted(r)[:12]}...). "
                     f"Defaulting it to 0 would read as a verify pass rate of zero.")
        correct = float(r[correct_metric] or 0) > min_acc
        for m in ("em", "cover_em"):
            if m in r and float(r[m] or 0) > min_acc:
                alt_pass[m] += 1
        if not correct and not keep_all:
            counts["incorrect"] += 1
            continue
        cont = continuation(r.get("output") or "")
        if not cont:
            counts["empty_completion"] += 1
            continue
        if stitch:
            prefix, n_pre = stitch_invalid_actions(prefix)
            cont, n_cont = stitch_invalid_actions(cont)
            stitched += n_pre + n_cont
            stitched_rows += bool(n_pre or n_cont)
        counts["kept" if correct else "kept_incorrect"] += 1
        records.append({"question": q, "prompt": prompt, "prefix": prefix, "completion": cont,
                        "correct": correct})

    # Every row of an eval over the verify set joins back to it, so a miss means the wrong file.
    if counts["unjoined"]:
        sys.exit(f"ABORT: {counts['unjoined']} of {sum(counts.values())} rows in {verify_rollouts} match no row of "
                 f"{verify_parquet}; --verify_rollouts must be the eval of exactly that verify set.")
    with open(out, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    total = sum(counts.values())
    print(f"of {total} verify rollouts (verify pass judged by {correct_metric}):")
    for k, v in counts.items():
        print(f"  {k:<18}{v:>5}{v / max(1, total):>8.1%}")
    print("  verify passes by metric: " + ", ".join(f"{m}={n}" for m, n in alt_pass.items()))
    if stitch:
        kept = counts["kept"] + counts["kept_incorrect"]
        print(f"  invalid-action nudges stitched out: {stitched} across {stitched_rows} rows "
              f"({stitched_rows / max(1, kept):.1%} of kept)")
    print(f"-> {out}")
    if counts["kept"] + counts["kept_incorrect"] == 0:
        sys.exit("ABORT: no trajectories to train on.")
    if counts["kept"] == 0:
        sys.exit("ABORT: no verify-passing trajectories; every record would supervise the program alone.")


def build_rft_records(dump: str, train_parquet: str, out: str, n_rows: int, correct_field: str = "em",
                      seed: int = 0, stitch: bool = True) -> None:
    """Emit the RFT control: the policy's own correct trajectories from a window, one drawn per question.

    The prefix ends at the last program's observation, so `split_prefix` returns the policy's own last program
    where a PM record carries the repaired one. A trajectory that never ran a program has no such span and is
    skipped, as is one with no final answer. `n_rows` is the size of the PM set being matched.

    `correct_field` must be strictly 0/1. Dumps predating the `em` field carry only `score`, which equals EM
    under the plain EM reward and is something else under any shaped reward -- so a non-binary value aborts
    instead of being thresholded into a selection that is not EM.
    """
    df = pd.read_parquet(train_parquet)
    prompt_by_q = {}
    for i in range(len(df)):
        content = [dict(m) for m in list(df.iloc[i]["prompt"])][-1]["content"]
        prompt_by_q[qkey(content)] = content
    counts = {"incorrect": 0, "no_program": 0, "no_answer": 0, "unjoined": 0, "kept": 0}
    by_q: dict[str, list[dict]] = {}
    stitched = 0
    for line in open(dump):
        if not line.strip():
            continue
        r = json.loads(line)
        if correct_field not in r:
            sys.exit(f"ABORT: {dump} has no {correct_field!r} field (has: {sorted(r)[:12]}...)")
        value = float(r[correct_field] or 0)
        if value not in (0.0, 1.0):
            sys.exit(f"ABORT: {correct_field}={value} is not 0/1, so it is not an EM signal in this dump")
        if value < 1.0:
            counts["incorrect"] += 1
            continue
        output = r.get("output") or ""
        i = output.rfind(OBS_CLOSE)
        if i < 0 or output.rfind("<code>", 0, i) < 0:
            counts["no_program"] += 1
            continue
        prefix, cont = output[: i + len(OBS_CLOSE)], continuation(output)
        if not split_completion(cont)[1]:
            counts["no_answer"] += 1
            continue
        q = qkey(r.get("input"))
        if q not in prompt_by_q:
            counts["unjoined"] += 1
            continue
        if stitch:
            prefix, n_pre = stitch_invalid_actions(prefix)
            cont, n_cont = stitch_invalid_actions(cont)
            stitched += n_pre + n_cont
        counts["kept"] += 1
        by_q.setdefault(q, []).append({"question": q, "prompt": prompt_by_q[q], "prefix": prefix,
                                       "completion": cont, "correct": True, "step": r.get("step")})
    total = sum(counts.values())
    print(f"of {total} window rollouts (correct = {correct_field} == 1):")
    for k, v in counts.items():
        print(f"  {k:<12}{v:>6}{v / max(1, total):>8.1%}")
    print(f"  questions with a usable correct trajectory: {len(by_q)}; nudges stitched out: {stitched}")
    if len(by_q) < n_rows:
        sys.exit(f"ABORT: only {len(by_q)} questions qualify but {n_rows} rows were requested.")
    rng = random.Random(seed)
    records = [rng.choice(by_q[q]) for q in rng.sample(sorted(by_q), n_rows)]
    with open(out, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"-> {out} ({n_rows} records, seed {seed})")


def encode_prompt(tok, prompt: str, prompt_format: str, enable_thinking: bool) -> list[int]:
    """Token ids for a record's prompt, built the way the RL agent loop builds the rollout prompt.

    `apply` mirrors `RPLAgentLoop._encode_prompt` under `chat_template_mode=apply`: the chat template with the
    generation prompt, so the ids end in the assistant header (and the empty think block when thinking is
    off). `raw` encodes the bare user text, which is what the reported PM rows were trained on.
    """
    if prompt_format == "apply":
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      tokenize=True, return_dict=True, enable_thinking=enable_thinking)
        return list(enc["input_ids"])
    if prompt_format == "raw":
        return tok(prompt, add_special_tokens=True)["input_ids"]
    raise ValueError(f"prompt_format must be 'apply' or 'raw', got {prompt_format!r}")


def tokenize(records: str, model: str, out: str, max_len: int = 16384, supervise: str = "program",
             prompt_format: str = "raw", enable_thinking: bool = False) -> None:
    """Emit `prompt_ids` / `response_ids` / `response_mask`, matching what the agent loop produces.

    The record is laid out as five spans -- earlier turns, the repaired program, the evidence it retrieved,
    the policy's reasoning, its answer -- and `supervise` decides which carry gradient (see `span_mask`).
    Injected text is never supervised in any mode, which is what `response_mask` exists to guarantee in RL.
    `prompt_format` decides how the prompt is encoded; see `encode_prompt`.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"] if s else []  # noqa: E731
    rows, counts = [], {"over_max_len": 0, "no_code": 0}
    for line in open(records):
        if not line.strip():
            continue
        rec = json.loads(line)
        # Records predating --supervise have neither field; they were built verify-only, hence correct.
        correct = bool(rec.get("correct", True))
        # The repaired program is what we are teaching, and it lives inside the prefix -- verify spliced it
        # in as the last <code> block. Masking the whole prefix to 0 supervises only the answer span, which
        # teaches "answer from this evidence" and not "write this program". So split the prefix at its last
        # code block and supervise that block too.
        head, code, obs = split_prefix(rec["prefix"])
        reasoning, answer = split_completion(rec["completion"])
        # The head is the earlier turns: the policy's own reasoning and <code> blocks interleaved with the
        # observations the loop injected between them. Tokenizing it as ONE span would force a single mask
        # value over both, so `rl` could not supervise the policy's own turns without the observations.
        head_spans = [(enc(text), kind) for text, kind in classify_spans(head)]
        tail = [enc(code), enc(obs), enc(reasoning), enc(answer)]
        pids = encode_prompt(tok, rec["prompt"], prompt_format, enable_thinking)
        if len(pids) + sum(len(ids) for ids, _ in head_spans) + sum(len(s) for s in tail) > max_len:
            counts["over_max_len"] += 1
            continue
        if not tail[0]:
            counts["no_code"] += 1
            continue
        rows.append({
            "prompt_ids": pids,
            "response_ids": [t for ids, _ in head_spans for t in ids] + [t for s in tail for t in s],
            "response_mask": span_mask([(len(ids), kind) for ids, kind in head_spans],
                                       *(len(s) for s in tail), mode=supervise, answer_trusted=correct),
            "question": rec["question"],
        })
    if not rows:
        sys.exit(f"ABORT: no records survived tokenization ({counts}); nothing to write.")
    # Invariants worth failing on rather than discovering during training.
    for r in rows:
        assert len(r["response_ids"]) == len(r["response_mask"]), "mask/ids length mismatch"
        assert any(r["response_mask"]), "a record with an all-zero mask contributes no gradient"
    pd.DataFrame(rows).to_parquet(out, index=False)
    tl = sorted(len(r["response_ids"]) for r in rows)
    sup = sorted(sum(r["response_mask"]) for r in rows)
    for k, v in counts.items():
        if v:
            print(f"  {v} records skipped: {k}")
    print(f"wrote {len(rows)} SFT rows (supervise={supervise}, prompt_format={prompt_format}) -> {out}")
    print(f"  response tokens: median {tl[len(tl) // 2]}, max {tl[-1]}")
    print(f"  supervised tokens (mask=1): median {sup[len(sup) // 2]}, min {sup[0]}, total {sum(sup)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify_parquet", default="", help="build mode: the verify set mutation_verify.py wrote")
    ap.add_argument("--verify_rollouts", default="",
                    help="build mode: the eval of --verify_parquet (eval.sh writes it under <out>/val_rollouts/; "
                         "it holds train-derived verify rows, not validation data)")
    ap.add_argument("--records", default="", help="tokenize mode: the jsonl produced by the build step")
    ap.add_argument("--tokenize", action="store_true")
    ap.add_argument("--model", default="", help="tokenize mode: HF model or path whose tokenizer to use")
    ap.add_argument("--max_len", type=int, default=16384)
    ap.add_argument("--keep_all", action=argparse.BooleanOptionalAction, default=True,
                    help="build mode: keep rows whose answer failed verify; they contribute their program only. "
                         "On by default, as in every reported round; --no-keep_all keeps verify passes only")
    ap.add_argument("--no_stitch_nudges", action="store_true",
                    help="build mode: keep the loop's invalid-action nudges instead of stitching them out")
    ap.add_argument("--correct_metric", default="cover_em", choices=["cover_em", "em", "acc"],
                    help="build mode: which metric counts as a verify pass. cover_em is the default and ~2x "
                         "the rows; the RL objective stays em (see the build_records docstring)")
    ap.add_argument("--supervise", default="program", choices=SUPERVISION_MODES,
                    help="tokenize mode: which spans carry gradient (see module docstring)")
    ap.add_argument("--rft_dump", default="",
                    help="RFT build mode: a merged train rollout window (merge_rollouts.py)")
    ap.add_argument("--train_parquet", default="", help="RFT build mode: the parquet the window trained on")
    ap.add_argument("--rft_rows", type=int, default=0, help="RFT build mode: rows to emit (the PM set's size)")
    ap.add_argument("--rft_correct_field", default="em", choices=["em", "score"],
                    help="RFT build mode: the 0/1 field marking a correct rollout; `score` only for dumps "
                         "without `em` from runs trained on the plain EM reward")
    ap.add_argument("--seed", type=int, default=0, help="RFT build mode: seed for the per-question draw")
    ap.add_argument("--prompt_format", choices=["apply", "raw"], default="",
                    help="tokenize mode, required: `apply` matches a chat-templated RL rollout; `raw` is what "
                         "the reported PM rows used (see encode_prompt)")
    ap.add_argument("--enable_thinking", type=int, choices=[0, 1], default=0,
                    help="tokenize mode, with --prompt_format apply: the run's --enable-thinking")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.tokenize:
        if not (a.records and a.model and a.prompt_format):
            sys.exit("--tokenize requires --records, --model and --prompt_format")
        tokenize(a.records, a.model, a.out, a.max_len, a.supervise, a.prompt_format, bool(a.enable_thinking))
    elif a.rft_dump:
        if not (a.train_parquet and a.rft_rows > 0):
            sys.exit("RFT build mode requires --train_parquet and --rft_rows")
        build_rft_records(a.rft_dump, a.train_parquet, a.out, a.rft_rows, a.rft_correct_field, a.seed,
                          stitch=not a.no_stitch_nudges)
    else:
        if not (a.verify_parquet and a.verify_rollouts):
            sys.exit("build mode requires --verify_parquet and --verify_rollouts")
        build_records(a.verify_parquet, a.verify_rollouts, a.out, a.keep_all, a.correct_metric,
                      stitch=not a.no_stitch_nudges)


if __name__ == "__main__":
    main()
