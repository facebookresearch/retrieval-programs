# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-dataset EM from validation dumps: checkpoint selection, and the seven-benchmark average.

    # The checkpoint to report: the best unweighted mean EM over the run's held-out split (nq + hotpotqa), among
    # steps 50..1000 that kept an HF export; ties go to the later step.
    python tools/qa_report.py select --run runs/rpl --parquet data/nq_hotpotqa_rpl/test.parquet

    # That checkpoint's seven-benchmark eval. The reported average is the unweighted mean of the seven.
    python tools/qa_report.py score --dump runs/rpl_eval7/val_rollouts/0.jsonl \
        --parquet data/{nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle}_rpl/test.parquet
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re

import pyarrow.parquet as pq

STEP_RE = re.compile(r"^(\d+)\.jsonl$")


def _key(prompt: str, targets: list) -> tuple:
    """A row's match key: the question after the prompt's last `Question:`, plus its gold answers.

    A RAG prompt continues `Question: <q> Context: <docs>`, so the question stops at ` Context: `.
    """
    question = prompt.rsplit("Question:", 1)[-1].split(" Context: ", 1)[0].split("\n", 1)[0].strip()
    return question, tuple(sorted(map(str, targets)))


def em_by_dataset(dump: str, parquets: list[str]) -> dict[str, float]:
    """Return {data_source: mean EM} for one validation dump of `parquets`.

    Raises:
        ValueError: if the dump is not exactly one validation of the parquets' rows.
    """
    sources: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for path in parquets:
        for r in pq.read_table(path, columns=["data_source", "prompt", "reward_model"]).to_pylist():
            key = _key(r["prompt"][-1]["content"], r["reward_model"]["ground_truth"]["target"])
            sources[key][r["data_source"]] += 1
    rows = [json.loads(line) for line in open(dump) if line.strip()]
    expected = sum(sum(c.values()) for c in sources.values())
    if len(rows) != expected:
        raise ValueError(f"{dump}: {len(rows)} rows, expected {expected}; not one validation of these parquets")
    hits = collections.defaultdict(list)
    for r in rows:
        key = _key(r["input"], r["gts"]["target"])
        left = +sources[key]
        if not left:
            raise ValueError(f"{dump}: no parquet row left for {r['input'][-160:]!r}")
        ds = next(iter(left))
        sources[key][ds] -= 1
        hits[ds].append(float(r["em"]))
    return {ds: sum(v) / len(v) for ds, v in sorted(hits.items())}


def select(run: str, parquet: str, min_step: int, max_step: int) -> None:
    """Print EM per held-out dataset at every validated step and the checkpoint the rule picks."""
    steps = sorted(int(m.group(1)) for n in os.listdir(os.path.join(run, "val_rollouts")) if (m := STEP_RE.match(n)))
    usable, datasets = [], None
    for step in [s for s in steps if min_step <= s <= max_step]:
        try:
            em = em_by_dataset(os.path.join(run, "val_rollouts", f"{step}.jsonl"), [parquet])
        except ValueError as e:
            print(f"  skip step {step}: {e}")
            continue
        if datasets is None:
            datasets = list(em)
            print(f"{'step':>6}  " + "  ".join(f"{d:>9}" for d in datasets) + f"  {'mean':>7}  hf export")
        mean = sum(em.values()) / len(em)
        ok = os.path.isdir(os.path.join(run, f"global_step_{step}", "actor", "huggingface"))
        cells = "  ".join(f"{em[d]:9.4f}" for d in datasets)
        print(f"{step:>6}  {cells}  {mean:7.4f}  {'yes' if ok else 'MISSING'}")
        if ok:
            usable.append((mean, step))
    if not usable:
        raise SystemExit(f"no selectable step in {min_step}..{max_step} with an HF export under {run}")
    mean, step = max(usable)
    print(f"-> step {step} (mean {mean:.4f}): {os.path.join(run, f'global_step_{step}', 'actor', 'huggingface')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("select", help="pick the checkpoint on the run's held-out split")
    sp.add_argument("--run", required=True, help="the run's --out dir (val_rollouts/, global_step_*/)")
    sp.add_argument("--parquet", required=True, help="the held-out parquet the run validated on")
    sp.add_argument("--min-step", type=int, default=50, help="step 0 is the untrained model")
    sp.add_argument("--max-step", type=int, default=1000, help="verl can write a stray step past the schedule")
    sc = sub.add_parser("score", help="per-dataset EM and the unweighted mean of one eval dump")
    sc.add_argument("--dump", required=True)
    sc.add_argument("--parquet", required=True, nargs="+", help="the parquets the eval ran (its --val)")
    a = ap.parse_args()
    if a.cmd == "select":
        select(a.run, a.parquet, a.min_step, a.max_step)
        return
    em = em_by_dataset(a.dump, a.parquet)
    for ds, v in em.items():
        print(f"  {ds:<18}{v:.4f}")
    print(f"  {'mean (unweighted)':<18}{sum(em.values()) / len(em):.4f}   over {len(em)} datasets")


if __name__ == "__main__":
    main()
