# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""GlobalQA metrics, overall and split by task type, in MARAG-R1's reporting shape.

The main table reports F1 and D-F1@20 per task type (TopK / Count / Sort / MinMax) plus an average, so a
comparable table needs the same dissection. GlobalQA ships no type labels, so every row is labeled from its
question text by the rules below, which match the type distribution the GlobalQA paper publishes.

Ordering matters in the rules -- "the 2 smallest ids" is TopK while "the smallest id" is MinMax, so TopK must be
tested first or every TopK row is misfiled as MinMax.

Usage:
    python tools/globalqa_report.py <run>/val_rollouts/<step>.jsonl [more.jsonl ...]
"""

from __future__ import annotations

import argparse
import json
import re

# First match wins; see the docstring on why TopK precedes MinMax.
RULES = [
    ("TopK", re.compile(r"\btop[-\s]?\d+\b|\b(?:two|three|four|five|\d+)\s+"
                        r"(?:smallest|largest|biggest|lowest|highest)\b", re.I)),
    ("Sort", re.compile(r"\bsort(?:ed)?\b|\bascending\b|\bdescending\b|\bin order\b|\bordered\b", re.I)),
    ("Count", re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal number\b", re.I)),
    ("MinMax", re.compile(r"\b(?:smallest|largest|biggest|minimum|maximum|highest|lowest)\b", re.I)),
]
TYPES = ["TopK", "Count", "Sort", "MinMax"]
COLS = [("set-EM", "acc"), ("set-F1", "f1"), ("token-F1", "token_f1"),
        ("doc-F1", "doc_f1"), ("dF1@20", "doc_f1@20"), ("#docs", "n_retrieved_docs")]


def task_type(rec: dict) -> str:
    """The row's task type, one of `TYPES`, from its question text; "OTHER" if no rule matches."""
    q = question_of(rec)
    for name, pat in RULES:
        if pat.search(q):
            return name
    return "OTHER"


def question_of(rec: dict) -> str:
    """The question only. The RAG prompt is `... Question: {q} Context: {docs}`, so without cutting at
    " Context: " this returned ~13k chars of retrieved resumes and the patterns matched words inside the
    documents rather than the question."""
    q = (rec.get("input") or "").rsplit(" Question: ", 1)[-1]
    return q.split(" Context: ", 1)[0].strip()


def report(path: str) -> None:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    by_type: dict[str, list] = {t: [] for t in TYPES + ["OTHER"]}
    for r in rows:
        by_type[task_type(r)].append(r)
    if by_type["OTHER"]:
        print(f"  WARNING: {len(by_type['OTHER'])} rows matched no task-type rule and are excluded per-type")

    print(f"\n== {path}   n={len(rows)}")
    print("  " + f"{'type':<9}{'n':>6}" + "".join(f"{lbl:>10}" for lbl, _ in COLS))
    for t in TYPES + (["OTHER"] if by_type["OTHER"] else []):
        sub = by_type[t]
        if not sub:
            continue
        cells = "".join(f"{sum(r.get(k, 0.0) for r in sub) / len(sub):>10.3f}" for _, k in COLS)
        print(f"  {t:<9}{len(sub):>6}{cells}")
    cells = "".join(f"{sum(r.get(k, 0.0) for r in rows) / len(rows):>10.3f}" for _, k in COLS)
    print(f"  {'ALL':<9}{len(rows):>6}{cells}")
    # Their table averages the four task types rather than pooling rows; report both, they differ when the
    # type distribution is uneven (it is: MinMax/TopK ~32% each vs Count/Sort ~18%).
    present = [t for t in TYPES if by_type[t]]
    cells = "".join(f"{sum(sum(r.get(k, 0.0) for r in by_type[t]) / len(by_type[t]) for t in present) / len(present):>10.3f}"
                    for _, k in COLS)
    print(f"  {'AVG(type)':<9}{'':>6}{cells}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help="GlobalQA validation dumps")
    a = ap.parse_args()
    for path in a.dumps:
        report(path)


if __name__ == "__main__":
    main()
