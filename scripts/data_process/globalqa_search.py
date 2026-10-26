# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build GlobalQA train/test parquets, with one prompt template baked into every row.

GlobalQA (Luo et al., arXiv 2510.26205) asks corpus-level aggregation questions -- Count / MinMax /
Sort / TopK over the resumes that match a criterion -- so each answer is a number or a set of
document ids. Fetch the release the reported runs used, then build one directory per template.
"""

import argparse
import json
import os
import re

import pandas as pd

from prompts import create_prompt

# File names as published in the dataset repo.
SPLIT_FILES = {"train": "train (1).json", "test": "test (3).json"}


def infer_task_type(question: str) -> str:
    """Guess the aggregation a question asks for from its wording."""
    q = question.lower()
    if "how many" in q or "number of" in q:
        return "count"
    if re.search(r"top[\s-]?\d", q) or ("top" in q and ("smallest" in q or "largest" in q)):
        return "topk"
    if "sort" in q or "rank" in q or "order" in q:
        return "sort"
    if any(w in q for w in ("smallest", "largest", "minimum", "maximum", "min ", "max ")):
        return "minmax"
    return "other"


def build_rows(items: list[dict], template_type: str, split: str, exclude: frozenset[str] = frozenset()) -> list[dict]:
    """One training row per question not in `exclude`; the reward grades `target` as an integer set.

    `extra_info.index` is the question's position in the source file, so excluding rows does not renumber the rest.
    """
    return [{
        "data_source": "globalqa",
        "prompt": [{"role": "user", "content": create_prompt(q["question"], template_type)}],
        "ability": "aggregation",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": [str(q["answer"])],
                "golden_doc_ids": [int(x) for x in q["golden_doc_ids"]],
                "task_type": infer_task_type(q["question"]),
            },
        },
        "extra_info": {"split": split, "index": idx},
    } for idx, q in enumerate(items) if q["question"] not in exclude]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="directory holding the downloaded GlobalQA files")
    parser.add_argument("--template_type", required=True, help="see `prompts.create_prompt`")
    parser.add_argument("--local_dir", required=True, help="output directory for <split>.parquet")
    parser.add_argument("--splits", default="train,test", help="comma-separated: train and/or test")
    parser.add_argument("--exclude_test_questions", action="store_true",
                        help="drop every train row whose question is also a test question")
    args = parser.parse_args()

    splits = args.splits.split(",")
    unknown = sorted(set(splits) - set(SPLIT_FILES))
    if unknown:
        raise SystemExit(f"--splits takes train and/or test, got {unknown}")

    def load(split: str) -> list[dict]:
        with open(os.path.join(args.src, SPLIT_FILES[split])) as f:
            return json.load(f)

    test_questions = frozenset(q["question"] for q in load("test")) if args.exclude_test_questions else frozenset()
    os.makedirs(args.local_dir, exist_ok=True)
    for split in splits:
        items = load(split)
        rows = build_rows(items, args.template_type, split, test_questions if split == "train" else frozenset())
        out = os.path.join(args.local_dir, f"{split}.parquet")
        pd.DataFrame(rows).to_parquet(out, index=False)
        dropped = f", dropped {len(items) - len(rows)} that are test questions" if len(rows) < len(items) else ""
        print(f"wrote {len(rows)} rows (template={args.template_type}{dropped}) -> {out}")


if __name__ == "__main__":
    main()
