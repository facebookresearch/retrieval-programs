# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build train/test parquets from FlashRAG QA datasets, with one prompt template baked into every row.

    python qa_search.py --data_sources nq,hotpotqa --template_type rpl-openqa --local_dir data/nq_hotpotqa_rpl
    python qa_search.py --data_sources popqa --splits test --template_type rpl-openqa --local_dir data/popqa_rpl
"""

import argparse
import os

import datasets

from prompts import create_prompt


def select_split(dataset: datasets.DatasetDict, data_source: str, split: str) -> datasets.Dataset:
    """The rows of `split` for one source; `test` falls back to dev where there is no test split."""
    if split == "train":
        if "train" not in dataset:
            raise SystemExit(f"{data_source} has no train split (it is eval-only); build it with --splits test")
        return dataset["train"]
    for name in ("test", "dev"):
        if name in dataset:
            print(f"Using the {data_source} {name} split for test")
            return dataset[name]
    raise SystemExit(f"{data_source} has neither a test nor a dev split")


def make_map_fn(data_source: str, split: str, template_type: str):
    def process_fn(example, idx):
        example["question"] = example["question"].strip()
        if example["question"][-1] != "?":
            example["question"] += "?"
        return {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": create_prompt(example["question"], template_type=template_type),
            }],
            "ability": "fact-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": {"target": example["golden_answers"]},
            },
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }

    return process_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local_dir", required=True, help="output directory for <split>.parquet")
    parser.add_argument("--template_type", default="base", help="see `prompts.create_prompt`")
    parser.add_argument("--data_sources", default="nq", help="comma-separated FlashRAG_datasets configs")
    parser.add_argument("--splits", default="train,test", help="comma-separated: train and/or test")
    args = parser.parse_args()

    splits = args.splits.split(",")
    unknown = sorted(set(splits) - {"train", "test"})
    if unknown:
        raise SystemExit(f"--splits takes train and/or test, got {unknown}")

    parts = {split: [] for split in splits}
    for data_source in args.data_sources.split(","):
        dataset = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", data_source)
        for split in splits:
            rows = select_split(dataset, data_source, split)
            parts[split].append(rows.map(function=make_map_fn(data_source, split, args.template_type),
                                         with_indices=True))

    os.makedirs(args.local_dir, exist_ok=True)
    for split, sets in parts.items():
        datasets.concatenate_datasets(sets).to_parquet(os.path.join(args.local_dir, f"{split}.parquet"))


if __name__ == "__main__":
    main()
