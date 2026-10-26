# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SFT dataset for program mutation: pre-tokenized RL token streams rather than chat messages.

`scripts/program_mutation/mutation_sft.py --tokenize` writes `prompt_ids` / `response_ids` / `response_mask`,
matching `AgentLoopOutput` exactly (mask 1 = the policy generated it, 0 = injected). This class concatenates
them into what the SFT trainer wants and maps that mask onto `loss_mask`.

Wired from `scripts/sft_mutation.sh` via `data.custom_cls`:
    data.custom_cls.path=<repo>/rpl/llm_agent/mutation_sft_dataset.py
    data.custom_cls.name=MutationSFTDataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch.utils.data import Dataset

from verl.utils.dataset.dataset_utils import DatasetPadMode

from rpl.llm_agent.sft_mask import assemble

PROMPT_KEY, RESPONSE_KEY, MASK_KEY = "prompt_ids", "response_ids", "response_mask"


def _ids(value) -> list[int]:
    """Coerce one parquet cell to a list of ints."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


class MutationSFTDataset(Dataset):
    """Flat pre-tokenized SFT records for the program-mutation verify trajectories.

    Example:
        ds = MutationSFTDataset("sft.parquet", tokenizer, config={"max_length": 16384})
        item = ds[0]                      # input_ids / position_ids / loss_mask (+ attention_mask if padded)
        tokenizer.decode([t for t, m in zip(item["input_ids"], item["loss_mask"]) if m])  # the supervised span
    """

    def __init__(self, parquet_files, tokenizer, config=None, processor=None, max_samples: int = -1):
        config = config or {}
        self.pad_mode = config.get("pad_mode", DatasetPadMode.NO_PADDING)
        if self.pad_mode not in (DatasetPadMode.RIGHT, DatasetPadMode.NO_PADDING):
            raise ValueError(f"pad_mode must be 'right' or 'no_padding', got {self.pad_mode!r}")
        self.truncation = config.get("truncation", "error")
        if self.truncation not in ("error", "left", "right"):
            raise ValueError(f"truncation must be error/left/right, got {self.truncation!r}")
        self.max_length = int(config.get("max_length", 16384))
        self.tokenizer = tokenizer
        self.processor = processor

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(f) for f in parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        missing = [k for k in (PROMPT_KEY, RESPONSE_KEY, MASK_KEY) if k not in self.dataframe.columns]
        if missing:
            raise ValueError(
                f"{parquet_files} is missing {missing}; this dataset reads the pre-tokenized records written "
                f"by `mutation_sft.py --tokenize`, not a messages parquet. Columns present: "
                f"{list(self.dataframe.columns)}"
            )
        if config.get("shuffle", False):
            self.dataframe = self.dataframe.sample(frac=1.0, random_state=config.get("seed")).reset_index(drop=True)
        if max_samples is not None and max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples]
        if len(self.dataframe) == 0:
            raise ValueError(f"{parquet_files} has no rows; nothing to fine-tune on")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, item: int) -> dict:
        row = self.dataframe.iloc[item]
        try:
            ids, mask = assemble(_ids(row[PROMPT_KEY]), _ids(row[RESPONSE_KEY]), _ids(row[MASK_KEY]))
        except ValueError as e:
            raise ValueError(f"row {item}: {e}") from e
        input_ids = torch.tensor(ids, dtype=torch.long)
        loss_mask = torch.tensor(mask, dtype=torch.long)
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)
        seq_len = input_ids.shape[0]

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            if seq_len > self.max_length:
                if self.truncation == "error":
                    raise ValueError(f"row {item}: {seq_len=} exceeds {self.max_length=}")
                input_ids, loss_mask, position_ids = self._truncate(input_ids, loss_mask, position_ids)
            return {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}

        attention_mask = torch.ones(seq_len, dtype=torch.long)
        if seq_len < self.max_length:
            pad = self.max_length - seq_len
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = torch.cat((input_ids, torch.full((pad,), pad_id, dtype=input_ids.dtype)))
            attention_mask = torch.cat((attention_mask, torch.zeros(pad, dtype=attention_mask.dtype)))
            loss_mask = torch.cat((loss_mask, torch.zeros(pad, dtype=loss_mask.dtype)))
            position_ids = F.pad(position_ids, (0, pad), value=0)
        elif seq_len > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"row {item}: {seq_len=} exceeds {self.max_length=}")
            input_ids, loss_mask, position_ids, attention_mask = self._truncate(
                input_ids, loss_mask, position_ids, attention_mask
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def _truncate(self, *tensors):
        """Slice every tensor the same way, then refuse a slice that dropped all supervision.

        `left` keeps the answer and loses the prompt; `right` keeps the prompt and loses the answer. Either
        can leave a row with an all-zero mask, which is a row that trains on nothing -- worth failing on
        rather than quietly averaging in.
        """
        sl = (
            slice(-self.max_length, None)
            if self.truncation == "left"
            else slice(0, self.max_length)
        )
        out = tuple(t[..., sl] for t in tensors)
        if not out[1].any():
            raise ValueError(
                f"truncation='{self.truncation}' removed every supervised token from a row "
                f"(max_length={self.max_length}); raise max_length or drop the row upstream"
            )
        return out
