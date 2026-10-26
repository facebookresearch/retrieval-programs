# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for the async (local-staged) checkpoint save in `fsdp_checkpoint_manager`."""

import os
import re
import shutil
import time

# Shard files are per-rank (`model_world_size_8_rank_3.pt`); anything else in a checkpoint dir is shared
# (huggingface/, fsdp_config.json, ...) and belongs to rank 0 alone.
RANK_FILE_RE = re.compile(r"_rank_(\d+)\.pt$")


def entries_for_rank(names: list[str], rank: int) -> list[str]:
    """Split checkpoint entry names by owner: each rank takes its own shards, rank 0 takes the rest.

    Every name goes to exactly one rank, so the copies partition the checkpoint with no duplicated work
    and nothing dropped.
    """
    owned = []
    for name in sorted(names):
        match = RANK_FILE_RE.search(name)
        if match is not None:
            if int(match.group(1)) == rank:
                owned.append(name)
        elif rank == 0:
            owned.append(name)
    return owned


def copy_entries(stage_path: str, final_path: str, names: list[str]) -> None:
    """Copy `names` from the staging dir to the durable dir."""
    os.makedirs(final_path, exist_ok=True)
    for name in names:
        src, dst = os.path.join(stage_path, name), os.path.join(final_path, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, copy_function=shutil.copyfile, dirs_exist_ok=True)
        else:
            shutil.copyfile(src, dst)


def remove_entries(stage_path: str, names: list[str]) -> None:
    """Delete only the staged entries this rank owns.

    Every rank on a node shares one staging dir, so an `rmtree(stage_path)` deletes files the other ranks
    have not copied yet. Rank 0 owns the shared entries (`huggingface/`, `fsdp_config.json`) on top of its
    own shards, so it is the slowest and always lost that race: whichever rank finished first wiped
    `huggingface/`, and rank 0's copy then died on ENOENT partway through, leaving `model_*_rank_0.pt` and
    `optim_*_rank_0.pt` unwritten and the tracker (correctly) never advanced.
    """
    for name in names:
        path = os.path.join(stage_path, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def prune_empty_dirs(stage_path: str) -> None:
    """Best-effort removal of the staging dirs once every rank has cleared its own entries.

    `rmdir` only succeeds on an empty directory, which is exactly the wanted interlock: whichever rank
    finishes last removes them, and the others fail harmlessly while files remain.
    """
    for path in (stage_path, os.path.dirname(stage_path.rstrip("/"))):
        try:
            os.rmdir(path)
        except OSError:
            return  # still holding another rank's files, or already gone


def wait_for_paths(required: list[str], timeout_s: float, poll_s: float = 5.0) -> list[str]:
    """Block until every path in `required` exists; return the ones still missing at timeout ([] on success)."""
    deadline = time.time() + timeout_s
    missing = list(required)
    while True:
        missing = [p for p in required if not os.path.exists(p)]
        if not missing or time.time() >= deadline:
            return missing
        time.sleep(min(poll_s, max(0.0, deadline - time.time())))
