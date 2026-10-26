# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Merge a window of train rollout dumps into the single jsonl harvest harvests from.

Each dump contributes its first `--rows_per_dump` rows (512): the reported rounds trimmed every train dump to
its first 512 rows (about 100 questions per step), while training here dumps the whole batch (2,560 rows).

Usage:
    python merge_rollouts.py --dir <run>/rollouts --lo 151 --hi 200 --out <round>/rollouts/window.jsonl
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="the run's rollouts/ directory")
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--rows_per_dump", type=int, default=512, help="first N rows of each dump; 0 keeps all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if os.path.basename(a.dir.rstrip("/")) != "rollouts":
        sys.exit(f"ABORT: --dir must be a train `rollouts/` directory, got {a.dir}")

    steps = [s for s in range(a.lo, a.hi + 1) if (s - a.lo) % a.every == 0]
    found, rows = [], 0
    with open(a.out, "w") as out:
        for s in steps:
            p = os.path.join(a.dir, f"{s}.jsonl")
            if not os.path.exists(p):
                continue
            found.append(s)
            kept = 0
            with open(p) as fh:
                for line in fh:
                    if a.rows_per_dump and kept >= a.rows_per_dump:
                        break
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        kept += 1
            rows += kept
    # A silently short merge produces a harvest that looks healthy but covers a fraction of the window.
    if len(found) < len(steps) // 2:
        sys.exit(f"ABORT: only {len(found)}/{len(steps)} dumps present in {a.dir}; the window is mostly missing")
    print(f"merged {len(found)}/{len(steps)} dumps (steps {found[0]}..{found[-1]}), {rows} rollouts -> {a.out}")


if __name__ == "__main__":
    main()
