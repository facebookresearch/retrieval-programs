#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# AutoRefine baseline (arXiv 2505.11277): the Search-R1 loop of train_sat.sh, plus a <refine> step in which
# the policy distills each <information> block before answering. Only the prompt and the reward differ from
# Search-R1. The reward is AutoRefine's cascade: answer F1 when it is nonzero, else 0.1 when the gold answer
# survives into the <refine> blocks. It is the one arm that trains on F1 rather than EM.
#
#   scripts/prepare_data.sh --template autorefine --out data/nq_hotpotqa_autorefine
#   scripts/serve_retriever.sh --index IDX --corpus CORPUS
#   scripts/train_autorefine.sh --data data/nq_hotpotqa_autorefine --out runs/autorefine

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/train.sh"

rpl_defaults
REWARD_MODE=autorefine

rpl_parse_args "$@"
rpl_resolve_topology
rpl_require_visible_gpus
rpl_require_data
rpl_require_retriever
# Without the autorefine prompt the policy is never asked for <refine>: the refine payout is unearnable and the
# run is Search-R1 trained on F1, which reads as "refining does not help".
if [ "$DRY_RUN" = "0" ]; then
  python - "$TRAIN_FILES" <<'PY'
import sys

import pyarrow.parquet as pq

prompt = pq.read_table(sys.argv[1], columns=["prompt"]).slice(0, 1).to_pylist()[0]["prompt"]
if "<refine>" not in prompt[-1]["content"]:
    sys.exit(f"ERROR: {sys.argv[1]} was not built with --template autorefine; its prompt never asks for <refine>.")
PY
fi
rpl_banner "AutoRefine baseline"

RPL_MODE=search
RPL_EXTRA=()
rpl_train
