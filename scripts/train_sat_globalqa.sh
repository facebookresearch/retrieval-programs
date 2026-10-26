#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Search-R1 on GlobalQA.
# The GlobalQA paper fixes retrieval depth at 20. The observation cap is sized so all twenty
# documents fit.
#
#   scripts/train_sat_globalqa.sh --data data/globalqa_base --out runs/sat_globalqa [--seed 2]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/train.sh"

rpl_defaults
rpl_globalqa_defaults
TOPK=20
MAX_OBS_LENGTH=3584

rpl_parse_args "$@"
rpl_resolve_topology
rpl_require_visible_gpus
rpl_require_data
rpl_require_retriever
rpl_banner "SaT (Search-as-a-Tool) on GlobalQA"

RPL_MODE=search
RPL_EXTRA=(
  +rpl.show_doc_id=1
)
rpl_train
