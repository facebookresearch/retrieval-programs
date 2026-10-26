#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The Search-R1 baseline. The policy emits a query between <search> tags and the retriever returns
# passages between <information> tags. Fixed retrieval, no argument choice.
#
#   scripts/serve_retriever.sh --index IDX --corpus CORPUS --bm25-index BM25 --multi
#   scripts/train_sat.sh --data data/nq_hotpotqa_base --out runs/sat

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/train.sh"

rpl_defaults
rpl_parse_args "$@"
rpl_resolve_topology
rpl_require_visible_gpus
rpl_require_data
rpl_require_retriever
rpl_banner "SaT (Search-as-a-Tool) baseline"

RPL_MODE=search
RPL_EXTRA=()
rpl_train
