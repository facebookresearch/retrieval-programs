#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# RPL on GlobalQA: corpus-level aggregation (Count / MinMax / Sort / TopK over the documents matching a
# criterion). The answer is a number or a set of document ids, so the observation must carry corpus ids --
# that is what `show_doc_id` does, and every GlobalQA arm gets it.
#
#   scripts/train_globalqa.sh --data data/globalqa_rpl --rerank-gpus 4 --out runs/globalqa [--seed 2]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/train.sh"

rpl_defaults
rpl_globalqa_defaults
# The program prints the ids it aggregated rather than the documents, so the open-domain cap is enough.
MAX_OBS_LENGTH=2048

rpl_parse_args "$@"
rpl_resolve_topology
rpl_require_visible_gpus
rpl_require_data
rpl_require_retriever dense sparse
rpl_require_reranker
rpl_banner "RPL on GlobalQA"

RPL_MODE=code
RPL_EXTRA=(
  +rpl.show_doc_id=1
)
rpl_train
