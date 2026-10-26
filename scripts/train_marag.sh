#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# MARAG-R1 baseline (arXiv 2510.27569) reimplemented teacher-free.
#
# DocumentFilter is backed by the reranker, and KeywordSearch by the sparse backend, so both services are
# mandatory.
#
#   scripts/train_marag.sh --data data/globalqa_marag --rerank-gpus 4 --out runs/marag [--seed 2]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/train.sh"

rpl_defaults
rpl_globalqa_defaults
# One tool per turn, so it needs more turns than RPL's 3; GlobalQA caps retrieval iterations at 10.
MAX_TURNS=10
MAX_GEN_TOKENS=1024
MAX_OBS_LENGTH=900
PPO_MAX_TOKEN_LEN=17408
MARAG_TOPK=20        # fixed retrieval breadth; the MARAG tools take no breadth argument
MARAG_FILTER_FRAC=0  # >0 keeps a fraction of DocumentFilter's input instead of its relevance judgement

rpl_parse_args "$@"
rpl_resolve_topology
rpl_require_visible_gpus
rpl_require_data
rpl_require_retriever dense sparse
rpl_require_reranker
rpl_banner "MARAG-R1 baseline"

# `marag`, not `code`: the one-tool-per-turn rule is enforced only in this mode, so under `code` the baseline
# could compose tools within a turn and would no longer be MARAG-R1.
RPL_MODE=marag
RPL_EXTRA=(
  +rpl.marag_topk="$MARAG_TOPK"
  +rpl.marag_filter_frac="$MARAG_FILTER_FRAC"
  +rpl.show_doc_id=1
)
rpl_train
