#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The policy writes Python that composes the retrieval SDK (dense_retrieval,  sparse_retrieval, rerank,
# fuse, dedup) inside a sandbox, instead of emitting a fixed search query.
#
# Needs BOTH retrievers (the SDK routes dense_retrieval/sparse_retrieval to the two backends of one server)
# and a reranker. Build the data with `scripts/prepare_data.sh --template rpl-openqa`.
#
#   scripts/serve_retriever.sh --index IDX --corpus CORPUS --bm25-index BM25 --multi
#   scripts/serve_reranker.sh --gpu-devices 4,5,6,7
#   scripts/train_rpl.sh --data data/nq_hotpotqa_rpl --rerank-gpus 4 --out runs/rpl

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/train.sh"

rpl_defaults
rpl_parse_args "$@"
rpl_resolve_topology
rpl_require_visible_gpus
rpl_require_data
rpl_require_retriever dense sparse
rpl_require_reranker
rpl_banner "RPL (Retrieval Program Learning)"

RPL_MODE=code
RPL_EXTRA=()
rpl_train
