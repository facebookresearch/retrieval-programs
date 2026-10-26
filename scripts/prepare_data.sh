#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Build the training and evaluation parquets.
#
#   scripts/prepare_data.sh --template base       --out data/nq_hotpotqa_base    # SaT baseline
#   scripts/prepare_data.sh --template rpl-openqa --out data/nq_hotpotqa_rpl     # the method
#   scripts/prepare_data.sh --template rpl-openqa --out data/nq_rpl --datasets nq
#   scripts/prepare_data.sh --template rpl-openqa --out data/popqa_rpl --datasets popqa --splits test
#
# Templates: see `create_prompt` in scripts/data_process/prompts.py.
# The RAG baselines also need retrieved passages: build a `direct` set here, then run
# scripts/data_process/qa_rag.py on it. GlobalQA is not a FlashRAG dataset; see
# scripts/data_process/globalqa_search.py.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DP="$HERE/data_process"

TEMPLATE=base
OUT=""
DATASETS="nq,hotpotqa"
SPLITS="train,test"

function usage {
  cat <<USAGE
Usage: $(basename "$0") --template NAME --out DIR [--datasets a,b] [--splits train,test]

  --template NAME   prompt template baked into the parquet   (default: ${TEMPLATE})
  --out DIR         output directory                          [required]
  --datasets LIST   comma-separated source datasets           (default: ${DATASETS})
  --splits LIST     train and/or test; popqa and bamboogle are eval-only (default: ${SPLITS})
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --template) TEMPLATE="$2"; shift 2 ;;
    --out)      OUT="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --splits)   SPLITS="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *)          echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "ERROR: --out is required" >&2; usage; exit 2; }

mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"   # absolute: the build runs from data_process/, where a relative --out would land
cd "$DP"   # the data_process scripts import `prompts` as a sibling module

python qa_search.py --data_sources "$DATASETS" --splits "$SPLITS" --template_type "$TEMPLATE" --local_dir "$OUT"
echo "== wrote $OUT (template: $TEMPLATE) =="
