#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Evaluate a checkpoint or an untrained model on the held-out sets.
#
#   scripts/eval.sh --data data/nq_hotpotqa_rpl --mode code --rerank-gpus 4 \
#       --ckpt runs/rpl/global_step_1000/actor/huggingface --out runs/rpl_eval
#   scripts/eval.sh --data data/nq_rag --mode search --single-pass --model Qwen/Qwen3-4B   # training-free baseline
#
# Each eval writes to its own --out (default: a fresh ./eval/<time>) and never resumes from a checkpoint there,
# so it always evaluates --ckpt, not the newest training checkpoint in a shared directory.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/train.sh"

rpl_defaults
EVAL_ONLY=1
OUT_DIR="./eval/$(date +%m%d-%H%M%S)"
MODE_ARG=""            # search | code -- must match the prompt the parquet was built with
SINGLE_PASS=0
GLOBALQA=0
CKPT=""
NEEDS_RERANKER=0
NO_RERANK=0

# Parse the eval-only flags, then hand the rest to the shared parser.
REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --ckpt)         CKPT="$2"; shift 2 ;;
    --mode)         MODE_ARG="$2"; shift 2 ;;
    --single-pass)  SINGLE_PASS=1; shift ;;
    --globalqa)     GLOBALQA=1; shift ;;
    --with-rerank)  NEEDS_RERANKER=1; shift ;;
    --no-rerank)    NO_RERANK=1; shift ;;
    -h|--help)      rpl_usage; exit 0 ;;
    --)             REST+=("$@"); break ;;
    *)              REST+=("$1"); shift ;;
  esac
done
case "$MODE_ARG" in
  search|code) ;;
  "") echo "ERROR: --mode search|code is required, matching the prompt the parquet was built with. An RPL" >&2
      echo "       parquet evaluated under search runs with no sandbox, so every <code> block scores as invalid." >&2
      exit 2 ;;
  *)  echo "ERROR: --mode must be search or code, not '$MODE_ARG'" >&2; exit 2 ;;
esac

# Before the shared parser, so the command line still wins.
GLOBALQA_EXTRA=()
if [ "$GLOBALQA" = "1" ]; then
  rpl_globalqa_defaults
  GLOBALQA_EXTRA=(+rpl.show_doc_id=1)
  # Search-R1 and IRCoT retrieve the 20 documents per call both GlobalQA papers fix, and all twenty fit.
  if [ "$MODE_ARG" = "search" ]; then TOPK=20; MAX_OBS_LENGTH=3584; fi
  # The RAG prompt carries its top-20 context, about 3.8k tokens.
  if [ "$SINGLE_PASS" = "1" ]; then MAX_PROMPT_LEN=8192; fi
fi
rpl_parse_args "${REST[@]+"${REST[@]}"}"

if [ -n "$CKPT" ]; then MODEL="$CKPT"; fi
if [ "$SINGLE_PASS" = "1" ]; then
  MAX_TURNS=1
  if [ "$CHAT_TEMPLATE" = "raw" ]; then
    echo "  NOTE: --single-pass under raw: a RAG prompt tends to continue its documents (the reported" \
         "training-free baselines used apply; the base-model mutation rounds used raw)"
  fi
fi
if [ "$MODE_ARG" = "code" ] && [ "$NO_RERANK" = "0" ]; then NEEDS_RERANKER=1; fi

rpl_resolve_topology
rpl_require_visible_gpus
rpl_require_data
if [ "$MODE_ARG" = "code" ]; then rpl_require_retriever dense sparse; else rpl_require_retriever; fi
if [ "$NEEDS_RERANKER" = "1" ]; then rpl_require_reranker; fi
rpl_banner "eval$([ "$SINGLE_PASS" = 1 ] && echo ' (single-pass)')"
echo "  checkpoint   ${CKPT:-<untrained $MODEL>}"

RPL_MODE="$MODE_ARG"
RPL_EXTRA=(+rpl.single_pass="$SINGLE_PASS" "${GLOBALQA_EXTRA[@]+"${GLOBALQA_EXTRA[@]}"}")
rpl_train
