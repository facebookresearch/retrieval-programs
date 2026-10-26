#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Serve a cross-encoder reranker on :6980 for the `rerank()` primitive.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

MODEL="Qwen/Qwen3-Reranker-4B"
TYPE=qwen3
GPU_DEVICES=""
CACHE_SIZE=2000000
SCORE_BATCH=256
MAX_BATCH_PAIRS=1024
LOG="./reranker.log"
FOREGROUND=0
WAIT_MIN=30

function usage {
  cat <<EOF
Usage: $(basename "$0") [options]

  --model M            reranker checkpoint (default: ${MODEL})
  --type T             qwen3 | sentence_transformer (default: ${TYPE})
  --gpu-devices LIST   CUDA_VISIBLE_DEVICES, e.g. 4,5,6,7 for the second half of a node
  --cache-size N       (query, doc) score LRU entries (default: ${CACHE_SIZE})
  --score-batch N      cross-encoder forward batch (default: ${SCORE_BATCH})
  --max-batch-pairs N  cap on pairs coalesced per batching window (default: ${MAX_BATCH_PAIRS})
  --log FILE           log file when backgrounded (default: ${LOG})
  --wait MIN            minutes to wait for readiness (default: ${WAIT_MIN})
  --foreground         run in the foreground
  -h, --help

The cache matters more than it looks: GRPO issues one identical rerank() per group member, and validation
questions recur at every eval, so it removes several-fold duplicate work.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model)            MODEL="$2"; shift 2 ;;
    --type)             TYPE="$2"; shift 2 ;;
    --gpu-devices)      GPU_DEVICES="$2"; shift 2 ;;
    --cache-size)       CACHE_SIZE="$2"; shift 2 ;;
    --score-batch)      SCORE_BATCH="$2"; shift 2 ;;
    --max-batch-pairs)  MAX_BATCH_PAIRS="$2"; shift 2 ;;
    --log)              LOG="$2"; shift 2 ;;
    --wait)             WAIT_MIN="$2"; shift 2 ;;
    --foreground)       FOREGROUND=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
  esac
done

if [ -n "$GPU_DEVICES" ]; then export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"; fi
export RERANK_SCORE_BATCH="$SCORE_BATCH" RERANK_MAX_BATCH_PAIRS="$MAX_BATCH_PAIRS"
export no_proxy="127.0.0.1,localhost,0.0.0.0" NO_PROXY="127.0.0.1,localhost,0.0.0.0"

NDEV="$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c . || true)"
echo "== reranker: $MODEL ($TYPE) on :6980, ${NDEV:-all} visible GPU(s) =="

ARGS=(--reranker_type "$TYPE" --rerank_model_name_or_path "$MODEL" --rerank_cache_size "$CACHE_SIZE")
if [ "$FOREGROUND" = "1" ]; then
  exec python "$REPO_ROOT/rpl/search/rerank_server.py" "${ARGS[@]}"
fi

nohup python "$REPO_ROOT/rpl/search/rerank_server.py" "${ARGS[@]}" > "$LOG" 2>&1 &
SRV_PID=$!
echo "  pid $SRV_PID -> $LOG"
echo "  waiting for the model to download and load (minutes on a cold cache)..."
PROBE='{"queries":["who wrote hamlet"],"documents":[[{"id":0,"contents":"Hamlet\nby William Shakespeare"}]],"return_scores":true}'
for i in $(seq 1 $((WAIT_MIN * 6))); do
  if curl -sf --noproxy '*' http://127.0.0.1:6980/rerank -X POST \
       -H 'Content-Type: application/json' -d "$PROBE" >/dev/null 2>&1; then
    echo "  RERANKER UP after ~$((i * 10))s"
    echo "  throughput and queue depth are reported at /stats -- check it before a rerank arm."
    exit 0
  fi
  sleep 10
  # A dead process will never answer; say so immediately instead of waiting out the budget.
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "ERROR: the reranker process exited after ~$((i * 10))s. Last lines of $LOG:" >&2
    tail -30 "$LOG" >&2; exit 1
  fi
done
# Still alive but not answering: loading, not broken. Do not call that a failure without saying so.
echo "ERROR: the reranker is still not answering after ${WAIT_MIN} min, but the process (pid $SRV_PID)" >&2
echo "       is ALIVE -- it is probably still loading. Raise --wait, or watch $LOG." >&2
tail -15 "$LOG" >&2
exit 1
