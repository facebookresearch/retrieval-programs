#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Serve the retriever on :8000: the wiki-18 corpus by default, or GlobalQA's with --retriever bge (see README).
#
# Two backends, both served by the same process:
#   dense  -- E5 (bge for GlobalQA) over a FAISS index (flat = exact, hnsw = approximate)
#   sparse -- BM25 over a Pyserini/Lucene index; needs a JDK on PATH (JAVA_HOME)
#
# `--multi` serves both at once, which is what the RPL arms need: `dense_retrieval()` and
# `sparse_retrieval()` route to the two backends of this one server.
#
# Build the indices first with rpl/search/build_index.sh. Loading a flat index takes minutes and tens of GB
# of RAM (or GPU memory with --faiss-gpu); the readiness probe at the end is what tells you it is up.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

INDEX=""
CORPUS=""
BM25_INDEX=""
RETRIEVER=e5
RETRIEVER_MODEL=intfloat/e5-base-v2
TOPK=3
MULTI=0
FAISS_GPU=0
BATCH_WINDOW=0.01
GPU_DEVICES=""
LOG="./retriever.log"
FOREGROUND=0
WAIT_MIN=40

function usage {
  cat <<EOF
Usage: $(basename "$0") --index DIR --corpus FILE [options]

  --index DIR          FAISS index directory (from rpl/search/build_index.sh)   [required unless --bm25-only]
  --corpus FILE        corpus JSONL with {"id", "contents"} per line            [required]
  --bm25-index DIR     Pyserini index directory; enables the sparse backend
  --multi              serve dense AND sparse together (what the RPL arms need)
  --retriever NAME     dense encoder family (default: ${RETRIEVER})
  --retriever-model M  dense encoder checkpoint (default: ${RETRIEVER_MODEL})
  --topk N             default top-k when a caller does not pass one (default: ${TOPK})
  --faiss-gpu          put the FAISS index on GPU (much faster, needs the memory)
  --batch-window S     batch requests arriving within S seconds; 0 disables (default: ${BATCH_WINDOW})
  --gpu-devices LIST   CUDA_VISIBLE_DEVICES for this server (default: inherit)
  --log FILE           log file when backgrounded (default: ${LOG})
  --wait MIN            minutes to wait for readiness (default: ${WAIT_MIN})
  --foreground         run in the foreground instead of backgrounding
  -h, --help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --index)           INDEX="$2"; shift 2 ;;
    --corpus)          CORPUS="$2"; shift 2 ;;
    --bm25-index)      BM25_INDEX="$2"; shift 2 ;;
    --multi)           MULTI=1; shift ;;
    --retriever)       RETRIEVER="$2"; shift 2 ;;
    --retriever-model) RETRIEVER_MODEL="$2"; shift 2 ;;
    --topk)            TOPK="$2"; shift 2 ;;
    --faiss-gpu)       FAISS_GPU=1; shift ;;
    --batch-window)    BATCH_WINDOW="$2"; shift 2 ;;
    --gpu-devices)     GPU_DEVICES="$2"; shift 2 ;;
    --log)             LOG="$2"; shift 2 ;;
    --wait)             WAIT_MIN="$2"; shift 2 ;;
    --foreground)      FOREGROUND=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)                 echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
  esac
done

[ -n "$CORPUS" ] || { echo "ERROR: --corpus is required" >&2; usage; exit 2; }
[ -f "$CORPUS" ] || { echo "ERROR: corpus not found: $CORPUS" >&2; exit 1; }

ARGS=(--corpus_path "$CORPUS" --topk "$TOPK")
if [ "$MULTI" = "1" ]; then
  [ -n "$INDEX" ] && [ -n "$BM25_INDEX" ] || {
    echo "ERROR: --multi needs both --index (dense) and --bm25-index (sparse)." >&2; exit 1; }
  ARGS+=(--index_path "$INDEX" --retriever_name "$RETRIEVER" --retriever_model "$RETRIEVER_MODEL"
         --multi --bm25_index_path "$BM25_INDEX")
elif [ -n "$BM25_INDEX" ] && [ -z "$INDEX" ]; then
  ARGS+=(--index_path "$BM25_INDEX" --retriever_name bm25)
else
  [ -n "$INDEX" ] || { echo "ERROR: --index is required for the dense backend" >&2; exit 1; }
  ARGS+=(--index_path "$INDEX" --retriever_name "$RETRIEVER" --retriever_model "$RETRIEVER_MODEL")
fi
if [ "$FAISS_GPU" = "1" ]; then
  ARGS+=(--faiss_gpu)
  # FAISS SHARDS the index across every visible GPU, so what matters is index_size / n_visible_gpus, plus
  # whatever else lives on those devices. Pinning a 60 GB flat index to one card OOMs; spreading it over 8
  # costs ~7.5 GB each, which fits under a vLLM at --gpu-mem-util 0.7. Warn rather than fail, because the
  # right answer depends on what else you are running.
  if [ -n "$INDEX" ] && [ -f "$INDEX" ]; then
    NGPU=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c . || true)
    [ "${NGPU:-0}" -eq 0 ] && NGPU=$( (nvidia-smi -L 2>/dev/null || true) | wc -l )
    if [ "${NGPU:-0}" -gt 0 ]; then
      PER=$(( $(stat -c%s "$INDEX") / NGPU / 1024 / 1024 / 1024 ))
      echo "  GPU-FAISS: ~${PER} GB per GPU across ${NGPU} visible device(s)"
      [ "$PER" -gt 30 ] && echo "  !! that is a lot per card -- expect an OOM at load unless these GPUs are near-empty." \
                               "Give it more devices (--gpu-devices), use --what hnsw, or drop --faiss-gpu."
    fi
  fi
fi

# BM25 runs on Lucene through Pyserini, so a JDK has to be on PATH.
if [ "$MULTI" = "1" ] || [ -n "$BM25_INDEX" ]; then
  if ! command -v java >/dev/null 2>&1 && [ -z "${JAVA_HOME:-}" ]; then
    echo "ERROR: the sparse (BM25) backend needs a JDK, and none is on PATH." >&2
    echo "       Install JDK 21 (e.g. Eclipse Temurin) and set JAVA_HOME." >&2
    exit 1
  fi
  # Pyserini constructs an `openai.OpenAI()` client at IMPORT time (pyserini.encode._openai), which raises
  # if no key is set -- so `from pyserini.search.lucene import LuceneSearcher` dies before BM25 does
  # anything, with an error about OpenAI credentials that has nothing to do with what you asked for.
  # Nothing here calls OpenAI; a placeholder is enough. Never clobber a real key.
  export OPENAI_API_KEY="${OPENAI_API_KEY:-unused-by-rpl}"
fi

if [ -n "$GPU_DEVICES" ]; then export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"; fi
export no_proxy="127.0.0.1,localhost,0.0.0.0" NO_PROXY="127.0.0.1,localhost,0.0.0.0"
# The server alone defaults to 0 (no batching); the reported runs served with 0.01, which turns the
# concurrent single-query calls from rollouts into one encode+search batch (~4-5x faster rollouts).
export RETRIEVER_BATCH_WINDOW_S="$BATCH_WINDOW"

if [ "$MULTI" = "1" ]; then BACKEND="dense ($RETRIEVER) + sparse (bm25)"
elif [ -n "$BM25_INDEX" ] && [ -z "$INDEX" ]; then BACKEND="sparse (bm25)"
else BACKEND="dense ($RETRIEVER)"; fi
echo "== retriever: $BACKEND on :8000 =="
if [ "$FOREGROUND" = "1" ]; then
  exec python "$REPO_ROOT/rpl/search/retrieval_server.py" "${ARGS[@]}"
fi

nohup python "$REPO_ROOT/rpl/search/retrieval_server.py" "${ARGS[@]}" > "$LOG" 2>&1 &
SRV_PID=$!
echo "  pid $SRV_PID -> $LOG"
echo "  waiting for the index to load (minutes for a flat index)..."
PROBE='{"queries":["who wrote hamlet"],"topk":1,"return_scores":false}'
for i in $(seq 1 $((WAIT_MIN * 6))); do
  if curl -sf --noproxy '*' http://127.0.0.1:8000/retrieve -X POST \
       -H 'Content-Type: application/json' -d "$PROBE" >/dev/null 2>&1; then
    echo "  RETRIEVER UP after ~$((i * 10))s"
    exit 0
  fi
  sleep 10
  # A dead process will never answer; say so immediately instead of waiting out the budget.
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "ERROR: the retriever process exited after ~$((i * 10))s. Last lines of $LOG:" >&2
    tail -30 "$LOG" >&2; exit 1
  fi
done
# Still alive but not answering: loading, not broken. Do not call that a failure without saying so.
echo "ERROR: the retriever is still not answering after ${WAIT_MIN} min, but the process (pid $SRV_PID)" >&2
echo "       is ALIVE -- it is probably still loading. Raise --wait, or watch $LOG." >&2
tail -15 "$LOG" >&2
exit 1
