#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adapted from Search-R1 (https://github.com/PeterGriffinJin/Search-R1), licensed under the
# Apache License, Version 2.0; see LICENSE-APACHE in the root directory of this source tree.

# Build the dense (FAISS) and sparse (BM25) indices that scripts/serve_retriever.sh serves, over a corpus JSONL
# with one {"id", "contents"} per line. Run it in the retrieval environment.
#
#   rpl/search/build_index.sh --corpus indices/wiki-18.jsonl --out indices          # e5, as Search-R1 (hours)
#   rpl/search/build_index.sh --corpus globalqa/corpus.jsonl --out indices_globalqa --retriever bge
#
# Pooling follows the retriever (e5 -> mean, bge -> cls), from the same table the server pools queries with: an
# index pooled unlike its queries still serves, just badly. The BM25 index stores no raw text, so the server
# reads each hit's document from the corpus by its id, as a line number -- the corpus must list ids 0..N-1 in
# order, which GlobalQA's corpus.jsonl does and this script checks.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS=""
OUT=""
RETRIEVER=e5
MODEL=""
WHAT=dense,bm25
FAISS_TYPE=Flat

function usage {
  cat <<EOF
Usage: $(basename "$0") --corpus FILE --out DIR [options]

  --corpus FILE      corpus JSONL, one {"id", "contents"} per line, ids 0..N-1 in order  [required]
  --out DIR          writes DIR/<retriever>_<faiss type>.index and DIR/bm25              [required]
  --retriever NAME   e5 | bge (default: ${RETRIEVER}); the reported GlobalQA runs used bge
  --model M          encoder checkpoint (default: intfloat/e5-base-v2 or BAAI/bge-base-en-v1.5)
  --what LIST        dense,bm25 or either one (default: ${WHAT})
  --faiss-type T     Flat (exact, what the reported runs used) or e.g. HNSW64 (default: ${FAISS_TYPE})
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --corpus)     CORPUS="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --retriever)  RETRIEVER="$2"; shift 2 ;;
    --model)      MODEL="$2"; shift 2 ;;
    --what)       WHAT="$2"; shift 2 ;;
    --faiss-type) FAISS_TYPE="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    *)            echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$CORPUS" ] && [ -n "$OUT" ] || { usage; exit 2; }
[ -f "$CORPUS" ] || { echo "ERROR: corpus not found: $CORPUS" >&2; exit 1; }
case "$RETRIEVER" in
  e5)  MODEL="${MODEL:-intfloat/e5-base-v2}" ;;
  bge) MODEL="${MODEL:-BAAI/bge-base-en-v1.5}" ;;
  *)   echo "ERROR: --retriever must be e5 or bge" >&2; exit 2 ;;
esac
mkdir -p "$OUT"

if [[ ",$WHAT," == *,dense,* ]]; then
  echo "== dense: $RETRIEVER ($MODEL), $FAISS_TYPE -> $OUT/${RETRIEVER}_${FAISS_TYPE}.index"
  python "$HERE/index_builder.py" --retrieval_method "$RETRIEVER" --model_path "$MODEL" --corpus_path "$CORPUS" \
    --save_dir "$OUT" --use_fp16 --max_length 256 --batch_size 512 --faiss_type "$FAISS_TYPE"
fi
if [[ ",$WHAT," == *,bm25,* ]]; then
  python - "$CORPUS" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    for i, line in enumerate(line for line in f if line.strip()):
        if str(json.loads(line)["id"]) != str(i):
            sys.exit(f"ERROR: line {i} of {sys.argv[1]} has id {json.loads(line)['id']!r}; BM25 hits are read back "
                     "by line number, so the corpus must list ids 0..N-1 in order")
PY
  echo "== sparse: bm25 -> $OUT/bm25"
  python "$HERE/index_builder.py" --retrieval_method bm25 --corpus_path "$CORPUS" --save_dir "$OUT"
fi

echo "Serve with:"
echo "  scripts/serve_retriever.sh --index $OUT/${RETRIEVER}_${FAISS_TYPE}.index --corpus $CORPUS \\"
echo "      --bm25-index $OUT/bm25 --multi --retriever $RETRIEVER --retriever-model $MODEL"
