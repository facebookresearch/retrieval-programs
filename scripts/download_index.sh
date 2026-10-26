#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Download the wiki-18 corpus and a prebuilt retrieval index from HuggingFace.
#
# RPL needs BOTH a dense and a sparse index: `dense_retrieval()` and `sparse_retrieval()` route to the two
# backends of one retriever process.
#
#   scripts/download_index.sh --out indices --what corpus,bm25,flat

set -euo pipefail

OUT=""
WHAT="corpus,bm25,flat"

function usage {
  cat <<EOF
Usage: $(basename "$0") --out DIR [--what LIST]

  --out DIR     where to put the corpus and indices                          [required]
  --what LIST   comma-separated, any of:
                  corpus  wiki-18 passage corpus (~5 GB)      -- always needed
                  bm25    Pyserini/Lucene sparse index (~2 GB)
                  flat    exact E5 FAISS index (~64 GB)
                  hnsw    approximate E5 FAISS index (~71 GB; less memory to serve)
                (default: ${WHAT})
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --out)     OUT="$2"; shift 2 ;;
    --what)    WHAT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)         echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] || { echo "ERROR: --out is required" >&2; usage; exit 2; }
mkdir -p "$OUT"

function want { case ",$WHAT," in *,"$1",*) return 0 ;; *) return 1 ;; esac; }

if want corpus; then
  echo "== corpus =="
  python - "$OUT" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="PeterJinGo/wiki-18-corpus", filename="wiki-18.jsonl.gz",
                repo_type="dataset", local_dir=sys.argv[1])
PY
  [ -f "$OUT/wiki-18.jsonl" ] || gunzip -k "$OUT/wiki-18.jsonl.gz"
  echo "  $OUT/wiki-18.jsonl"
fi

if want bm25; then
  echo "== BM25 index =="
  python - "$OUT" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="PeterJinGo/wiki-18-bm25-index", repo_type="dataset",
                  local_dir=sys.argv[1], allow_patterns=["bm25/*"])
PY
  echo "  $OUT/bm25"
fi

if want flat; then
  echo "== dense E5 flat index (~64 GB, downloaded in two parts) =="
  if [ ! -f "$OUT/e5_Flat.index" ]; then
    python - "$OUT" <<'PY'
import sys
from huggingface_hub import hf_hub_download
for part in ("part_aa", "part_ab"):
    hf_hub_download(repo_id="PeterJinGo/wiki-18-e5-index", filename=part,
                    repo_type="dataset", local_dir=sys.argv[1])
PY
    cat "$OUT"/part_a* > "$OUT/e5_Flat.index" && rm -f "$OUT"/part_a*
  fi
  # A truncated index is the failure worth catching here: FAISS reports a read error, the retriever dies,
  # and training then proceeds against a retriever that answers nothing.
  SZ=$(stat -c%s "$OUT/e5_Flat.index")
  EXPECT=64559075328
  if [ "$SZ" -lt "$EXPECT" ]; then
    echo "ERROR: $OUT/e5_Flat.index is truncated ($SZ < $EXPECT bytes). Delete it and re-run." >&2
    exit 1
  fi
  echo "  $OUT/e5_Flat.index ($((SZ / 1024 / 1024 / 1024)) GB)"
fi

if want hnsw; then
  echo "== dense E5 HNSW64 index (~71 GB) =="
  python - "$OUT" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="PeterJinGo/wiki-18-e5-index-HNSW64", repo_type="dataset",
                  local_dir=sys.argv[1], allow_patterns=["part_*"])
PY
  [ -f "$OUT/e5_HNSW64.index" ] || { cat "$OUT"/part_* > "$OUT/e5_HNSW64.index" && rm -f "$OUT"/part_*; }
  echo "  $OUT/e5_HNSW64.index"
fi

echo "== done -> $OUT =="
