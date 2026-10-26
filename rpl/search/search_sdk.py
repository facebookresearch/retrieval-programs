# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""In-sandbox search primitives for Retrieval Program Learning (RPL).

Model-generated code (run in the subprocess sandbox) uses these to retrieve from the corpus.
Uses stdlib-only HTTP client to the retrieval server, client-side arg validation
(ValueError before the round-trip), bounded retry, and normalized result rows.

Env config: RETRIEVER_URL (default http://127.0.0.1:8000/retrieve), RERANK_URL
(default http://127.0.0.1:6980/rerank), TOPK_DEFAULT (3), MAX_SEARCH_CALLS (0 = unlimited).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

_MAX_TOP_K = 200
_MAX_RETRIES = 6
_INITIAL_BACKOFF_SEC = 0.5

_search_calls = 0
# Per-method call counts (reset per block by the sandbox) for the complementary-search reward + routing analytics.
_dense_calls = 0
_sparse_calls = 0
_rerank_calls = 0   # rerank() calls (post-processing, not retrieval); tracked for analytics, not the search budget
_search_time = 0.0  # cumulative retriever HTTP time (s); reset per block by the sandbox for latency profiling
# Per-call retrieved / reranked doc contents ("title\ntext"), reset per block by the sandbox and shipped OUT in
# the envelope so the reward layer -- which holds the gold label (NEVER the sandbox) -- can score answer
# recall/precision without leaking the label into model code. Bounded to cap the shipped payload.
_call_docs: list = []          # one entry per retrieval call: [doc_contents, ...]
_rerank_call_docs: list = []   # one entry per rerank() call: [doc_contents, ...]
# Corpus ids of the retrieved docs, in retrieval order, one entry per call. Kept SEPARATELY from the text
# buckets and unbounded by bytes: ids are a few characters each, and truncating them would silently deflate
# the GlobalQA document-coverage metric (MARAG-R1's D-F1@k), which is scored against `golden_doc_ids`.
_call_doc_ids: list = []
_MAX_DOC_CHARS = 800                # cap per recorded doc (wiki-18 passages are shorter; bounds payload)
_MAX_DOCS_PER_CALL = 20             # cap docs recorded per call
_MAX_RECORD_BYTES = 32 * 1024  # cap recorded doc text PER bucket (retrieval and rerank budgeted separately)


def _record_docs(bucket: list, docs: list) -> None:
    """Record this call's doc contents into `bucket` for out-of-sandbox recall/precision."""
    if sum(len(t) for call in bucket for t in call) >= _MAX_RECORD_BYTES:
        return
    bucket.append([f"{d.get('title', '')}\n{d.get('text', '')}"[:_MAX_DOC_CHARS] for d in docs[:_MAX_DOCS_PER_CALL]])


def _retriever_url() -> str:
    return os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000/retrieve")


def _rerank_url() -> str:
    return os.environ.get("RERANK_URL", "http://127.0.0.1:6980/rerank")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _check_query(query) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return query


def _check_top_k(top_k) -> int:
    if not isinstance(top_k, int) or isinstance(top_k, bool):  # bool subclasses int; True/False are not valid top_k
        raise ValueError("top_k must be an int")
    if not (1 <= top_k <= _MAX_TOP_K):
        raise ValueError(f"top_k must be in [1, {_MAX_TOP_K}], got {top_k}")
    return top_k


def _post(url: str, payload: dict, timeout_sec: float = 60.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise  # deterministic server error: don't retry
        except (OSError, urllib.error.URLError) as exc:
            last = exc
            if attempt >= _MAX_RETRIES:
                break
            time.sleep(_INITIAL_BACKOFF_SEC * (2 ** attempt))
    raise RuntimeError(f"retriever request failed after {_MAX_RETRIES + 1} attempts: {last}")


def _normalize(doc_row: dict, score) -> dict:
    # /retrieve rows are {"id", "contents"}; contents is "<title>\n<text...>".
    contents = doc_row.get("contents", "") or ""
    lines = contents.split("\n")
    title = lines[0].strip('"') if lines and lines[0] else ""
    text = "\n".join(lines[1:]) if len(lines) > 1 else ""
    return {"id": doc_row.get("id"), "title": title, "text": text, "score": score}


def _retrieve(query: str, top_k: int | None = None, method: str | None = None) -> list:
    """Shared retrieval body for `search`/`dense_retrieval`/`sparse_retrieval`.

    All three share the `_search_calls`/`_search_time` counters, so the reward gate counts any retrieval
    regardless of method. `method` (dense|sparse) routes the merged multi-retriever server; it is omitted
    from the payload when None so single-retriever servers see an unchanged request.

    Returns a list of {"id", "title", "text", "score"} dicts, best first.
    """
    global _search_calls, _dense_calls, _sparse_calls, _search_time
    query = _check_query(query)
    top_k = _check_top_k(_int_env("TOPK_DEFAULT", 3) if top_k is None else top_k)
    cap = _int_env("MAX_SEARCH_CALLS", 0)
    if cap and _search_calls >= cap:
        raise RuntimeError(f"search() call budget exhausted (max_search_calls={cap})")
    _search_calls += 1
    if method == "dense":
        _dense_calls += 1
    elif method == "sparse":
        _sparse_calls += 1
    payload = {"queries": [query], "topk": top_k, "return_scores": True}
    if method is not None:
        payload["method"] = method
    _t0 = time.perf_counter()
    resp = _post(_retriever_url(), payload)
    _search_time += time.perf_counter() - _t0
    result = (resp.get("result") or [[]])[0]
    out = []
    for item in result:
        doc_row = item.get("document", item)  # return_scores=True -> {"document":..., "score":...}
        out.append(_normalize(doc_row, item.get("score")))
    _record_docs(_call_docs, out)
    _call_doc_ids.append([d["id"] for d in out if d.get("id") is not None])
    return out


def search(query: str, top_k: int | None = None) -> list:
    """Retrieve the top_k most relevant documents for `query` (server default backend)."""
    return _retrieve(query, top_k)


def dense_retrieval(query: str, top_k: int | None = None) -> list:
    """Semantic/dense retrieval (e5/faiss): best for paraphrase / "what is" / "who is" questions."""
    return _retrieve(query, top_k, method="dense")


def sparse_retrieval(query: str, top_k: int | None = None) -> list:
    """Lexical/sparse retrieval (bm25): best for exact names, numbers, rare terms, quoted strings."""
    return _retrieve(query, top_k, method="sparse")


# ---- post-processing primitives (pure Python; no retrieval, so they don't touch the search-call budget) ----

_RRF_K = 60  # reciprocal-rank-fusion constant (standard default; larger => flatter rank weighting)


def _doc_key(doc: dict):
    """Identity for dedup/fusion: the corpus `id`, falling back to (title, text) when `id` is missing."""
    key = doc.get("id")
    return key if key is not None else (doc.get("title"), doc.get("text"))


def _require_doc_list(obj, where: str, hint: str) -> None:
    """Reject a document passed where a list of documents belongs, naming the fix in one line.

    The sandbox reports only the last traceback line, so an error has to be self-contained to teach
    anything. Without this, `fuse(*flat_doc_list)` hands `_doc_key` a dict's string keys and dies three
    frames down with "'str' object has no attribute 'get'", which names neither the primitive nor the
    mistake -- and a policy that cannot tell "called it wrong" from "it is broken" stops calling it at all.
    """
    if not isinstance(obj, (list, tuple)):
        raise TypeError(f"{where} must be a list of documents, got {type(obj).__name__}. {hint}")
    if obj and not isinstance(obj[0], dict):
        raise TypeError(f"{where} must contain document dicts, got {type(obj[0]).__name__}. {hint}")


def dedup(docs: list) -> list:
    """Drop duplicate documents from a single list, keeping the first occurrence and original order.

    Documents are compared by corpus `id`. Useful after issuing several queries to one retriever (e.g. a
    decomposed multi-hop question) before reading.
    """
    _require_doc_list(docs, "dedup() argument", "Pass one list, e.g. dedup(dense_retrieval(q, top_k=20)).")
    seen = set()
    out = []
    for doc in docs:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def fuse(*doc_lists: list, method: str = "rrf") -> list:
    """Merge several ranked document lists into one deduplicated list, best first, via reciprocal rank fusion.

    Dense and sparse scores are not comparable, so RRF fuses by RANK, not raw score: a document's fused score is
    the sum over the input lists of `1 / (K + rank)`, with rank 1-based and `K=60`. Documents are keyed by corpus
    `id`, deduplicated across lists, and returned in descending fused score (written to each row's `score`). Use
    it to combine dense and sparse results, or per-subquery results from a decomposed question, e.g.
    `fuse(dense_retrieval(q), sparse_retrieval(q))`.
    """
    if method != "rrf":
        raise ValueError(f"unsupported fuse method: {method!r} (only 'rrf' is supported)")
    hint = ("Use fuse(dense_results, sparse_results); write fuse(*x) only when every element of x is itself "
            "a list, e.g. built with x.append(dense_retrieval(q, top_k=20)).")
    for i, docs in enumerate(doc_lists, start=1):
        _require_doc_list(docs, f"fuse() argument {i}", hint)
    fused_score: dict = {}
    first_doc: dict = {}
    for docs in doc_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _doc_key(doc)
            fused_score[key] = fused_score.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            first_doc.setdefault(key, doc)
    out = []
    for key in sorted(fused_score, key=lambda k: fused_score[k], reverse=True):
        doc = dict(first_doc[key])
        doc["score"] = round(fused_score[key], 6)
        out.append(doc)
    return out


# Relevance probability above which a document is KEPT when `rerank()` is called without `top_k`.
# 0.5 is not a tuned cutoff: the qwen3 reranker backend scores a pair as P("yes") from a two-way softmax
# over the yes/no tokens (`et / (et + ef)` in rerank_server.py), answering "does this document contain
# information to answer the query?".
_RERANK_THRESHOLD = float(os.environ.get("RERANK_RELEVANCE_THRESHOLD", "0.5"))


def _keep_relevant(ranked: list) -> list:
    """Keep the documents the reranker judged relevant, however many that is."""
    return [d for d in ranked if d.get("score") is None or float(d["score"]) > _RERANK_THRESHOLD]


def rerank(query: str, docs: list, top_k: int | None = None) -> list:
    """Reorder `docs` by relevance to `query` with the cross-encoder rerank server, best first.

    Use this to turn HIGH RECALL into HIGH PRECISION: retrieve/fuse a large candidate set (e.g. top_k=20-50),
    then `rerank(query, docs, top_k=3)` to keep only the few genuinely relevant documents and drop the
    distractors before reading. A cross-encoder scores each (query, document) jointly, so it is far more
    accurate than the first-stage retriever's score — reading a short, reranked, on-topic set is easier to
    answer from than a long noisy one.

    Two modes depending on whether `top_k` is passed in:
    * `top_k=N` -- keep the N best documents. A FIXED breadth: you get N whether or not N are relevant.
    * `top_k` omitted -- keep every document the reranker judges RELEVANT, however many that is. The
      breadth adapts to the question: a query with two supporting documents yields two, one with twenty
      yields twenty.

    `docs` are retrieval rows (`{"id","title","text","score"}`, e.g. from `search`/`dense_retrieval`/
    `sparse_retrieval`/`fuse`). Returns the same rows reordered, with `score` replaced by the reranker's
    relevance score.
    """
    global _rerank_calls
    query = _check_query(query)
    if not isinstance(docs, list) or not docs:
        raise ValueError("docs must be a non-empty list")
    if top_k is not None:
        top_k = _check_top_k(top_k)
    _rerank_calls += 1
    # Cap the candidate set.
    max_docs = _int_env("RERANK_MAX_DOCS", 0)
    if max_docs > 0 and len(docs) > max_docs:
        docs = docs[:max_docs]
    payload_docs = [
        {"id": d.get("id"), "contents": f"{d.get('title', '')}\n{d.get('text', '')}"}
        for d in docs
    ]
    # rerank_topk defaults to 3 server-side; pass len(docs) when top_k is None so "no truncation" returns all.
    payload = {
        "queries": [query], "documents": [payload_docs], "return_scores": True,
        "rerank_topk": top_k if top_k is not None else len(docs),
    }
    resp = _post(_rerank_url(), payload)
    ranked = (resp.get("result") or [[]])[0]
    out = []
    for item in ranked:
        doc_row = item.get("document", item)  # return_scores=True -> {"document":..., "score":...}
        out.append(_normalize(doc_row, item.get("score")))
    if top_k is None:
        out = _keep_relevant(out)
    _record_docs(_rerank_call_docs, out)
    return out


"""
MARAG-R1 tool surface (arXiv 2510.27569), for the `marag` mode.

Reimplemented so the architecture can be compared against Retrieval Program Learning with everything else held fixed:
same corpus, retriever, backbone and RL. What distinguishes their method is the INTERFACE, not the tool set.
"""

# Fixed retrieval breadth. Their implementation section states "TopK set to 20 for retrieval",
_MARAG_TOPK = int(os.environ.get("MARAG_TOPK", "20"))
# 0 (default) = the predicate filter, matching their F_DF.
_MARAG_FILTER_FRAC = float(os.environ.get("MARAG_FILTER_FRAC", "0"))


def DenseSearch(query: str) -> list:
    """Semantic search over the corpus (MARAG-R1's F_DR). Fixed breadth; no `top_k` to choose."""
    return dense_retrieval(query, top_k=_MARAG_TOPK)


def KeywordSearch(query: str) -> list:
    """Keyword search over the corpus (MARAG-R1's F_KR). Fixed breadth; no `top_k` to choose."""
    return sparse_retrieval(query, top_k=_MARAG_TOPK)


def DocumentFilter(query: str, docs: list) -> list:
    """Keep the documents relevant to `query` (MARAG-R1's F_DF)."""
    if not isinstance(docs, list) or not docs:
        raise ValueError("DocumentFilter needs a non-empty list of documents")
    if _MARAG_FILTER_FRAC > 0:
        return rerank(query, docs, top_k=max(1, int(round(len(docs) * _MARAG_FILTER_FRAC))))
    return rerank(query, docs)
