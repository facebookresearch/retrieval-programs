# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the program-mutation scripts.

A rollout's retrieval behavior is fully described by its calls -- (method, query, top_k) -- which are
parseable from the dumped `<code>` blocks. So the pipeline can replay those calls against the retriever and
reranker and check whether gold reached the context, with no code execution and no generation. This module
parses the calls, builds the question join key, tests for gold, and replays calls.
"""

from __future__ import annotations

import ast
import json
import re
import urllib.request

CODE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
# A retrieval call with its own literal query and optional top_k. Only string-literal queries are parsed: a
# query built from a variable cannot be replayed without executing the block, so callers skip such rollouts
# (`has_var_query`) rather than replay them with that call missing.
CALL = re.compile(
    r"\b(dense_retrieval|sparse_retrieval|search)\s*\(\s*(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')"
    r"\s*(?:,\s*top_k\s*=\s*(\d+))?\s*\)"
)
METHOD = {"dense_retrieval": "dense", "sparse_retrieval": "sparse", "search": None}


def _unquote(lit: str) -> str:
    """A quoted query literal, read the way python reads it -- which is how the sandbox runs it.

    `json.loads` is the wrong parser here: the policy emitted `dense_retrieval("Who produced the song
    'Smokin\\' and Drinkin\\'?")`, where `\\'` is a legal Python escape and an illegal JSON one.
    """
    try:
        return ast.literal_eval(lit)
    except (ValueError, SyntaxError):
        return lit[1:-1]  # unterminated / exotic literal: fall back to stripping the quotes


def parse_calls(output: str) -> list[dict]:
    """Retrieval calls in execution order: [{method, query, top_k}]. `method=None` = server default."""
    calls = []
    for block in CODE.findall(output or ""):
        for m in CALL.finditer(block):
            fn, q, k = m.group(1), m.group(2), m.group(3)
            calls.append({"method": METHOD[fn], "query": _unquote(q), "top_k": int(k) if k else 3})
    return calls


def has_var_query(output: str) -> bool:
    """True if some retrieval call's query is not a string literal (so the rollout cannot be replayed)."""
    for block in CODE.findall(output or ""):
        for m in re.finditer(r"\b(?:dense_retrieval|sparse_retrieval|search)\s*\(", block):
            tail = block[m.end():].lstrip()
            if not tail[:1] in ("'", '"'):
                return True
    return False


def golds(gts) -> list[str]:
    """Gold answer strings. Training and val dumps disagree: `{"target": [...]}` vs a bare list/string."""
    if isinstance(gts, dict):
        gts = gts.get("target") or []
    if isinstance(gts, str):
        gts = [gts]
    return [str(g) for g in (gts or []) if str(g).strip()]


ASSIST = re.compile(r"\n\s*assistant\b.*$", re.S)


def norm_question(question: str) -> str:
    """Normalize a raw question string to the join key."""
    return (question or "").strip().lower().rstrip("?").strip()


def qkey(prompt_text: str) -> str:
    """Join key between a dump row and the parquet, matching routing_report/globalqa_report."""
    return norm_question(ASSIST.sub("", (prompt_text or "").rsplit("Question:", 1)[-1]))


def _norm_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def load_supporting(parquet: str) -> dict:
    """{question_key: [normalized supporting-passage titles]} from a nq_hotpotqa parquet."""
    import pyarrow.parquet as pq
    out = {}
    for r in pq.read_table(parquet, columns=["question", "metadata"]).to_pylist():
        sf = ((r.get("metadata") or {}).get("supporting_facts") or {}).get("title") or []
        titles = sorted({_norm_title(x) for x in sf if x})
        if titles:
            out[qkey(r["question"])] = titles
    return out


def titles_of(docs) -> set:
    """Normalized titles of retrieved docs. `contents` is "title\ntext", so the title is its first line."""
    vals = docs.values() if isinstance(docs, dict) else docs
    return {_norm_title(v.split("\n", 1)[0]) for v in vals if v}


def supporting_covered(docs, supporting: list[str]) -> int:
    """How many supporting passages are present among `docs` (substring-tolerant title match)."""
    got = titles_of(docs)
    return sum(1 for s in supporting if any(s == g or s in g or g in s for g in got))


def golden_doc_ids(gts) -> set[str]:
    """The GlobalQA gold DOCUMENT set, as strings. Empty when the row carries no such annotation."""
    ids = (gts or {}).get("golden_doc_ids") if isinstance(gts, dict) else None
    return {str(i) for i in (ids or [])}


def doc_f1(docs, gold_ids: set[str]) -> float:
    """F1 of the retrieved document-id set against the gold set."""
    got = {str(k) for k in (docs.keys() if isinstance(docs, dict) else docs)}
    if not got or not gold_ids:
        return 0.0
    hit = len(got & gold_ids)
    if not hit:
        return 0.0
    p, r = hit / len(got), hit / len(gold_ids)
    return 2 * p * r / (p + r)


def answer_in(docs, targets: list[str]) -> bool:
    """Mirror of the reward manager's `_answer_in_text` over a union of docs -- keep these identical.

    Accepts a {id: contents} mapping or any iterable of texts.
    """
    texts = docs.values() if isinstance(docs, dict) else docs
    return any(any(g and g.lower() in t.lower() for g in targets) for t in texts)


def _post(url: str, payload: dict) -> dict:
    """POST json and return the parsed reply. Stdlib only, so this file imports with no third-party deps."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def retrieve(url: str, calls: list[dict]) -> dict:
    """Issue `calls` and return the deduped union as {doc_id: contents}, insertion-ordered by first sighting."""
    seen = {}
    for c in calls:
        payload = {"queries": [c["query"]], "topk": c["top_k"], "return_scores": True}
        if c["method"]:
            payload["method"] = c["method"]
        for item in (_post(url, payload).get("result") or [[]])[0]:
            doc = item.get("document", item)
            seen.setdefault(doc.get("id", doc.get("contents")), doc.get("contents", ""))
    return seen


def prune(rerank_url: str, query: str, docs: dict, keep: int) -> dict:
    """Cross-encoder rerank `docs` down to `keep`."""
    if len(docs) <= keep:
        return docs
    payload_docs = [{"id": k, "contents": v} for k, v in docs.items()]
    payload = {"queries": [query], "documents": [payload_docs], "return_scores": True, "rerank_topk": keep}
    out = {}
    for item in (_post(rerank_url, payload).get("result") or [[]])[0]:
        d = item.get("document", item)
        out[d.get("id", d.get("contents"))] = d.get("contents", "")
    return out


def run_program(url: str, rerank_url: str, calls: list[dict], keep, scope: str = "call",
                rerank_query: str | None = None) -> dict:
    """Execute a call list and return what would reach the context, as {doc_id: contents}.

    `keep=None` is the raw union. An integer prunes per call against that call's own query (`scope="call"`), or
    the fused union once against `rerank_query` (`scope="union"`), the form `rerank(q, fuse(...), top_k=k)`.
    """
    if scope == "union":
        if keep is not None and not rerank_query:
            raise ValueError("a union-scope rerank needs the query it reranks against")
        union = retrieve(url, calls)
        return prune(rerank_url, rerank_query, union, keep) if keep is not None else union
    if scope != "call":
        raise ValueError(f"unknown rerank scope {scope!r}")
    seen = {}
    for c in calls:
        got = retrieve(url, [c])
        if keep is not None:
            got = prune(rerank_url, c["query"], got, keep)
        for k, v in got.items():
            seen.setdefault(k, v)
    return seen
