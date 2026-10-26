# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adapted from Search-R1 (https://github.com/PeterGriffinJin/Search-R1), licensed under the
# Apache License, Version 2.0; see LICENSE-APACHE in the root directory of this source tree.

"""Cross-encoder reranking server.

Serves POST /rerank: given per-query candidate doc lists, scores each (query, doc) pair and returns the docs
re-sorted best-first (optionally truncated to `rerank_topk`). Backends via `--reranker_type`:
  - `sentence_transformer`: a classic cross-encoder (e.g. ms-marco MiniLM).
  - `qwen3` / `causal_lm`: Qwen3-Reranker, a causal LM scored as P("yes") from the last-token yes/no logits
    over the model's judge template (system + <Instruct>/<Query>/<Document> + assistant). The instruction
    matters -- the model is trained with it, so omitting it degrades scores (Qwen3-Reranker model card).

Binds `SERVER_BIND_HOST` (`::` for a routable standalone serve job, else `0.0.0.0`) on :6980.
"""
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
import hashlib
import itertools
import os
import queue
import threading
import time
from typing import Optional

from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import torch
from transformers import HfArgumentParser
import uvicorn

# Official Qwen3-Reranker judge template (model card).
_QWEN3_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
                 "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
                 "<|im_start|>user\n")
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# Default task instruction (model card). Qwen3-Reranker is trained with an instruction; this default fits our
# web-search QA setting.
_QWEN3_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def _qwen3_prompt(instruction: str, query: str, doc: str) -> str:
    return f"{_QWEN3_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{_QWEN3_SUFFIX}"


class BaseCrossEncoder:
    # Class-level defaults rather than __init__ state: the Qwen3 backends do not chain to this __init__, so
    # these keep _predict_cached() valid on an instance whose cache was never enabled.
    _cache = None
    _cache_lock = None
    _cache_max = 0
    _stat_pairs = 0
    _stat_hits = 0
    _stat_fwd = 0

    def __init__(self, model, batch_size=32, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.model.to(device)

    def enable_cache(self, max_entries: int) -> None:
        """Enable the (query, doc) -> score LRU.

        The score is a deterministic function of the pair text for a fixed instruction/max_length (both set at
        construction), so a hit returns exactly the score a forward pass would produce -- this changes
        throughput only, never the scores the policy sees. Note that hit rate is set by how often the policy
        re-emits a query, not by how often a question recurs: the rerank query is model-generated.
        """
        if max_entries <= 0:
            return
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_max = max_entries

    @staticmethod
    def _cache_key(query: str, doc: str) -> bytes:
        # Digest, not the raw text: wiki passages run ~1KB, so raw keys would cost GBs at useful cache sizes.
        # 16 bytes (128-bit) keeps the LRU at ~3MB for 200k entries; a collision would silently return the
        # wrong score, but the birthday odds at that size are ~1e-28. The NUL separator stops ("ab","c") and
        # ("a","bc") hashing alike -- safe because neither a query nor a wiki passage contains NUL.
        return hashlib.blake2b(f"{query}\x00{doc}".encode(), digest_size=16).digest()

    def cache_stats(self) -> dict:
        n, h, f = self._stat_pairs, self._stat_hits, self._stat_fwd
        return {
            "pairs": n, "hits": h, "forwards": f,
            "hit_rate": round(h / n, 3) if n else 0.0,
            "dedup_x": round(n / f, 2) if f else 0.0,
            "entries": len(self._cache) if self._cache is not None else 0,
        }

    def _predict_cached(self, pairs: list[tuple[str, str]]):
        """`_predict` with the LRU in front; also collapses pairs duplicated within a single request."""
        if self._cache is None:
            self._stat_pairs += len(pairs)
            self._stat_fwd += len(pairs)
            return self._predict(pairs)
        out = [None] * len(pairs)
        todo = {}  # cache key -> (pair, [indices in out])
        with self._cache_lock:
            for i, (q, d) in enumerate(pairs):
                key = self._cache_key(q, d)
                score = self._cache.get(key)
                if score is None:
                    todo.setdefault(key, (pairs[i], []))[1].append(i)
                else:
                    self._cache.move_to_end(key)
                    out[i] = score
            missed = sum(len(idxs) for _p, idxs in todo.values())
            self._stat_pairs += len(pairs)
            self._stat_hits += len(pairs) - missed
            self._stat_fwd += len(todo)
        if todo:
            # Forward OUTSIDE the lock -- _predict blocks on the GPU worker queue. Two requests racing on the
            # same miss just score it twice; the result is identical either way.
            items = list(todo.items())
            scores = self._predict([pair for _key, (pair, _idxs) in items])
            with self._cache_lock:
                for (key, (_pair, idxs)), score in zip(items, scores):
                    for i in idxs:
                        out[i] = score
                    self._cache[key] = score
                    self._cache.move_to_end(key)
                while len(self._cache) > self._cache_max:
                    self._cache.popitem(last=False)
        return out

    def _passage_to_string(self, doc_item):
        content = doc_item["contents"] if "document" not in doc_item else doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        return f"(Title: {title}) {text}"

    def rerank(self, queries: list[str], documents: list[list[dict]]):
        """Rerank each query's own candidate list; return {qid: [(doc_item, score), ...]} best-first.

        `documents[qid]` is the candidate list for `queries[qid]` (aligned, one list per query). The scored
        rows carry the ORIGINAL doc dict (with `id`/`contents`), not the flattened string, so the caller keeps
        doc identity/structure.
        """
        assert len(queries) == len(documents)
        pairs = []
        owners = []  # owners[i] = (qid, original doc_item) aligned to pairs[i]
        for qid, query in enumerate(queries):
            for doc_item in documents[qid]:
                pairs.append((query, self._passage_to_string(doc_item)))
                owners.append((qid, doc_item))
        scores = self._predict_cached(pairs) if pairs else []
        assert len(scores) == len(pairs) == len(owners)
        query_to_doc_scores = defaultdict(list)
        for (qid, doc_item), score in zip(owners, scores):
            query_to_doc_scores[qid].append((doc_item, score))
        return {qid: sorted(ds, key=lambda x: x[1], reverse=True) for qid, ds in query_to_doc_scores.items()}

    def _predict(self, pairs: list[tuple[str, str]]):
        raise NotImplementedError

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        raise NotImplementedError


class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    def _predict(self, pairs: list[tuple[str, str]]):
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return scores.tolist() if isinstance(scores, (torch.Tensor, np.ndarray)) else scores

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        from sentence_transformers import CrossEncoder  # lazy: only this backend needs sentence_transformers
        return cls(CrossEncoder(model_name_or_path), **kwargs)


class Qwen3Reranker(BaseCrossEncoder):
    """Qwen3-Reranker (transformers): relevance = P("yes") from the last-token yes/no logits over the model's
    judge template (system + <Instruct>/<Query>/<Document> + assistant). Per the model card.

    Cross-request micro-batching: the model is replicated on ALL visible GPUs, and concurrent /rerank requests
    enqueue their (query, doc) pairs into one shared queue. One worker thread per GPU pulls up to
    `max_batch_pairs` pending pairs (waiting at most `batch_window_s` to coalesce across requests), runs ONE
    batched forward on its replica, and scatters scores back to the waiting requests. This keeps every GPU fed
    with pairs from MANY requests instead of one-request-per-GPU, so the reranker sustains the RL rollout herd
    (thousands of rerank calls burst per training step) with each call staying well under the timeout.
    With a single visible GPU (on-node co-located reranker) there is one worker -> forwards serialize, bounding
    peak memory on the shared GPU."""

    def __init__(self, models, tokenizer, batch_size=64, max_length=512, instruction=None,
                 batch_window_s=0.01, max_batch_pairs=256):
        self.models = models  # {device_str: model_replica}
        self.tokenizer = tokenizer
        self.batch_size = batch_size  # per-forward GPU micro-batch (bounds peak activation memory)
        self.max_length = max_length
        self.instruction = instruction or _QWEN3_DEFAULT_INSTRUCTION
        self._yes = tokenizer.convert_tokens_to_ids("yes")
        self._no = tokenizer.convert_tokens_to_ids("no")
        # Shared pair queue drained by one worker thread per GPU replica; items are (prompt_text, out_idx, state).
        self._batch_window_s = batch_window_s
        self._max_batch_pairs = max_batch_pairs
        self._q: "queue.Queue" = queue.Queue()
        for dev in models:
            threading.Thread(target=self._worker, args=(dev,), daemon=True).start()

    @torch.no_grad()
    def _forward(self, model, dev, texts: list[str]) -> list[float]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(dev)
            last = model(**enc, logits_to_keep=1).logits[:, -1, :]  # last-token logits only; the full
            # (B,S,V) logits (~5GiB at B=32,S=512,V=152k) would blow up GPU memory
            two = torch.stack([last[:, self._no], last[:, self._yes]], dim=1)
            # softmax(.)[:,1] == log_softmax(.)[:,1].exp(): the same P("yes") the model card computes.
            out.extend(torch.softmax(two, dim=1)[:, 1].float().cpu().tolist())
        return out

    def _worker(self, dev) -> None:
        """One per GPU: coalesce pending pairs across requests into a batch, forward on this replica, scatter."""
        model = self.models[dev]
        while True:
            batch = [self._q.get()]  # block for the first item
            deadline = time.monotonic() + self._batch_window_s
            while len(batch) < self._max_batch_pairs:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                scores = self._forward(model, dev, [text for text, _idx, _st in batch])
            except Exception:  # never wedge the queue on a bad batch -> neutral scores unblock the requests
                scores = [0.0] * len(batch)
            for (_text, idx, st), s in zip(batch, scores):
                with st["lock"]:
                    st["out"][idx] = s
                    st["rem"] -= 1
                    if st["rem"] == 0:
                        st["ev"].set()

    def _predict(self, pairs: list[tuple[str, str]]):
        if not pairs:
            return []
        st = {"out": [0.0] * len(pairs), "rem": len(pairs), "lock": threading.Lock(), "ev": threading.Event()}
        for idx, (q, d) in enumerate(pairs):
            self._q.put((_qwen3_prompt(self.instruction, q, d), idx, st))
        st["ev"].wait()
        return st["out"]

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", trust_remote_code=True)
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        devices = [f"cuda:{i}" for i in range(n)] if n else ["cpu"]
        models = {}
        for dev in devices:
            models[dev] = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        score_batch = int(os.environ.get("RERANK_SCORE_BATCH", "64"))
        window = float(os.environ.get("RERANK_BATCH_WINDOW_S", "0.01"))
        max_pairs = int(os.environ.get("RERANK_MAX_BATCH_PAIRS", "256"))
        print(f"Qwen3Reranker: loaded {len(models)} replica(s) on {devices}; "
              f"batching window={window}s max_batch_pairs={max_pairs} score_batch={score_batch}", flush=True)
        return cls(models, tok, batch_size=score_batch,
                   max_length=kwargs.get("max_length", 512), instruction=kwargs.get("instruction"),
                   batch_window_s=window, max_batch_pairs=max_pairs)


class RerankRequest(BaseModel):
    queries: list[str]
    documents: list[list[dict]]
    rerank_topk: Optional[int] = None
    return_scores: bool = False


@dataclass
class RerankerArguments:
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    # sentence_transformer | qwen3 (== causal_lm)
    reranker_type: str = field(default="sentence_transformer")
    # Task instruction for the Qwen3 backends; None -> the model-card default (fits web-search QA).
    instruction: Optional[str] = field(default=None)

    # Entries in the (query, doc) -> score LRU. GRPO issues `rollout.n` identical rerank() calls per question
    # and val questions recur at every eval, so this cuts cross-encoder forwards several-fold. 0 disables.
    rerank_cache_size: int = field(default=200_000)


def get_reranker(config):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if config.reranker_type == "sentence_transformer":
        reranker = SentenceTransformerCrossEncoder.load(config.rerank_model_name_or_path,
                                                        batch_size=config.batch_size, device=dev)
    elif config.reranker_type in ("qwen3", "causal_lm"):
        reranker = Qwen3Reranker.load(config.rerank_model_name_or_path, batch_size=config.batch_size,
                                      max_length=config.max_length, device=dev, instruction=config.instruction)
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")
    reranker.enable_cache(config.rerank_cache_size)
    print(f"rerank pair cache: max_entries={config.rerank_cache_size}", flush=True)
    return reranker


app = FastAPI()
_req_count = itertools.count(1)


@app.get("/stats")
def stats_endpoint():
    """Cache/throughput counters -- `dedup_x` is the factor by which the cache cut cross-encoder forwards."""
    return reranker.cache_stats()


@app.post("/rerank")
def rerank_endpoint(request: RerankRequest):
    """Rerank each query's candidate docs; return the top `rerank_topk` best-first per query.

    Input: {"queries": [...], "documents": [[doc, ...], ...], "rerank_topk": 3, "return_scores": true}.
    With `return_scores`, each returned doc is `{"document": <doc>, "score": <float>}`; otherwise the bare doc.
    """
    topk = request.rerank_topk or config.rerank_topk
    query_to_doc_scores = reranker.rerank(request.queries, request.documents)  # sorted best-first per query
    resp = []
    for qid in range(len(request.queries)):  # by index -> stays aligned even if a query had no candidates
        doc_scores = query_to_doc_scores.get(qid, [])[:topk]
        if request.return_scores:
            resp.append([{"document": doc, "score": score} for doc, score in doc_scores])
        else:
            resp.append([doc for doc, _ in doc_scores])
    if next(_req_count) % 500 == 0:
        print(f"rerank cache {reranker.cache_stats()}", flush=True)
    return {"result": resp}


if __name__ == "__main__":
    config = HfArgumentParser(RerankerArguments).parse_args_into_dataclasses()[0]
    reranker = get_reranker(config)  # load once, reuse across requests
    # SERVER_BIND_HOST=:: (dual-stack) for a standalone serve job so the routable task IPv6 is reachable;
    # defaults to 0.0.0.0 for the on-node/localhost path.
    uvicorn.run(app, host=os.environ.get("SERVER_BIND_HOST", "0.0.0.0"), port=6980)
