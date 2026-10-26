# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adapted from Search-R1 (https://github.com/PeterGriffinJin/Search-R1), licensed under the
# Apache License, Version 2.0; see LICENSE-APACHE in the root directory of this source tree.

import argparse
from collections import defaultdict
import json
import os
import queue as _queuelib
import threading
import time
from typing import Optional
import warnings

import datasets
import faiss
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import uvicorn

# Per-model query pooling. Deliberately DUPLICATED from index_builder.MODEL2POOLING rather than imported:
# this file is launched as a script inside the retriever env (`python .../retrieval_server.py`), so
# `rpl` is not an importable package, and importing the sibling would drag in index_builder's
# `datasets` dependency, which the index-build block installs on demand and the cache-reuse path skips.
# `retrieval_server_pooling_test.py` asserts the two tables stay identical.
MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    "jina": "mean",
}


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        "json",
        data_files=corpus_path,
        split="train",
        num_proc=4,
    )
    return corpus


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs["input_ids"].shape[0], 1), dtype=torch.long
            ).to(inputs["input_ids"].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs["attention_mask"],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()

        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: list[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!")
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())["contents"]
                for hit in hits
            ]
            results = [
                {
                    "title": content.split("\n")[0].strip("\""),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content
                }
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        return results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        return results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


class Config:
    """Minimal config class simulating the argparse."""
    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False
    # Multi-retriever routing: "dense" | "sparse". None -> the server's default; a method the server does not
    # serve is rejected rather than answered by the default backend.
    method: Optional[str] = None


# Method -> backend registry. Populated in __main__: one entry for a single-retriever server, both
# ("dense" + "sparse") for --multi. The /retrieve endpoint and the batcher route on the request method.
RETRIEVERS: dict[str, "BaseRetriever"] = {}
DEFAULT_METHOD = "dense"


def _get_retriever(method: Optional[str]) -> "BaseRetriever":
    """Return the backend for `method` (None = the server default)."""
    return RETRIEVERS[method or DEFAULT_METHOD]


app = FastAPI()  # FastAPI server

# Dynamic batching enabled when RETRIEVER_BATCH_WINDOW_S > 0. RPL issues one search() per worker,
# so ~SANDBOX_WORKERS single-query /retrieve requests arrive concurrently; coalescing them turns
# N batch-1 encode+faiss calls into one batch-N call. Off when the window is 0.
_BATCH_WINDOW = float(os.environ.get("RETRIEVER_BATCH_WINDOW_S", "0"))
_BATCH_MAX = int(os.environ.get("RETRIEVER_BATCH_MAX", "256"))
_req_q: "_queuelib.Queue" = _queuelib.Queue()


def _batcher_loop():
    """Pull concurrent requests, coalesce those with the same (topk, return_scores) into one
    batch_search, and split the results back to each caller by its query count."""
    while True:
        batch = [_req_q.get()]                     # block for the first request
        deadline = time.perf_counter() + _BATCH_WINDOW
        while len(batch) < _BATCH_MAX:                            # gather more within the window
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(_req_q.get(timeout=remaining))
            except _queuelib.Empty:
                break
        groups = defaultdict(list)  # batch_search takes a single (method, topk)
        for it in batch:
            groups[(it["method"], it["topk"], it["return_scores"])].append(it)
        for (method, topk, return_scores), items in groups.items():
            queries = [q for it in items for q in it["queries"]]
            try:
                out = _get_retriever(method).batch_search(query_list=queries, num=topk, return_score=return_scores)
                results, scores = out if return_scores else (out, None)
            except Exception as exc:                     # surface the error to every caller in the group
                for it in items:
                    it["error"] = exc
                    it["event"].set()
                continue
            offset = 0
            for it in items:
                n = len(it["queries"])
                it["results"] = results[offset:offset + n]
                it["scores"] = scores[offset:offset + n] if return_scores else None
                offset += n
                it["event"].set()


def _batched_retrieve(queries, topk, return_scores, method):
    item = {
        "queries": queries,
        "topk": topk,
        "return_scores": return_scores,
        "method": method,
        "event": threading.Event(),
    }
    _req_q.put(item)
    item["event"].wait()
    if "error" in item:
        raise item["error"]
    return item["results"], item["scores"]


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default
    method = request.method or DEFAULT_METHOD
    if method not in RETRIEVERS:
        # Answering from the default backend would make sparse_retrieval() silently dense on a server started
        # without --multi.
        raise HTTPException(status_code=400, detail=f"method {method!r} is not served (serving "
                            f"{sorted(RETRIEVERS)}); start the retriever with --multi for both")

    # Perform retrieval (coalesced with concurrent requests when dynamic batching is on).
    if _BATCH_WINDOW > 0:
        results, scores = _batched_retrieve(request.queries, request.topk, request.return_scores, method)
    else:
        out = _get_retriever(method).batch_search(
            query_list=request.queries, num=request.topk, return_score=request.return_scores)
        results, scores = out if request.return_scores else (out, None)

    # Format response.
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--index_path", type=str, default="e5_Flat.index", help="Corpus indexing file.")
    parser.add_argument("--corpus_path", type=str, default="wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    # Must match what index_builder.py used, or queries are pooled differently from the passages they are
    # scored against -- which degrades retrieval silently rather than erroring. Default None = derive from
    # the retriever name via the SAME table the builder uses, so the two cannot drift apart.
    parser.add_argument("--retrieval_pooling_method", type=str, default=None,
                        choices=[None, "mean", "cls", "pooler"], help="Query pooling; default = per-model.")
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")
    # Multi-retriever mode: also load a sparse (bm25) backend so /retrieve can route on the `method` field.
    parser.add_argument("--multi", action="store_true", help="Serve BOTH dense + sparse, routed by the request `method`.")
    parser.add_argument("--bm25_index_path", type=str, default=None, help="Sparse (bm25/Lucene) index dir; required with --multi.")

    args = parser.parse_args()
    pooling_method = args.retrieval_pooling_method or next(
        (v for k, v in MODEL2POOLING.items() if k in args.retriever_name.lower()), "mean")

    # Dense config (also the single-retriever config when --multi is off).
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method=pooling_method,
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    # Build the method -> backend registry. --multi loads BOTH (dense CUDA + sparse JVM coexist in one
    # process); otherwise just the single configured retriever.
    if args.multi:
        if not args.bm25_index_path:
            parser.error("--multi requires --bm25_index_path")
        bm25_config = Config(retrieval_method="bm25", index_path=args.bm25_index_path,
                             corpus_path=args.corpus_path, retrieval_topk=args.topk)
        RETRIEVERS["dense"] = DenseRetriever(config)
        RETRIEVERS["sparse"] = BM25Retriever(bm25_config)
        DEFAULT_METHOD = "dense"
        print(f"[retriever] MULTI mode: dense (faiss{'-gpu' if args.faiss_gpu else ''}) + sparse (bm25), routed by `method`", flush=True)
    else:
        method_name = "sparse" if args.retriever_name == "bm25" else "dense"
        RETRIEVERS[method_name] = get_retriever(config)
        DEFAULT_METHOD = method_name

    # 2b) Start the dynamic-batching thread if enabled (RETRIEVER_BATCH_WINDOW_S > 0).
    if _BATCH_WINDOW > 0:
        threading.Thread(target=_batcher_loop, daemon=True).start()
        print(f"[retriever] dynamic batching ON (window={_BATCH_WINDOW}s, max={_BATCH_MAX})", flush=True)

    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host=os.environ.get("SERVER_BIND_HOST", "0.0.0.0"), port=8000)  # :: for a cross-host serve job
