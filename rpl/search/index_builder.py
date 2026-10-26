# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adapted from Search-R1 (https://github.com/PeterGriffinJin/Search-R1), licensed under the
# Apache License, Version 2.0; see LICENSE-APACHE in the root directory of this source tree.

"""Build the indexes the retrieval server loads."""

import argparse
import os
import shutil
import subprocess
import warnings
from typing import cast

import datasets
import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_model(model_path: str, use_fp16: bool = False):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        "json",
        data_files=corpus_path,
        split="train",
        num_proc=4,
    )
    return corpus


class IndexBuilder:
    """A tool class used to build an index used in retrieval."""

    def __init__(
        self,
        retrieval_method,
        model_path,
        corpus_path,
        save_dir,
        max_length,
        batch_size,
        use_fp16,
        pooling_method,
        faiss_type=None,
        embedding_path=None,
        save_embedding=False,
        faiss_gpu=False,
    ):
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else "Flat"
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu

        self.gpu_num = torch.cuda.device_count()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")
        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")
        self.corpus = load_corpus(self.corpus_path)
        print("Finish loading...")

    @staticmethod
    def _check_dir(dir_path):
        """Return False if `dir_path` already holds files; otherwise make sure it exists and return True."""
        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        """Build a BM25 index for `bm25`, otherwise a dense FAISS index."""
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

    def build_bm25_index(self):
        """Build a BM25 index with Pyserini.

        See https://github.com/castorini/pyserini/blob/master/docs/usage-index.md
        """
        # Pyserini's JsonCollection indexes a directory, so stage the corpus file in one.
        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir)
        shutil.copyfile(self.corpus_path, temp_file_path)

        print("Start building bm25 index...")
        pyserini_args = ["--collection", "JsonCollection", "--input", temp_dir, "--index", self.save_dir]
        pyserini_args += ["--generator", "DefaultLuceneDocumentGenerator", "--threads", "1"]

        # Pyserini builds an `openai.OpenAI()` client at IMPORT time and raises without a key, so this
        # subprocess dies on an OpenAI credentials error before it indexes anything. Nothing here calls
        # OpenAI; a placeholder is enough, and a real key in the environment is left alone.
        env = {**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "unused-by-rpl")}
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args, env=env, check=True)

        shutil.rmtree(temp_dir)

        print("Finish!")

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        all_embeddings = np.memmap(embedding_path, mode="r", dtype=np.float32).reshape(corpus_size, hidden_size)
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(self.embedding_save_path, shape=all_embeddings.shape, mode="w+", dtype=all_embeddings.dtype)
        length = all_embeddings.shape[0]
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i:j] = all_embeddings[i:j]
        else:
            memmap[:] = all_embeddings

    def encode_all(self):
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.encoder = torch.nn.DataParallel(self.encoder)
            self.batch_size = self.batch_size * self.gpu_num

        all_embeddings = []

        for start_idx in tqdm(range(0, len(self.corpus), self.batch_size), desc="Inference Embeddings:"):
            batch_data = self.corpus[start_idx : start_idx + self.batch_size]["contents"]

            if self.retrieval_method == "e5":
                batch_data = [f"passage: {doc}" for doc in batch_data]

            inputs = self.tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to("cuda")

            if "T5" in type(self.encoder).__name__:
                decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                    inputs["input_ids"].device
                )
                output = self.encoder(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
                embeddings = output.last_hidden_state[:, 0, :]

            else:
                output = self.encoder(**inputs, return_dict=True)
                embeddings = pooling(
                    output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
                )
                if "dpr" not in self.retrieval_method:
                    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

            embeddings = cast(torch.Tensor, embeddings)
            embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings

    @torch.no_grad()
    def build_dense_index(self):
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")

        self.encoder, self.tokenizer = load_model(model_path=self.model_path, use_fp16=self.use_fp16)
        if self.embedding_path is not None:
            hidden_size = self.encoder.config.hidden_size
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            all_embeddings = self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus

        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)

        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)

        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")


MODEL2POOLING = {"e5": "mean", "bge": "cls", "contriever": "mean", "jina": "mean"}


def main():
    parser = argparse.ArgumentParser(description="Creating index.")

    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--save_dir", default="indexes/", type=str)

    # Dense-index parameters (ignored for bm25)
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--faiss_type", default=None, type=str)
    parser.add_argument("--embedding_path", default=None, type=str)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")

    args = parser.parse_args()

    if args.pooling_method is None:
        pooling_method = "mean"
        for k, v in MODEL2POOLING.items():
            if k in args.retrieval_method.lower():
                pooling_method = v
                break
    elif args.pooling_method not in ["mean", "cls", "pooler"]:
        raise NotImplementedError(f"--pooling_method must be mean, cls or pooler, got {args.pooling_method!r}")
    else:
        pooling_method = args.pooling_method

    index_builder = IndexBuilder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=pooling_method,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
