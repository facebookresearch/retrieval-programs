# RPL — Retrieval Program Learning

RPL trains a language model to write the retrieval procedure rather than to call a fixed one. Instead of
emitting a search query, the policy writes Python inside a sandbox to access external information:

```python
candidates = fuse(dense_retrieval("who wrote the novel Beloved", top_k=20),
                  sparse_retrieval("Beloved novel author", top_k=20))
for i, doc in enumerate(rerank("who wrote the novel Beloved", candidates, top_k=3)):
    print(f"Doc {i+1}(Title: {doc['title']}) {doc['text']}")
```

---

## Install

```bash
# 1. PyTorch matching your CUDA, then flash-attn for your GPU architecture.
#    (Both are platform-specific, which is why they are not in requirements.txt.)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation

# 2. Training, the agent loop and the reranker service.
pip install -r requirements.txt

# 3. The retriever service. Separate because faiss and pyserini pull in heavy pins that can conflict with
#    the training stack — and the retriever is a separate process reached over HTTP, so it is fine (often
#    easier) to give it its own environment. One environment also works if the resolver is happy.
pip install -r requirements-retrieval.txt

# 4. A JDK for the sparse (BM25) retriever. Pyserini runs Lucene; we ship no Java.
#    JDK 21, e.g. Eclipse Temurin. Set JAVA_HOME and put java on PATH.
java -version
```

Every script below expects the training environment to be active (`python` on PATH is the one with the
requirements installed). The launch scripts first write `rpl_repo.pth`, holding the repo path, into that
environment's site-packages, so every Python process in it, Ray workers included, can import `rpl` and the
vendored `verl` without installing the repo. `scripts/serve_retriever.sh` is the one script to run with the
retrieval environment active instead.

## Corpus and indices

RPL uses the 2018 Wikipedia snapshot (wiki-18) from [DPR](https://github.com/facebookresearch/DPR). Download the
corpus and the prebuilt indices Search-R1 published:

```bash
scripts/download_index.sh --out indices --what corpus,bm25,flat
```

To build the indices yourself instead — hours of GPU encoding for wiki-18 — run
`rpl/search/build_index.sh --corpus indices/wiki-18.jsonl --out indices`. GlobalQA has its own corpus and index;
see [GlobalQA](#globalqa).

## Data

Build one dataset per method, each with its prompt template from `scripts/data_process/prompts.py`:

```bash
scripts/prepare_data.sh --template base       --out data/nq_hotpotqa_base         # Search-R1
scripts/prepare_data.sh --template autorefine --out data/nq_hotpotqa_autorefine   # AutoRefine
scripts/prepare_data.sh --template rpl-openqa --out data/nq_hotpotqa_rpl          # RPL
```

---

## Running

### Start the services first

The retriever and the reranker are separate processes, and both fail soft if absent — an unreachable
retriever makes `dense_retrieval()` raise, the policy falls back to answering from memory, and the run
completes with plausible-looking but much lower accuracy. The launch scripts refuse to start without them.

```bash
# Config A (8 training GPUs, no reranker): let FAISS shard the index across all of them.
scripts/serve_retriever.sh --index indices/e5_Flat.index --corpus indices/wiki-18.jsonl \
                           --bm25-index indices/bm25 --multi --faiss-gpu

# Config B (4 training GPUs + 4 reranker GPUs): pin the retriever to the training half, so it
# does not collide with the reranker, and lower vLLM's share to leave room for the index.
scripts/serve_retriever.sh --index indices/e5_Flat.index --corpus indices/wiki-18.jsonl \
                           --bm25-index indices/bm25 --multi --faiss-gpu --gpu-devices 0,1,2,3
scripts/serve_reranker.sh  --gpu-devices 4,5,6,7
scripts/train_rpl.sh --data data/nq_hotpotqa_rpl --rerank-gpus 4 --gpu-mem-util 0.55 --out runs/rpl
```

`--faiss-gpu` shards the index across every visible GPU, so what matters is `index_size / n_visible_gpus`
plus whatever else is resident on those cards; the flat index does not fit on a single GPU and OOMs at load.
If you are short of memory, switch to the approximate HNSW index, which needs less memory to serve
(`download_index.sh --what hnsw`), drop `--faiss-gpu` for CPU FAISS, or give the retriever the reranker's half
with `--gpu-devices`.

### GlobalQA

GlobalQA ([`QiiLuoo/GlobalQA`](https://huggingface.co/datasets/QiiLuoo/GlobalQA)) brings its own corpus,
`corpus.jsonl`, and the reported runs retrieve from it with `BAAI/bge-base-en-v1.5` plus BM25. Build both
indices and serve them in place of the wiki-18 retriever.

```bash
rpl/search/build_index.sh --corpus globalqa/corpus.jsonl --out indices_globalqa --retriever bge
scripts/serve_retriever.sh --index indices_globalqa/bge_Flat.index --corpus globalqa/corpus.jsonl \
    --bm25-index indices_globalqa/bm25 --multi --retriever bge --retriever-model BAAI/bge-base-en-v1.5
```

The data is built per template by `scripts/data_process/globalqa_search.py` from the download's
`train (1).json` and `test (3).json`, one directory per method: `rpl-globalqa` for RPL, `base-globalqa` for
Search-R1, and `marag` for MARAG-R1. Always pass `--exclude_test_questions`: 1,073 of the 1,160 test questions
also appear verbatim in the official train split.

```bash
for t in rpl-globalqa base-globalqa marag; do
  python scripts/data_process/globalqa_search.py --src globalqa \
      --template_type "$t" --local_dir "data/globalqa_${t%-globalqa}" --exclude_test_questions
done

scripts/train_globalqa.sh     --data data/globalqa_rpl   --rerank-gpus 4 --out runs/rpl_globalqa
scripts/train_sat_globalqa.sh --data data/globalqa_base                  --out runs/sat_globalqa
scripts/train_marag.sh        --data data/globalqa_marag --rerank-gpus 4 --out runs/marag
```

### Evaluation

The reported wiki-18 numbers come from the checkpoint with the best unweighted mean EM over the run's own
held-out split (nq and hotpotqa test), among steps 50..1000; ties go to the later step. That checkpoint is
then evaluated on the seven benchmarks, and the reported average is their unweighted mean:

```bash
python tools/qa_report.py select --run runs/rpl --parquet data/nq_hotpotqa_rpl/test.parquet

for d in nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle; do
  scripts/prepare_data.sh --template rpl-openqa --datasets "$d" --splits test --out "data/eval7_rpl/$d"
done
VAL=$(printf 'data/eval7_rpl/%s/test.parquet:' nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle)
scripts/eval.sh --data data/eval7_rpl/nq --val "${VAL%:}" --mode code --rerank-gpus 4 \
                --ckpt runs/rpl/global_step_<selected>/actor/huggingface --out runs/rpl_eval7
python tools/qa_report.py score --dump runs/rpl_eval7/val_rollouts/0.jsonl --parquet ${VAL//:/ }
```

Direct, CoT and RAG need `--single-pass`: without it a model that does not answer cleanly on the first try is
re-prompted with a search nudge and scored on the retry. The reported CoT, RAG, IRCoT and Search-o1 rows ran
with thinking on; Direct ran with it off. Search-o1 is IRCoT with its Reason-in-Documents step switched on:
after each search the same model condenses the passages into a short summary (at most `condense_max_tokens`,
256), and only the summary enters the context.

```bash
for t in direct cot ircot; do
  scripts/prepare_data.sh --template "$t" --datasets nq --splits test --out "data/nq_$t"
done
python scripts/data_process/qa_rag.py --preset openqa --src data/nq_direct --out data/nq_rag   # retriever up

scripts/eval.sh --data data/nq_direct --mode search --single-pass --model Qwen/Qwen3-4B
scripts/eval.sh --data data/nq_cot    --mode search --single-pass --enable-thinking 1 --model Qwen/Qwen3-4B
scripts/eval.sh --data data/nq_rag    --mode search --single-pass --enable-thinking 1 --model Qwen/Qwen3-4B
scripts/eval.sh --data data/nq_ircot  --mode search --enable-thinking 1 --model Qwen/Qwen3-4B

# Search-o1: the IRCoT data with document condensation on
scripts/eval.sh --data data/nq_ircot  --mode search --enable-thinking 1 --model Qwen/Qwen3-4B -- +rpl.condense_docs=True
```

### Program mutation

The overall pipeline: for rollouts whose program failed to retrieve gold, harvest a nearby program that does
retrieve it, measure against a resample control how much answering from the repaired evidence helps, fine-tune
the policy on the repaired programs, and continue RL from the fine-tuned policy.

```bash
# The paper's rounds harvest steps 151-199 of a run (every 2nd dump, the first 512 rows of each) and fine-tune
# its step-200 checkpoint.
scripts/run_mutation_round.sh --rollouts runs/rpl/rollouts --from-step 151 --to-step 200 \
                              --data data/nq_hotpotqa_rpl --out round1
```

The driver stages the verify and control sets, then prints the remaining commands with the reported settings:

1. Both sets are answered greedily, in one pass, by the step-200 policy (`eval.sh --single-pass`). The
   re-executed program's observation is cut to `--max-obs-length` tokens, as live observations are in RL.
   Score each set against its own parquet with `tools/qa_report.py score`: verify minus control EM is what the
   repairs themselves earn.
2. The SFT records are every verify row: the repaired program is supervised on all of them, the answer only
   where it was correct. `sft_mutation.sh --model <step-200 export>` trains at lr 1e-6 for one epoch.
3. `train_rpl.sh --init-from <SFT export> --steps 800` forks RL from the fine-tuned weights, with a fresh
   optimizer and no LR warmup.

Every run writes its generations alongside its checkpoints — `<out>/rollouts/<step>.jsonl` for training and
`<out>/val_rollouts/<step>.jsonl` for whatever set it evaluates. Program mutation harvests only from
`rollouts/`. The `val_rollouts/` it reads are evals of a round's verify and control sets, which are built from
those training rollouts. No held-out question becomes SFT data: `mutation_verify.py` refuses a `val_rollouts/`
dump, and `mutation_sft.py` aborts on any rollout that does not belong to the verify set.

---

## Repository layout

```
rpl/
  llm_agent/
    rpl_agent_loop.py       the turn loop: agentic search interfaces, observation handling
    rpl_reward_manager.py   EM, F1, coverage, efficiency and format rewards
    code_executor.py        sandbox parent: spawn, timeout, loopback-only networking
    sandbox_runner.py       sandbox child: a fresh namespace per <code> block
    sft_mask.py             loss masking for program-mutation SFT
  search/
    search_sdk.py           the retrieval SDK the policy programs against
    retrieval_server.py     dense (FAISS) + sparse (BM25) retrieval service
    rerank_server.py        cross-encoder service
    index_builder.py        index construction
scripts/
  lib/                      shared argument parsing and the single verl invocation
  download_index.sh         fetch the wiki-18 corpus and a prebuilt index
  data_process/prompts.py   prompt templates
  program_mutation/         harvest -> verify -> SFT records
tools/
  qa_report.py              checkpoint selection and the seven-benchmark average
  globalqa_report.py        GlobalQA metrics per task type
verl/                       verl v0.8.0
```

## License

The majority of Retrieval Program Learning is licensed under CC-BY-NC, however portions of the project are
available under separate license terms: Search-R1 and verl are licensed under the Apache 2.0 license.

This repository is derived from, and vendors code from:

* [Search-R1](https://github.com/PeterGriffinJin/Search-R1) (Apache License 2.0), vendored at commit
  `598e61bd1d36895726d28a8d06b3a15bed19f5d3`.
* [verl](https://github.com/volcengine/verl) (Apache License 2.0), vendored at tag `v0.8.0`.
  Copyright 2023-2024 Bytedance Ltd. and/or its affiliates

`LICENSE` is the CC BY-NC 4.0 license. `LICENSE-APACHE` is the Apache License 2.0, which governs the vendored
Search-R1 and verl code.

## Citation

```bibtex
@article{kim2026learning,
  title={Learning to Program Retrievals: Composing Search Primitives via Reinforcement Learning},
  author={Kim, Kyuyoung and Xiao, Zilin and Wang, Zhiyuan and Atreya, Avinash and Yang, Ke and Shin, Jinwoo and Hao, Kecheng},
  journal={arXiv preprint arXiv:2610.24854},
  year={2026}
}
```
