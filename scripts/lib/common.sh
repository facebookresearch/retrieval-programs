#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Shared argument parsing, defaults and preflight checks for the RPL launch scripts.

set -euo pipefail

# ---------------
# Defaults.
# ---------------

function rpl_defaults {
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

  # Topology. TRAIN_GPUS is derived: GPUS - RERANK_GPUS.
  # `|| true` inside the substitution matters: under `set -o pipefail` a missing nvidia-smi makes the whole
  # pipeline return 127, which `set -e` turns into a silent abort before anything has printed.
  GPUS="${GPUS:-$( (nvidia-smi -L 2>/dev/null || true) | wc -l )}"
  RERANK_GPUS=0

  MODEL="Qwen/Qwen3-4B"
  INIT_FROM=""             # set by --init-from: a fork from a fine-tuned checkpoint
  DATA_DIR=""              # holds train.parquet and test.parquet
  VAL_FILES=""             # colon-separated list; defaults to $DATA_DIR/test.parquet
  OUT_DIR="./outputs"
  STEPS=1000
  PROJECT="rpl"
  EXP_NAME=""

  RETRIEVER_URL="http://127.0.0.1:8000/retrieve"
  RERANK_URL="http://127.0.0.1:6980/rerank"

  # Optimization.
  ADV_ESTIMATOR=grpo
  TRAIN_BS=512
  PPO_MINI=256
  N_ROLL=5                 # GRPO group size
  LR=1e-6
  # Empty resolves to 150, or to 0 after --init-from: warmup counts absolute steps, so a fork would re-warm.
  LR_WARMUP_STEPS=""
  KL_COEF=0.001
  MAX_PROMPT_LEN=1024
  # The whole multi-turn response, observations included. A rollout cut at this cap loses its answer and
  # scores 0, so it must hold the longest trajectories; every reported Qwen3 run used 15360.
  MAX_RESPONSE_LEN=15360
  PPO_MAX_TOKEN_LEN=""     # empty = MAX_PROMPT_LEN + MAX_RESPONSE_LEN

  # Rollout.
  TP=2                     # vLLM tensor-parallel size; raise for larger models
  GPU_MEM_UTIL=0.7         # vLLM's share. The retriever's GPU-FAISS index is co-resident on a
                           # single-node setup, so this cannot be pushed near 1.0.

  # Park the actor's FSDP params and Adam state on CPU while vLLM owns the GPU for rollout.
  # ON by default. REQUIRED to train 8B on a single node, and for long-context arms (GlobalQA TopK=20)
  # which OOM at 4B too.
  ACTOR_OFFLOAD=1
  MAX_TURNS=3
  MAX_OBS_LENGTH=2048
  TOPK=3
  # >0 truncates each retrieved document to that many characters in text observations, instead of letting
  # MAX_OBS_LENGTH cut the concatenation and drop whole documents.
  DOC_CHARS=0
  # >0 caps how many documents one rerank() call scores.
  RERANK_MAX_DOCS=0

  # Instruct models take the chat template (`apply`); the `base` ones take a raw prompt (`raw`).
  CHAT_TEMPLATE=apply
  ENABLE_THINKING=0
  MAX_GEN_TOKENS=2048
  ADD_BOS=1

  # Sandbox (RPL arms).
  SANDBOX_TIMEOUT_SEC=600
  PERSIST_SANDBOX_STATE=1
  MAX_STATE_BYTES=$((4 * 1024 * 1024))

  # Reward.
  REWARD_MODE=em
  MIN_SEARCHES=0
  DOC_REWARD=0             # GlobalQA only: weight of document-coverage F1 (MARAG-R1's R_E term)

  # How many validation generations to echo into the log (the full set still goes to /val_rollouts).
  LOG_VAL_GENERATIONS=10

  # Checkpointing / logging.
  SEED=""
  SAVE_FREQ=50
  TEST_FREQ=50
  MAX_CKPT_KEEP=1
  LOGGER='["console"]'

  DRY_RUN=0
  EVAL_ONLY=0
  EXTRA_ARGS=()
}

# Settings every GlobalQA arm shares, so the methods train on one protocol. Call after rpl_defaults and
# before rpl_parse_args, so the command line still wins.
function rpl_globalqa_defaults {
  DOC_REWARD=0.5          # MARAG-R1's coverage term at the weight every reported GlobalQA arm used
  DOC_CHARS=600           # so twenty documents fit one observation instead of the first few
  RERANK_MAX_DOCS=200
  STEPS=500               # the GlobalQA table reports step 500
}

function rpl_usage {
  cat <<EOF
Usage: $(basename "$0") [options]

  --gpus N              GPUs on this node (default: all visible)
  --rerank-gpus N       GPUs reserved for an on-node reranker (default: ${RERANK_GPUS})
  --model PATH          HF id or local path (default: ${MODEL})
  --init-from PATH      fork from a fine-tuned HF checkpoint (the program-mutation SFT): --model PATH with
                        LR warmup 0 unless --lr-warmup-steps is given
  --data DIR            directory holding train.parquet / test.parquet  [required]
  --val FILES           colon-separated eval parquets (default: DIR/test.parquet)
  --out DIR             checkpoint + output dir (default: ${OUT_DIR})
  --steps N             total training steps (default: ${STEPS})
  --lr-warmup-steps N   absolute warmup steps (default: 150; 0 with --init-from)
  --max-response-len N  whole multi-turn response, observations included (default: ${MAX_RESPONSE_LEN})
  --max-obs-length N    tokens per observation (default: ${MAX_OBS_LENGTH})
  --project NAME        project name for logging (default: ${PROJECT})
  --name NAME           experiment name (default: derived from the script name)
  --retriever-url URL   (default: ${RETRIEVER_URL})
  --rerank-url URL      (default: ${RERANK_URL})
  --tp N                vLLM tensor-parallel size (default: ${TP})
  --gpu-mem-util F      vLLM GPU memory fraction (default: ${GPU_MEM_UTIL})
  --actor-offload 0|1   park actor params+optimizer on CPU during rollout; required at 8B (default: ${ACTOR_OFFLOAD})
  --max-turns N         turn budget per trajectory (default: ${MAX_TURNS})
  --topk N              documents per retrieval call (default: ${TOPK})
  --seed N              data-order seed for replicates (default: verl's order)
  --doc-reward F        GlobalQA document-coverage reward weight (default: ${DOC_REWARD})
  --reward-mode MODE    em | conditional | ... (default: ${REWARD_MODE})
  --chat-template M     apply | raw  -- raw for base models (default: ${CHAT_TEMPLATE})
  --enable-thinking 0|1 only meaningful under apply (default: ${ENABLE_THINKING})
  --max-gen-tokens N    PER TURN; keep >=2048 if thinking is on (default: ${MAX_GEN_TOKENS})
  --dry-run             print the resolved command and exit
  -h, --help            this message

Anything after -- is appended verbatim to the Hydra command line.
EOF
}

function rpl_parse_args {
  while [ $# -gt 0 ]; do
    case "$1" in
      --gpus)          GPUS="$2"; shift 2 ;;
      --rerank-gpus)   RERANK_GPUS="$2"; shift 2 ;;
      --model)         MODEL="$2"; shift 2 ;;
      --init-from)     MODEL="$2"; INIT_FROM="$2"; shift 2 ;;
      --data)          DATA_DIR="$2"; shift 2 ;;
      --val)           VAL_FILES="$2"; shift 2 ;;
      --out)           OUT_DIR="$2"; shift 2 ;;
      --steps)         STEPS="$2"; shift 2 ;;
      --lr-warmup-steps)  LR_WARMUP_STEPS="$2"; shift 2 ;;
      --max-response-len) MAX_RESPONSE_LEN="$2"; shift 2 ;;
      --max-obs-length)   MAX_OBS_LENGTH="$2"; shift 2 ;;
      --project)       PROJECT="$2"; shift 2 ;;
      --name)          EXP_NAME="$2"; shift 2 ;;
      --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
      --rerank-url)    RERANK_URL="$2"; shift 2 ;;
      --tp)            TP="$2"; shift 2 ;;
      --gpu-mem-util)  GPU_MEM_UTIL="$2"; shift 2 ;;
      --actor-offload) ACTOR_OFFLOAD="$2"; shift 2 ;;
      --max-turns)     MAX_TURNS="$2"; shift 2 ;;
      --topk)          TOPK="$2"; shift 2 ;;
      --seed)          SEED="$2"; shift 2 ;;
      --doc-reward)    DOC_REWARD="$2"; shift 2 ;;
      --reward-mode)   REWARD_MODE="$2"; shift 2 ;;
      --chat-template) CHAT_TEMPLATE="$2"; shift 2 ;;
      --enable-thinking) ENABLE_THINKING="$2"; shift 2 ;;
      --max-gen-tokens)  MAX_GEN_TOKENS="$2"; shift 2 ;;
      --dry-run)       DRY_RUN=1; shift ;;
      -h|--help)       rpl_usage; exit 0 ;;
      --)              shift; EXTRA_ARGS=("$@"); break ;;
      *)               echo "ERROR: unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
  done
  [ -n "$LR_WARMUP_STEPS" ] || LR_WARMUP_STEPS=$([ -n "$INIT_FROM" ] && echo 0 || echo 150)
}

# ---------------
# Preflight.
# ---------------

function rpl_resolve_topology {
  [ "${GPUS:-0}" -ge 1 ] || { echo "ERROR: no GPUs detected; pass --gpus N" >&2; exit 1; }
  TRAIN_GPUS=$((GPUS - RERANK_GPUS))
  if [ "$TRAIN_GPUS" -lt 1 ]; then
    echo "ERROR: --rerank-gpus $RERANK_GPUS leaves $TRAIN_GPUS GPUs for training (of $GPUS)." >&2
    exit 1
  fi
  if [ $((TRAIN_GPUS % TP)) -ne 0 ]; then
    echo "ERROR: --tp $TP does not divide the $TRAIN_GPUS training GPUs." >&2
    exit 1
  fi
  # The reranker takes the LAST N devices so it never contends with the vLLM rollout on 0..TRAIN_GPUS-1.
  TRAIN_DEVS="$(seq -s, 0 $((TRAIN_GPUS - 1)))"
  RERANK_DEVS=""
  if [ "$RERANK_GPUS" -gt 0 ]; then RERANK_DEVS="$(seq -s, "$TRAIN_GPUS" $((GPUS - 1)))"; fi
  # balance_batch reshuffles across the full node; with GPUs carved out it would assume ranks that do not exist.
  BALANCE_BATCH=True
  if [ "$RERANK_GPUS" -gt 0 ]; then BALANCE_BATCH=False; fi
}

# Confirm the GPUs are actually reachable from Python before handing off to verl.
function rpl_require_visible_gpus {
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  local seen
  seen="$(CUDA_VISIBLE_DEVICES="$TRAIN_DEVS" python - <<'PY' 2>/dev/null || echo "error"
import torch
print(torch.cuda.device_count())
PY
)"
  if [ "$seen" != "$TRAIN_GPUS" ]; then
    echo "ERROR: asked for $TRAIN_GPUS training GPUs (devices $TRAIN_DEVS) but torch reports '$seen'." >&2
    echo "       Check CUDA_VISIBLE_DEVICES and that the CUDA driver is visible to this Python." >&2
    exit 1
  fi
  local ray_seen
  ray_seen="$(python - <<'PY' 2>/dev/null || echo "error"
from ray._private.accelerators import NvidiaGPUAcceleratorManager as M
print(M.get_current_node_num_accelerators())
PY
)"
  if [ "$ray_seen" = "0" ] || [ "$ray_seen" = "error" ]; then
    echo "ERROR: torch sees $seen GPU(s) but Ray's detector reports '$ray_seen', so verl will fail to" >&2
    echo "       place its workers. Ray counts GPUs through NVML -- make libnvidia-ml.so loadable" >&2
    echo "       (it is a separate library from libcuda.so, which is why torch works and Ray does not)." >&2
    exit 1
  fi
  echo "  gpus OK       torch:$seen  ray:$ray_seen"
}

function rpl_require_data {
  [ -n "$DATA_DIR" ] || { echo "ERROR: --data DIR is required (see scripts/prepare_data.sh)" >&2; exit 1; }
  if [ "$EVAL_ONLY" = "0" ] && [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: $DATA_DIR/train.parquet not found." >&2
    echo "       Build it with scripts/prepare_data.sh --template <name>." >&2
    exit 1
  fi
  [ -n "$VAL_FILES" ] || VAL_FILES="$DATA_DIR/test.parquet"
  local f
  for f in ${VAL_FILES//:/ }; do
    [ -f "$f" ] || { echo "ERROR: eval parquet not found: $f" >&2; exit 1; }
  done
  # verl loads train_files even under val_only, though it never samples them, and an eval-only set (a RAG
  # baseline, PopQA, Bamboogle) has no train split -- so point it at the first eval parquet instead.
  TRAIN_FILES="$DATA_DIR/train.parquet"
  if [ "$EVAL_ONLY" = "1" ] && [ ! -f "$TRAIN_FILES" ]; then TRAIN_FILES="${VAL_FILES%%:*}"; fi
}

# POST a tiny probe per backend the arm calls: `rpl_require_retriever dense sparse` for the SDK arms, no
# argument for the server's default backend.
function rpl_require_retriever {
  # A dry run prints a command; it should not require the services that command will need.
  if [ "$DRY_RUN" = "1" ]; then echo "  (dry run: skipped the retriever check)"; return 0; fi
  local method probe
  for method in "${@:-}"; do
    probe="{\"queries\":[\"who wrote hamlet\"],\"topk\":1,\"return_scores\":false${method:+,\"method\":\"$method\"}}"
    curl -sf --noproxy '*' "$RETRIEVER_URL" -X POST -H 'Content-Type: application/json' \
      -d "$probe" >/dev/null 2>&1 && continue
    if [ -n "$method" ] && curl -sf --noproxy '*' "$RETRIEVER_URL" -X POST -H 'Content-Type: application/json' \
         -d '{"queries":["who wrote hamlet"],"topk":1}' >/dev/null 2>&1; then
      echo "ERROR: the retriever at $RETRIEVER_URL does not serve '$method', which this arm calls." >&2
      echo "       Serve both backends:  scripts/serve_retriever.sh ... --bm25-index <dir> --multi" >&2
    else
      echo "ERROR: no retriever answering at $RETRIEVER_URL" >&2
      echo "       Start one first:  scripts/serve_retriever.sh --index <dir> --corpus <corpus.jsonl>" >&2
      echo "       Training without it does NOT fail loudly -- the policy silently answers from memory." >&2
    fi
    exit 1
  done
  echo "  retriever OK  $RETRIEVER_URL${1:+ ($*)}"
}

function rpl_require_reranker {
  # A dry run prints a command; it should not require the services that command will need.
  if [ "$DRY_RUN" = "1" ]; then echo "  (dry run: skipped the reranker check)"; return 0; fi
  local probe='{"queries":["who wrote hamlet"],"documents":[[{"id":0,"contents":"Hamlet\nby William Shakespeare"}]],"return_scores":true}'
  if ! curl -sf --noproxy '*' "$RERANK_URL" -X POST -H 'Content-Type: application/json' \
       -d "$probe" >/dev/null 2>&1; then
    echo "ERROR: no reranker answering at $RERANK_URL, but this arm calls rerank()." >&2
    echo "       Start one first:  scripts/serve_reranker.sh --gpus N" >&2
    echo "       Without it the policy's rerank() calls raise, RL prunes them, and the arm reads as" >&2
    echo "       'the model declined to rerank' rather than as a broken configuration." >&2
    exit 1
  fi
  echo "  reranker OK   $RERANK_URL"
}

# Make the repo importable inside Ray actors.
function rpl_install_pth {
  command -v python >/dev/null 2>&1 || {
    echo "ERROR: no \`python\` on PATH. Activate the training environment first (see README: Install)." >&2
    exit 1
  }
  local site
  site="$(python -c 'import site; print(site.getsitepackages()[0])')"
  echo "$REPO_ROOT" > "$site/rpl_repo.pth"
}

function rpl_banner {
  echo "== $1 =="
  echo "  model        $MODEL$([ -n "$INIT_FROM" ] && echo "   (fork: weights only, fresh optimizer)")"
  echo "  data         $DATA_DIR"
  echo "  eval         $VAL_FILES"
  echo "  gpus         $GPUS total -> $TRAIN_GPUS train (devices $TRAIN_DEVS)$([ -n "$RERANK_DEVS" ] && echo ", $RERANK_GPUS rerank (devices $RERANK_DEVS)")"
  echo "  steps        $STEPS   (save/test every $SAVE_FREQ/$TEST_FREQ, LR warmup $LR_WARMUP_STEPS)"
  echo "  out          $OUT_DIR"
}
