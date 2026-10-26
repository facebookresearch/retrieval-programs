#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Supervised fine-tuning on program-mutation repairs.
# Consumes the record file that scripts/run_mutation_round.sh produces.
#
# The loss mask is the part that fails silently: a misaligned mask trains on the wrong tokens and still
# produces a plausible loss curve. `--supervise` picks which spans carry gradient (see rpl/llm_agent/
# sft_mask.py); `program` is the default and supervises only the repaired retrieval program.
#
# `--prompt-format` decides how each prompt is tokenized. By default it follows `--chat-template`, so the SFT
# prompt is token for token the prompt an RL rollout sees: `apply` wraps the question in the chat template,
# `raw` encodes it bare. The reported PM rows used `raw` under `--chat-template apply`; pass
# `--prompt-format raw` to reproduce them.
#
# `--model` is required: the RL checkpoint the round harvested from, e.g. the step-200 export the reported rounds
# used. The defaults (lr 1e-6, 1 epoch, batch 32, 16384 tokens) are the reported recipe.
#
#   scripts/sft_mutation.sh --records round1/sft.jsonl --model runs/rpl/global_step_200/actor/huggingface \
#       --out runs/sft1

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"

rpl_defaults
MODEL=""
RECORDS=""
SFT_EPOCHS=1
SFT_BS=32
SFT_MICRO_BS=1
SFT_LR=1e-6
SFT_MAX_LEN=16384
SUPERVISE=program
PROMPT_FORMAT=""

REST=()
while [ $# -gt 0 ]; do
  case "$1" in
    --records)    RECORDS="$2"; shift 2 ;;
    --epochs)     SFT_EPOCHS="$2"; shift 2 ;;
    --batch-size) SFT_BS="$2"; shift 2 ;;
    --lr)         SFT_LR="$2"; shift 2 ;;
    --max-len)    SFT_MAX_LEN="$2"; shift 2 ;;
    --supervise)  SUPERVISE="$2"; shift 2 ;;
    --prompt-format) PROMPT_FORMAT="$2"; shift 2 ;;
    --)           REST+=("$@"); break ;;
    *)            REST+=("$1"); shift ;;
  esac
done
EVAL_ONLY=1   # no RL data needed; skips the train.parquet check
rpl_parse_args "${REST[@]+"${REST[@]}"}"
PROMPT_FORMAT="${PROMPT_FORMAT:-$CHAT_TEMPLATE}"
rpl_resolve_topology
rpl_require_visible_gpus

[ -n "$RECORDS" ] || { echo "ERROR: --records FILE is required (from scripts/run_mutation_round.sh)" >&2; exit 1; }
[ -n "$MODEL" ] || { echo "ERROR: --model is required: the RL checkpoint the round harvested from" >&2; exit 1; }
[ -f "$RECORDS" ] || { echo "ERROR: records file not found: $RECORDS" >&2; exit 1; }
case "$PROMPT_FORMAT" in apply|raw) ;; *) echo "ERROR: --prompt-format must be apply or raw" >&2; exit 1 ;; esac
if [ "$PROMPT_FORMAT" = "apply" ] && [ "$CHAT_TEMPLATE" != "apply" ]; then
  echo "ERROR: --prompt-format apply with --chat-template $CHAT_TEMPLATE would SFT on a templated prompt the RL" >&2
  echo "       rollouts never see -- the raw-under-apply mismatch, reversed." >&2
  exit 1
fi
if [ "$PROMPT_FORMAT" = "raw" ] && [ "$CHAT_TEMPLATE" = "apply" ]; then
  echo "  NOTE: --prompt-format raw under --chat-template apply: the reported PM encoding, no chat markers"
fi

# Tokenize with the tokenizer of the model being trained -- a mismatch shifts every mask boundary.
SFT_PARQUET="${OUT_DIR}/sft_records.parquet"
mkdir -p "$OUT_DIR"
python "$HERE/program_mutation/mutation_sft.py" --records "$RECORDS" --tokenize \
  --model "$MODEL" --supervise "$SUPERVISE" --out "$SFT_PARQUET" --max_len "$SFT_MAX_LEN" \
  --prompt_format "$PROMPT_FORMAT" --enable_thinking "$ENABLE_THINKING"

ROWS="$(python -c "import pandas; print(len(pandas.read_parquet('$SFT_PARQUET')))")"
[ "$ROWS" -gt 0 ] || { echo "ERROR: $SFT_PARQUET has 0 rows -- nothing to train on." >&2; exit 1; }
# A set smaller than one batch would run zero optimizer steps and still write a checkpoint.
if [ "$ROWS" -lt "$SFT_BS" ]; then
  echo "  NOTE: $ROWS rows < batch $SFT_BS; training with batch $ROWS"
  SFT_BS="$ROWS"
fi
echo "== SFT on $ROWS rows (supervise=$SUPERVISE, prompt_format=$PROMPT_FORMAT) =="

rpl_install_pth
# verl asks Ray for 10 CPUs per worker and reads the cgroup's CPU share, which makes the autoscaler spin
# forever on a container. This is Ray's documented workaround for that detection.
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1 RAY_DISABLE_DOCKER_CPU_WARNING=1

CUDA_VISIBLE_DEVICES="$TRAIN_DEVS" python -m verl.trainer.sft_trainer_ray \
  trainer.n_gpus_per_node="$TRAIN_GPUS" trainer.nnodes=1 \
  data.train_files="$SFT_PARQUET" data.val_files=null \
  data.custom_cls.path="$REPO_ROOT/rpl/llm_agent/mutation_sft_dataset.py" \
  data.custom_cls.name=MutationSFTDataset \
  data.pad_mode=no_padding data.truncation=error data.max_length="$SFT_MAX_LEN" \
  data.train_batch_size="$SFT_BS" data.micro_batch_size_per_gpu="$SFT_MICRO_BS" \
  data.use_dynamic_bsz=True data.max_token_len_per_gpu="$SFT_MAX_LEN" \
  model.path="$MODEL" model.enable_gradient_checkpointing=True \
  optim.lr="$SFT_LR" \
  trainer.total_epochs="$SFT_EPOCHS" \
  trainer.project_name="$PROJECT" trainer.experiment_name="${EXP_NAME:-sft-mutation}" \
  trainer.default_local_dir="$OUT_DIR" \
  trainer.logger='["console"]' \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
  ${SEED:+trainer.seed="$SEED"} \
  checkpoint.save_contents='["model","hf_model"]' \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

# The fork starts from this export, so a run that wrote none must fail here rather than at the fork.
ls -d "$OUT_DIR"/global_step_*/huggingface >/dev/null 2>&1 \
  || { echo "ERROR: no global_step_*/huggingface export under $OUT_DIR" >&2; exit 1; }
echo "== SFT done: $(ls -d "$OUT_DIR"/global_step_*/huggingface | sort -V | tail -1) =="
