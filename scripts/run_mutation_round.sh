#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# One round of program mutation: harvest -> verify sets here, then (printed) eval -> SFT records -> SFT -> fork.
#
#   harvest  for each rollout whose program failed to retrieve gold, search the program grammar for a repair
#            (by default the nearest edit) whose evidence holds a gold answer. Retrieval only, no generation --
#            it measures the retrieval CEILING an edit could deliver.
#   verify   re-answer each repaired prefix, and separately re-answer the ORIGINAL prefix as a resample
#            control. The difference between the two is the part attributable to the repair rather than to
#            simply answering again; without the control a round reports resampling noise as a method effect.
#   sft      every verify row becomes a record: the repaired program is supervised on all of them, the answer
#            only where the repaired prefix was answered correctly (`--keep_all`, as in every reported round).
#
# Harvest and verify replay programs against both retrieval backends and the reranker. Start them first.
#
#   scripts/run_mutation_round.sh --rollouts runs/rpl/rollouts --from-step 151 --to-step 200 \
#       --data data/nq_hotpotqa_rpl --out round1

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM="$HERE/program_mutation"
source "$HERE/lib/common.sh"   # the service probes

ROLLOUT_DIR=""
DATA_DIR=""
OUT=""
FROM_STEP=""
TO_STEP=""
RETRIEVER_URL="http://127.0.0.1:8000/retrieve"
RERANK_URL="http://127.0.0.1:6980/rerank"
REPAIR_ORDER=nearest
LIMIT=""
CHAT_TEMPLATE=apply
DRY_RUN=0

function usage {
  cat <<USAGE
Usage: $(basename "$0") --rollouts DIR --from-step N --to-step N --data DIR --out DIR [options]

  --rollouts DIR       an RPL run's rollouts/ dir, i.e. <out>/rollouts  [required]
  --from-step N        first training step to harvest from              [required]
  --to-step N          last training step; the round's policy is this step's HF export  [required]
  --data DIR           the parquet dir that run trained on              [required]
  --out DIR            round artifacts                                  [required]
  --retriever-url URL  (default: ${RETRIEVER_URL})
  --rerank-url URL     (default: ${RERANK_URL})
  --chat-template M    apply | raw, as the run trained; carried into the printed commands (default: ${CHAT_TEMPLATE})
  --limit N            cap the rows processed, for a smoke run
  --repair-order O     nearest | cheapest | breadth | rerank (default: ${REPAIR_ORDER})
                       Which repair is chosen when several work. This decides what the SFT set teaches and
                       it changes the outcome -- it is not a tie-break detail.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --rollouts)      ROLLOUT_DIR="$2"; shift 2 ;;
    --from-step)     FROM_STEP="$2"; shift 2 ;;
    --to-step)       TO_STEP="$2"; shift 2 ;;
    --data)          DATA_DIR="$2"; shift 2 ;;
    --out)           OUT="$2"; shift 2 ;;
    --retriever-url) RETRIEVER_URL="$2"; shift 2 ;;
    --rerank-url)    RERANK_URL="$2"; shift 2 ;;
    --chat-template) CHAT_TEMPLATE="$2"; shift 2 ;;
    --limit)         LIMIT="$2"; shift 2 ;;
    --repair-order)  REPAIR_ORDER="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *)               echo "ERROR: unknown option '$1'" >&2; exit 2 ;;
  esac
done

for pair in "rollouts:$ROLLOUT_DIR" "from-step:$FROM_STEP" "to-step:$TO_STEP" "data:$DATA_DIR" "out:$OUT"; do
  [ -n "${pair#*:}" ] || { echo "ERROR: --${pair%%:*} is required" >&2; usage; exit 2; }
done
[ -d "$ROLLOUT_DIR" ] || { echo "ERROR: rollout dir not found: $ROLLOUT_DIR" >&2; exit 1; }
[ -f "$DATA_DIR/train.parquet" ] || { echo "ERROR: $DATA_DIR/train.parquet not found" >&2; exit 1; }
case "$CHAT_TEMPLATE" in apply|raw) ;; *) echo "ERROR: --chat-template must be apply or raw" >&2; exit 2 ;; esac

# The modules import each other as siblings and run from $PM, so resolve the user's paths first.
ROLLOUT_DIR="$(cd "$ROLLOUT_DIR" && pwd)"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
mkdir -p "$OUT/rollouts"
OUT="$(cd "$OUT" && pwd)"
WINDOW="$OUT/rollouts/window.jsonl"   # harvest and verify read only from a rollouts/ dir
POLICY="$(dirname "$ROLLOUT_DIR")/global_step_${TO_STEP}/actor/huggingface"
CT=""; [ "$CHAT_TEMPLATE" = "raw" ] && CT=" --chat-template raw"

rpl_require_retriever dense sparse
rpl_require_reranker
[ -f "$POLICY/config.json" ] || echo "  WARNING: no HF export at $POLICY; the evals and the SFT below start from it" >&2

cd "$PM"
LIM=(); [ -n "$LIMIT" ] && LIM=(--limit "$LIMIT")

echo "== 1/3 merge rollout dumps (steps $FROM_STEP..$TO_STEP) =="
python merge_rollouts.py --dir "$ROLLOUT_DIR" --lo "$FROM_STEP" --hi "$TO_STEP" --out "$WINDOW"

echo "== 2/3 harvest repairs (repair_order=$REPAIR_ORDER) =="
python mutation_harvest.py --dump "$WINDOW" --parquet "$DATA_DIR/train.parquet" \
  --url "$RETRIEVER_URL" --rerank_url "$RERANK_URL" \
  --repair_order "$REPAIR_ORDER" "${LIM[@]+"${LIM[@]}"}" --out "$OUT/repairs.jsonl"

echo "== 3/3 build the verify and control evaluation sets =="
python mutation_verify.py --dump "$WINDOW" --repairs "$OUT/repairs.jsonl" \
  --parquet "$DATA_DIR/train.parquet" --url "$RETRIEVER_URL" --rerank_url "$RERANK_URL" \
  "${LIM[@]+"${LIM[@]}"}" --out "$OUT/verify.parquet" --control_out "$OUT/control.parquet"

cat <<NEXT
== round staged in $OUT ==
1. Answer both sets with the policy the window came from, greedily and in one pass. Verify minus control EM is
   what the repairs themselves earn:

  scripts/eval.sh --data $OUT --val $OUT/verify.parquet  --mode code --single-pass$CT \\
      --ckpt $POLICY --out $OUT/eval_verify
  scripts/eval.sh --data $OUT --val $OUT/control.parquet --mode code --single-pass$CT \\
      --ckpt $POLICY --out $OUT/eval_control
  python tools/qa_report.py score --dump $OUT/eval_verify/val_rollouts/0.jsonl  --parquet $OUT/verify.parquet
  python tools/qa_report.py score --dump $OUT/eval_control/val_rollouts/0.jsonl --parquet $OUT/control.parquet

2. Turn the verify rollouts into SFT records and fine-tune the same policy on them (the reported rows also
   passed --prompt-format raw):

  python $PM/mutation_sft.py --verify_parquet $OUT/verify.parquet \\
      --verify_rollouts $OUT/eval_verify/val_rollouts/0.jsonl --out $OUT/sft.jsonl
  scripts/sft_mutation.sh --records $OUT/sft.jsonl --model $POLICY$CT --out $OUT/sft

3. Fork RL from the export the SFT printed last. The reported forks ran 800 steps after a step-200 round:

  scripts/train_rpl.sh --data $DATA_DIR --init-from $OUT/sft/global_step_<last>/huggingface --steps 800$CT \\
      --out $OUT/fork

Then compare the fork against its parent at matched total steps:

  python tools/qa_report.py score --dump $OUT/fork/val_rollouts/<step>.jsonl --parquet $DATA_DIR/test.parquet
  python tools/qa_report.py score --dump <parent run>/val_rollouts/<step + $TO_STEP>.jsonl \\
      --parquet $DATA_DIR/test.parquet
NEXT
