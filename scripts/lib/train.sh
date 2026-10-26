#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The single invocation every training and eval arm goes through.
#
# Arms differ only in the variables they set before calling `rpl_train`:
#   RPL_MODE        search | code | marag   -- the action interface
#   RPL_EXTRA       array of additional `+rpl.*` Hydra assignments

function rpl_train {
  : "${RPL_MODE:?rpl_train: RPL_MODE must be set}"
  # `${arr[@]:-}` on an EMPTY array expands to one empty string, which Hydra then rejects as a
  # malformed override. `+()` keeps an empty array empty under `set -u`.
  local -a extra=("${RPL_EXTRA[@]+"${RPL_EXTRA[@]}"}")
  local name="${EXP_NAME:-${RPL_MODE}-$(basename "$MODEL")-$(date +%m%d-%H%M)}"

  # Hydra reads `a.parquet:b.parquet` as one path, so the eval sets go in as a list.
  local -a val=()
  local f
  for f in ${VAL_FILES//:/ }; do val+=("'$f'"); done
  # verl asserts that every sequence fits the dynamic-batch token budget.
  local ppo_max="${PPO_MAX_TOKEN_LEN:-$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))}"
  if [ "$ppo_max" -lt $((MAX_PROMPT_LEN + MAX_RESPONSE_LEN)) ]; then
    echo "ERROR: PPO_MAX_TOKEN_LEN $ppo_max < prompt $MAX_PROMPT_LEN + response $MAX_RESPONSE_LEN" >&2
    exit 1
  fi

  local -a cmd=(
    python -m verl.trainer.main_ppo_sync
    algorithm.adv_estimator="$ADV_ESTIMATOR"

    data.train_files="${TRAIN_FILES:-$DATA_DIR/train.parquet}"
    data.val_files="[$(IFS=,; echo "${val[*]}")]"
    data.train_batch_size="$TRAIN_BS"
    data.max_prompt_length="$MAX_PROMPT_LEN"
    data.max_response_length="$MAX_RESPONSE_LEN"
    # Truncating a prompt silently changes the task; fail instead.
    data.filter_overlong_prompts=False
    data.truncation=error

    actor_rollout_ref.model.path="$MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr="$LR"
    actor_rollout_ref.actor.optim.lr_warmup_steps="$LR_WARMUP_STEPS"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef="$KL_COEF"
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ppo_max"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.param_offload=$([ "$ACTOR_OFFLOAD" = "1" ] && echo True || echo False)
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$([ "$ACTOR_OFFLOAD" = "1" ] && echo True || echo False)

    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP"
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
    actor_rollout_ref.rollout.n="$N_ROLL"
    actor_rollout_ref.rollout.temperature=1
    # Validation is greedy.
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.agent.default_agent_loop=rpl
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO_ROOT/rpl/config/agent_loop.yaml"

    # The agent loop's own config block. It lives at the config TOP LEVEL rather than under `rollout.agent`,
    # which is a strict dataclass that rejects unknown keys.
    +rpl.mode="$RPL_MODE"
    +rpl.search_url="$RETRIEVER_URL"
    +rpl.rerank_url="$RERANK_URL"
    +rpl.topk="$TOPK"
    +rpl.max_turns="$MAX_TURNS"
    +rpl.max_gen_tokens="$MAX_GEN_TOKENS"
    +rpl.max_obs_length="$MAX_OBS_LENGTH"
    +rpl.chat_template_mode="$CHAT_TEMPLATE"
    +rpl.enable_thinking="$ENABLE_THINKING"
    +rpl.add_bos_to_prompt="$ADD_BOS"
    +rpl.sandbox_timeout_sec="$SANDBOX_TIMEOUT_SEC"
    +rpl.persist_sandbox_state="$PERSIST_SANDBOX_STATE"
    +rpl.max_state_bytes="$MAX_STATE_BYTES"
    +rpl.reward_mode="$REWARD_MODE"
    +rpl.min_searches="$MIN_SEARCHES"
    +rpl.doc_chars="$DOC_CHARS"
    +rpl.rerank_max_docs="$RERANK_MAX_DOCS"
    +rpl.globalqa_doc_reward="$DOC_REWARD"

    reward.reward_manager.source=importlib
    reward.reward_manager.name=RPLRewardManager
    reward.reward_manager.module.path=pkg://rpl.llm_agent.rpl_reward_manager

    trainer.logger="$LOGGER"
    trainer.project_name="$PROJECT"
    trainer.experiment_name="$name"
    trainer.n_gpus_per_node="$TRAIN_GPUS"
    trainer.nnodes=1
    trainer.balance_batch="$BALANCE_BATCH"
    trainer.save_freq="$SAVE_FREQ"
    trainer.test_freq="$TEST_FREQ"
    trainer.total_training_steps="$STEPS"
    # total_epochs is an independent cap: leave it high or it silently ends a run below --steps.
    trainer.total_epochs=100
    trainer.max_actor_ckpt_to_keep="$MAX_CKPT_KEEP"
    trainer.max_critic_ckpt_to_keep="$MAX_CKPT_KEEP"
    # The HF export is all that survives at older steps (the prune keeps only the latest FSDP shards), and the
    # evals, checkpoint selection, the mutation SFT and its fork all load it.
    actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra,hf_model]'
    # An eval must load --model, never the newest checkpoint it finds under --out.
    trainer.resume_mode=$([ "$EVAL_ONLY" = "1" ] && echo disable || echo auto)
    trainer.default_local_dir="$OUT_DIR"

    # Rollout dumps. These are not optional extras: the program-mutation pipeline consumes them.
    #   rollouts/<step>.jsonl      training generations -> scripts/run_mutation_round.sh --rollouts
    #   val_rollouts/<step>.jsonl  validation generations -> mutation_sft.py --verify_rollouts, and
    #                              tools/qa_report.py
    # Without them a run looks fine and then there is simply nothing downstream to read.
    trainer.rollout_data_dir="$OUT_DIR/rollouts"
    trainer.validation_data_dir="$OUT_DIR/val_rollouts"
    trainer.log_val_generations="$LOG_VAL_GENERATIONS"
  )

  if [ "$EVAL_ONLY" = "1" ]; then cmd+=(trainer.val_only=True); fi
  # The seven-set union (51,713 rows) in one batch runs out of file descriptors.
  if [ "${#val[@]}" -gt 1 ]; then cmd+=(data.val_batch_size=4096); fi
  # `++`: data_loader_seed already exists in the generated config, and `+` on an existing key is a Hydra error.
  if [ -n "$SEED" ]; then cmd+=("++data.seed=$SEED" "++actor_rollout_ref.actor.data_loader_seed=$SEED"); fi
  cmd+=("${extra[@]+"${extra[@]}"}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

  # Print before touching anything: a dry run must not write to site-packages or create directories.
  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "${cmd[@]}"; echo
    return 0
  fi

  rpl_install_pth
  mkdir -p "$OUT_DIR"
  echo "  experiment   $name"
  CUDA_VISIBLE_DEVICES="$TRAIN_DEVS" "${cmd[@]}"
}
