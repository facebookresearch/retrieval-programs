# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin
from transformers.dynamic_module_utils import custom_object_save

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank
from verl.utils.transformers_compat import drop_tied_target_keys, get_auto_model_for_vision2seq

from .async_ckpt import copy_entries, entries_for_rank, prune_empty_dirs, remove_entries, wait_for_paths
from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Node-local scratch that `async_save` stages shards into before copying them to durable storage. Ambient
# per-node config (the path only makes sense on the box), hence env rather than a config field.
_STAGE_DIR_ENV = "VERL_CKPT_STAGE_DIR"
_STAGE_DIR_DEFAULT = "/tmp/verl_ckpt_stage"

# How long rank 0 waits for every rank's copy before giving up on advancing the tracker. Comfortably above
# a slow 8B copy (~19 min) but well under one checkpoint interval, because the next save joins this thread:
# if some rank's copy dies, this timeout is how long that save stalls.
_FINALIZE_TIMEOUT_S = 45 * 60


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
        trust_remote_code: Whether to trust_remote_code when loading the model configuration
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        if processing_class is None and "tokenizer" in kwargs:
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )
        self.trust_remote_code = trust_remote_code
        # Async (local-staged) saving -- see `_start_async_copy`. Reuses verl's own `async_save` flag, which
        # the Megatron manager already implements and which `ray_trainer._save_checkpoint` already honours by
        # skipping the `latest_checkpointed_iteration.txt` write; we take over writing that tracker once the
        # copy to durable storage lands.
        self.async_save = bool(self.checkpoint_config and self.checkpoint_config.get("async_save", False))
        self.stage_root = os.environ.get(_STAGE_DIR_ENV, _STAGE_DIR_DEFAULT)
        self._copy_thread: Optional[threading.Thread] = None

    def _get_lora_train_meta(self, unwrap_model):
        peft_config = getattr(unwrap_model, "peft_config", None)
        if not peft_config:
            return None
        if isinstance(peft_config, dict):
            peft_config = peft_config.get("default") or next(iter(peft_config.values()), None)
        if peft_config is None:
            return None

        lora_rank = int(getattr(peft_config, "r", 0) or 0)
        if lora_rank <= 0:
            return None

        lora_alpha = int(getattr(peft_config, "lora_alpha", lora_rank) or 0)
        task_type = getattr(peft_config, "task_type", None) or "CAUSAL_LM"
        if hasattr(task_type, "value"):
            task_type = task_type.value

        return {"r": lora_rank, "lora_alpha": lora_alpha, "task_type": str(task_type)}

    def _save_lora_train_meta(self, local_path: str, unwrap_model):
        lora_train_meta = self._get_lora_train_meta(unwrap_model)
        if lora_train_meta is None:
            return None

        lora_meta_path = os.path.join(local_path, "lora_train_meta.json")
        with open(lora_meta_path, "w", encoding="utf-8") as f:
            json.dump(lora_train_meta, f, ensure_ascii=False, indent=4)
        log_with_rank(
            f"Saved LoRA rank/alpha metadata to {os.path.abspath(lora_meta_path)}",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )
        return lora_meta_path

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # check if the checkpoint_load_contents is valid
        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        # every rank download its own checkpoint
        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_model
            else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if self.should_load_model:
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                self.model.load_state_dict(model_state_dict)
                log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)

            if self.should_load_optimizer:
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state_dict)
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

        if self.should_load_extra:
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        if self.rank == 0 and del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        if self.rank == 0:
            self.ensure_checkpoint_capacity(max_ckpt_to_keep)

        # With async_save the shards are written to node-local NVMe and copied to `final_path` in the
        # background, so the training loop only pays the local write (seconds) instead of the WSF write
        # (~19 min for an 8B checkpoint). Everything below is unchanged and simply targets the staging dir.
        final_path = local_path
        if self.async_save:
            self.wait_for_pending_saves()  # bounds staging disk and keeps pruning ordered
            local_path = os.path.join(self.stage_root, os.path.basename(os.path.dirname(final_path.rstrip("/"))),
                                      os.path.basename(final_path.rstrip("/")))
            shutil.rmtree(local_path, ignore_errors=True)  # drop a partial stage left by a killed attempt

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:
                    model_state_dict = self.model.state_dict()
                    torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:
                    optimizer_state_dict = self.optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

            if hasattr(model_config, "auto_map") and None in model_config.auto_map:
                model_config.auto_map = {k: v for k, v in model_config.auto_map.items() if k is not None}

            model_config.save_pretrained(hf_config_tokenizer_path)
            if self.processing_class is not None:
                self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
            # loaded from the Hub.
            if hasattr(model_config, "auto_map"):
                custom_object_save(unwrap_model, hf_config_tokenizer_path, config=model_config)

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)
            self._save_lora_train_meta(local_path, unwrap_model)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.should_save_hf_model:
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    auto_model_cls = get_auto_model_for_vision2seq()
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(
                        model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.trust_remote_code
                    )

                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                drop_tied_target_keys(state_dict, save_model, model_config)

                # safetensors' serialize_file EINVALs writing straight to some network filesystems, so save
                # the (correctly multi-sharded) HF model to a LOCAL temp dir first, then copy each file over
                # with plain read+write (shutil.copyfile), which they do support.
                with tempfile.TemporaryDirectory() as _local_hf_dir:
                    save_model.save_pretrained(_local_hf_dir, state_dict=state_dict)
                    for _hf_file in os.listdir(_local_hf_dir):
                        shutil.copyfile(
                            os.path.join(_local_hf_dir, _hf_file), os.path.join(hf_local_path, _hf_file)
                        )
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        if self.async_save:
            self._start_async_copy(local_path, final_path, global_step, max_ckpt_to_keep)
        elif self.rank == 0:
            self.register_checkpoint(local_path, max_ckpt_to_keep)

    def wait_for_pending_saves(self, timeout: Optional[float] = None) -> None:
        """Block until this rank's outstanding async copy (if any) has finished.

        Called at the top of every save so at most one copy is ever in flight, and worth calling once more
        at the end of training: the copy threads are non-daemon, so a normal interpreter exit waits for
        them, but an explicit join gives a clearer failure than a silent hang.
        """
        thread, self._copy_thread = self._copy_thread, None
        if thread is None:
            return
        thread.join(timeout)
        if thread.is_alive():
            logger.error("checkpoint copy thread still running after %.0fs", timeout or 0.0)
            self._copy_thread = thread  # not finished -- keep tracking it rather than losing the handle

    def _start_async_copy(self, stage_path: str, final_path: str, global_step: int, max_ckpt_to_keep) -> None:
        """Copy this rank's staged files to `final_path` in the background, then finalize on rank 0.

        Durability protocol: a checkpoint only becomes authoritative when
        `latest_checkpointed_iteration.txt` names it, so a copy interrupted by preemption is simply never
        pointed at and resume falls back to the previous step. That is why no atomic rename is needed --
        on a network filesystem the rename is the fragile part; plain `copyfile` is not.
        """
        marker = os.path.join(final_path, f"_copied_rank_{self.rank}")

        def _worker():
            started = time.time()
            names: list[str] = []
            try:
                names = entries_for_rank(os.listdir(stage_path), self.rank)
                copy_entries(stage_path, final_path, names)
                with open(marker, "w") as f:
                    f.write(str(global_step))
                log_with_rank(
                    f"async checkpoint copy for step {global_step} took {time.time() - started:.0f}s",
                    rank=self.rank,
                    logger=logger,
                )
                if self.rank == 0:
                    self._finalize_async_copy(final_path, global_step, max_ckpt_to_keep)
            except Exception:
                # Leave the tracker alone: the previous checkpoint stays authoritative.
                logger.exception("async checkpoint copy failed for step %s; tracker not advanced", global_step)
            finally:
                # Only this rank's entries -- `stage_path` is shared by every rank on the node.
                remove_entries(stage_path, names)
                prune_empty_dirs(stage_path)

        # Non-daemon so a normal interpreter exit waits for an in-flight copy instead of truncating it.
        self._copy_thread = threading.Thread(target=_worker, name=f"ckpt-copy-{global_step}", daemon=False)
        self._copy_thread.start()

    def _finalize_async_copy(self, final_path: str, global_step: int, max_ckpt_to_keep) -> None:
        """Rank 0: wait for every rank's copy, then register the checkpoint and advance the tracker.

        Ranks signal completion with marker files rather than a collective: this runs on a side thread, and
        calling into the process group from off the training thread is not safe.
        """
        # <root>/global_step_N/<role> -> step dir holds data.pt, root holds the tracker.
        step_dir = os.path.dirname(final_path.rstrip("/"))
        root_dir = os.path.dirname(step_dir)
        # data.pt is written by the trainer just after save_checkpoint returns; on a fast copy we could
        # otherwise advertise a step whose dataloader state has not landed yet.
        required = [os.path.join(final_path, f"_copied_rank_{r}") for r in range(self.world_size)]
        required.append(os.path.join(step_dir, "data.pt"))

        # print, not logger: these run on a Ray worker where info-level records do not reach the job log,
        # and a finalize that is merely still WAITING looks identical to one that never started.
        print(f"CKPT: finalize step {global_step}: waiting for {len(required)} paths", flush=True)
        missing = wait_for_paths(required, _FINALIZE_TIMEOUT_S)
        if missing:
            logger.error(
                "async checkpoint step %s incomplete after %ss (missing %s); tracker not advanced",
                global_step, _FINALIZE_TIMEOUT_S, missing[:3],
            )
            print(f"CKPT: finalize step {global_step} TIMED OUT, missing {missing[:3]}", flush=True)
            return

        self.register_checkpoint(final_path, max_ckpt_to_keep)
        tracker = os.path.join(root_dir, "latest_checkpointed_iteration.txt")
        with open(tracker, "w") as f:
            f.write(str(global_step))
        print(f"CKPT: finalize step {global_step} complete; wrote {tracker}", flush=True)
        log_with_rank(
            f"async checkpoint step {global_step} durable; tracker advanced",
            rank=self.rank,
            logger=logger,
            log_only_rank_0=True,
        )
