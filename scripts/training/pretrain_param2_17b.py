#!/usr/bin/env python3
"""

# Single GPU (dev / smoke test):
    torchrun --nproc-per-node=1 train.py --recipe single_gpu --mock

# 8 GPUs, one node (EP=8):
    torchrun --nproc-per-node=8 train.py --recipe 8gpu --mock

# 32 GPUs, 4 nodes (PP=2, EP=16):
    torchrun --nnodes=4 --nproc-per-node=8 \\
             --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29400 \\
             train.py --recipe 32gpu \\
             --train-data /data/train --valid-data /data/valid

# Continue from checkpoint:
    torchrun --nproc-per-node=8 train.py --recipe 8gpu \\
             --load /checkpoints/param2_17b_8gpu/checkpoints

Configuration overrides (dot-notation):
    torchrun --nproc-per-node=8 train.py --recipe 8gpu \\
             train.train_iters=500000 \\
             train.global_batch_size=128 \\
             scheduler.lr=1e-4

Requirements
    pip install megatron-bridge transformers torch torchvision
"""

import argparse
import os
import sys
import logging
from typing import Optional


from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.gpt_step import forward_step


from param2_recipe import (
    param2_17b_single_gpu_config,
    param2_17b_8gpu_config,
    param2_17b_32gpu_config,
    HF_MODEL_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-train Param2-17B-A2.4B-Thinking with Megatron-Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Recipe selection ---------------------------------------------------
    parser.add_argument(
        "--recipe",
        choices=["single_gpu", "8gpu", "32gpu"],
        default="single_gpu",
        help="Pre-built parallelism recipe to use.",
    )

    # ---- Model source -------------------------------------------------------
    parser.add_argument(
        "--hf-path",
        default=HF_MODEL_PATH,
        help="HuggingFace model id or local path for Param2MoE config/weights.",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load pretrained HF weights before training (continue from HF ckpt).",
    )

    # ---- Data ---------------------------------------------------------------
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use mock (synthetic) data. Useful for testing the training loop "
             "without real data.",
    )
    parser.add_argument(
        "--data-paths", nargs="+", default=None,
        help="Megatron-style binary data paths (e.g. /data/corpus_text_document). "
             "Weights are interleaved if multiple paths are given.",
    )
    parser.add_argument(
        "--train-data", default=None,
        help="Path to training data (Megatron binary format prefix).",
    )
    parser.add_argument(
        "--valid-data", default=None,
        help="Path to validation data (Megatron binary format prefix).",
    )
    parser.add_argument(
        "--test-data", default=None,
        help="Path to test data (Megatron binary format prefix).",
    )

    # ---- Checkpoint ---------------------------------------------------------
    parser.add_argument(
        "--load", default=None,
        help="Path to Megatron checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--save", default=None,
        help="Override checkpoint save directory.",
    )
    parser.add_argument(
        "--save-interval", type=int, default=None,
        help="Override checkpoint save interval (steps).",
    )

    # ---- Output dirs --------------------------------------------------------
    parser.add_argument(
        "--output-dir", default=None,
        help="Root directory for experiments (checkpoints, logs, tensorboard).",
    )
    parser.add_argument(
        "--name", default=None,
        help="Experiment name (sub-directory inside output-dir).",
    )

    # ---- Quick overrides (common training knobs) ----------------------------
    parser.add_argument("--train-iters",       type=int,   default=None)
    parser.add_argument("--global-batch-size", type=int,   default=None)
    parser.add_argument("--micro-batch-size",  type=int,   default=None)
    parser.add_argument("--seq-length",        type=int,   default=None)
    parser.add_argument("--lr",                type=float, default=None,
                        help="Peak learning rate.")
    parser.add_argument("--min-lr",            type=float, default=None)
    parser.add_argument("--lr-warmup-iters",   type=int,   default=None)

    # ---- Parallelism overrides (for custom cluster topologies) --------------
    parser.add_argument("--tp",  type=int, default=None, dest="tp",
                        help="Tensor model parallel size.")
    parser.add_argument("--pp",  type=int, default=None, dest="pp",
                        help="Pipeline model parallel size.")
    parser.add_argument("--ep",  type=int, default=None, dest="ep",
                        help="Expert model parallel size.")

    # ---- FP8 ----------------------------------------------------------------
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Enable FP8 training (requires H100/Blackwell with TE).",
    )
    parser.add_argument(
        "--fp8-recipe",
        choices=["tensorwise", "blockwise", "mxfp8"],
        default="tensorwise",
    )

    # ---- Dot-notation overrides (Hydra-style, passed after all flags) -------
    # Any remaining args of the form key=value are parsed as config overrides.
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


# =========================================================================== #
#   Config builder                                                             #
# =========================================================================== #

def build_config(args):
    """Build ConfigContainer from recipe + CLI overrides."""

    recipe_kwargs = dict(
        hf_path=args.hf_path,
        mock=args.mock or (
            args.data_paths is None
            and args.train_data is None
        ),
        data_paths=args.data_paths,
        train_data_path=args.train_data,
        valid_data_path=args.valid_data,
        test_data_path=args.test_data,
        dir=args.output_dir,
        name=args.name or f"param2_17b_{args.recipe}",
        load_weights=args.load_weights,
    )

    recipe_map = {
        "single_gpu": param2_17b_single_gpu_config,
        "8gpu":       param2_17b_8gpu_config,
        "32gpu":      param2_17b_32gpu_config,
    }
    cfg = recipe_map[args.recipe](**recipe_kwargs)

    # ---- Apply structured CLI overrides ------------------------------------
    if args.train_iters       is not None: cfg.train.train_iters           = args.train_iters
    if args.global_batch_size is not None: cfg.train.global_batch_size     = args.global_batch_size
    if args.micro_batch_size  is not None: cfg.train.micro_batch_size      = args.micro_batch_size
    if args.seq_length        is not None:
        cfg.model.seq_length  = args.seq_length
        cfg.dataset.seq_length = args.seq_length
    if args.lr                is not None: cfg.scheduler.lr                = args.lr
    if args.min_lr            is not None: cfg.scheduler.min_lr            = args.min_lr
    if args.lr_warmup_iters   is not None: cfg.scheduler.lr_warmup_iters   = args.lr_warmup_iters
    if args.save              is not None: cfg.checkpoint.save             = args.save
    if args.load              is not None: cfg.checkpoint.load             = args.load
    if args.save_interval     is not None: cfg.checkpoint.save_interval    = args.save_interval

    # ---- Parallelism overrides --------------------------------------------
    if args.tp is not None: cfg.model.tensor_model_parallel_size   = args.tp
    if args.pp is not None: cfg.model.pipeline_model_parallel_size = args.pp
    if args.ep is not None: cfg.model.expert_model_parallel_size   = args.ep

    # ---- FP8 ---------------------------------------------------------------
    if args.fp8:
        cfg.mixed_precision.fp8 = True
        cfg.mixed_precision.fp8_recipe = args.fp8_recipe
        cfg.model.moe_router_padding_for_fp8 = True
        log.info(f"FP8 training enabled with recipe: {args.fp8_recipe}")

    # ---- Dot-notation overrides (e.g. train.train_iters=500000) -----------
    for override in args.overrides:
        if "=" not in override:
            log.warning(f"Skipping unrecognised argument: {override!r}")
            continue
        key, value = override.split("=", 1)
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        # Attempt type inference
        old_val = getattr(obj, parts[-1], None)
        if isinstance(old_val, bool):
            new_val = value.lower() in ("true", "1", "yes")
        elif isinstance(old_val, int):
            new_val = int(value)
        elif isinstance(old_val, float):
            new_val = float(value)
        else:
            new_val = value
        setattr(obj, parts[-1], new_val)
        log.info(f"Override: {key} = {new_val!r}")

    return cfg


# =========================================================================== #
#   Validation helpers                                                         #
# =========================================================================== #

def validate_config(cfg):
    """Sanity-check parallelism settings for Param2-17B."""
    tp = cfg.model.tensor_model_parallel_size
    pp = cfg.model.pipeline_model_parallel_size
    ep = cfg.model.expert_model_parallel_size
    num_layers = 21

    # PP: num_layers must be divisible (approximately — MCore handles odd splits)
    if pp > 1 and num_layers % pp != 0:
        log.warning(
            f"num_layers={num_layers} is not exactly divisible by PP={pp}. "
            "Using pipeline_model_parallel_layout to split manually."
        )

    # EP: should divide num_experts=64
    if 64 % ep != 0:
        raise ValueError(
            f"expert_model_parallel_size={ep} does not divide num_experts=64. "
            "Choose EP from: 1, 2, 4, 8, 16, 32, 64."
        )

    log.info(f"Parallelism: TP={tp}, PP={pp}, EP={ep}")
    log.info(f"Global batch size: {cfg.train.global_batch_size}")
    log.info(f"Micro batch size : {cfg.train.micro_batch_size}")
    log.info(f"Seq length       : {cfg.model.seq_length}")
    log.info(f"Train iters      : {cfg.train.train_iters}")
    log.info(f"Checkpoint save  : {cfg.checkpoint.save}")
    log.info(f"Checkpoint load  : {cfg.checkpoint.load}")


# =========================================================================== #
#   Optional: load HF weights into MCore model before training               #
# =========================================================================== #

class HFWeightLoader:
    """
    Callback hook used to load HF pretrained weights into the MCore model
    after it has been instantiated by Megatron-Bridge's pretrain() function.

    This is injected via cfg.model._weight_loader_callback if load_weights=True.
    """
    def __init__(self, hf_path: str):
        from param2_bridge import Param2MoEBridge
        self.bridge = Param2MoEBridge.from_hf_pretrained(hf_path)

    def __call__(self, mcore_model):
        log.info("Loading HF pretrained weights into MCore model ...")
        self.bridge.load_hf_weights_into_mcore(mcore_model)
        log.info("HF weights loaded successfully.")
        return mcore_model


# =========================================================================== #
#   Main                                                                       #
# =========================================================================== #

def main():
    args = parse_args()

    log.info("=" * 70)
    log.info("  Param2-17B-A2.4B-Thinking Pre-training")
    log.info("  Backend: Megatron-Bridge (Megatron Core)")
    log.info(f"  Recipe : {args.recipe}")
    log.info(f"  HF path: {args.hf_path}")
    log.info("=" * 70)

    # Build config
    cfg = build_config(args)
    validate_config(cfg)

    # Optionally attach HF weight loader
    if args.load_weights:
        loader = HFWeightLoader(args.hf_path)
        # Megatron-Bridge's pretrain() accepts a model_post_init_fn kwarg
        # that is called after model construction. We use it to inject weights.
        log.info("HF weight loading is enabled — weights will be loaded after "
                 "model construction.")
        pretrain(cfg, forward_step, model_post_init_fn=loader)
    else:
        # Standard pre-training from random init (or Megatron checkpoint via --load)
        pretrain(cfg, forward_step)


if __name__ == "__main__":
    main()