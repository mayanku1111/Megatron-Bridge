
# Architecture summary (from config.json):
#   Total params  : ~17B | Active : ~2.4B
#   Layers        : 21 (1 dense + 20 MoE)
#   Hidden size   : 2048
#   Attention     : 32H / 8KV GQA, head_dim=64
#   Experts       : 64 routed + 2 shared, top-6, sigmoid router
#   Vocab         : 128008 (tied embeddings)
#   Seq length    : 4096

import os
import torch
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer

# Import our custom bridge
from megatron.bridge.recipes.param2moe import (
    param2_17b_single_gpu_config,
    param2_17b_8gpu_config,
    param2_17b_32gpu_config,
)

HF_MODEL_PATH = os.environ.get(
    "PARAM2_HF_PATH",
    "bharatgenai/Param2-17B-A2.4B-Thinking",
)

def _build_model_provider(hf_path: str, tp: int, pp: int, ep: int,
                           load_weights: bool = False):
    """Construct MCore provider from Param2MoE HF config via our custom bridge."""
    bridge = Param2MoEBridge.from_hf_pretrained(hf_path)
    mcore_cfg = bridge.get_mcore_config(tp=tp, pp=pp, ep=ep)
    mcore_cfg.perform_initialization = not load_weights  # skip init when loading ckpt
    return mcore_cfg, bridge


def param2_17b_single_gpu_config(
    hf_path: str = HF_MODEL_PATH,
    mock: bool = True,
    data_paths=None,
    train_data_path=None,
    valid_data_path=None,
    test_data_path=None,
    dir: str = None,
    name: str = "param2_17b_pretrain",
    load_weights: bool = False,
) -> ConfigContainer:
    """
    Minimal single-GPU config for smoke-testing / development.

    Recommended GPU: ≥80 GB (A100-80G or H100-80G).
    Parallelism: TP=1, PP=1, EP=1  (no model parallelism).
    Uses gradient recomputation and micro_batch_size=1 to fit in memory.
    """
    cfg = _pretrain_common()

    # ---- Model (obtained via our custom bridge) -----------------------------
    mcore_cfg, bridge = _build_model_provider(hf_path, tp=1, pp=1, ep=1,
                                              load_weights=load_weights)
    cfg.model = mcore_cfg

    # ---- Output dirs --------------------------------------------------------
    base_dir = dir or os.path.join(os.getcwd(), "nemo_experiments")
    run_dir  = os.path.join(base_dir, name)
    cfg.checkpoint.save = os.path.join(run_dir, "checkpoints")

    cfg.tokenizer.tokenizer_type   = "NullTokenizer"
    cfg.tokenizer.tokenizer_model  = None
    cfg.tokenizer.vocab_size       = 128008    # Param2 vocab size

    # ---- Dataset ------------------------------------------------------------
    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, None,
        train_data_path, valid_data_path, test_data_path,
        None, mock,
    )
    cfg.dataset.blend          = blend
    cfg.dataset.blend_per_split = blend_per_split
    cfg.dataset.split          = split
    cfg.dataset.num_workers    = 4
    cfg.dataset.seq_length     = 4096

    # ---- Parallelism --------------------------------------------------------
    cfg.model.tensor_model_parallel_size          = 1
    cfg.model.pipeline_model_parallel_size        = 1
    cfg.model.pipeline_model_parallel_layout      = None
    cfg.model.pipeline_dtype                      = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size               = 1
    cfg.model.expert_model_parallel_size          = 1
    cfg.model.sequence_parallel                   = False
    cfg.model.seq_length                          = 4096

    # ---- MoE token dispatcher -----------------------------------------------
    cfg.model.moe_token_dispatcher_type  = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None   # no DeepEP on single GPU
    cfg.model.moe_grouped_gemm           = True
    cfg.model.moe_shared_expert_overlap  = False   # pointless on 1 GPU

    # ---- Training -----------------------------------------------------------
    cfg.train.train_iters        = 1_000_000
    cfg.train.global_batch_size  = 8
    cfg.train.micro_batch_size   = 1
    cfg.train.manual_gc          = True
    cfg.train.manual_gc_interval = 100

    # ---- Validation ---------------------------------------------------------
    cfg.validation.eval_interval = 2000

    # ---- Scheduler ----------------------------------------------------------
    cfg.scheduler.lr_warmup_iters = 2000

    # ---- Mixed precision ----------------------------------------------------
    # bf16_mixed is already set in _pretrain_common
    cfg.model.transformer_impl = "transformer_engine"

    # ---- CUDA Graph ---------------------------------------------------------
    cfg.model.cuda_graph_impl        = "none"
    cfg.model.cuda_graph_scope       = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # ---- MoE-specific kernels -----------------------------------------------
    cfg.model.moe_router_fusion   = False
    cfg.model.moe_permute_fusion  = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.moe_router_padding_for_fp8 = False

    # ---- Memory saving (full recompute – needed for 80GB single GPU) -------
    cfg.model.recompute_granularity         = "full"
    cfg.model.recompute_method              = "uniform"
    cfg.model.recompute_num_layers          = 1
    cfg.model.fine_grained_activation_offloading = False

    # ---- Optimizer precision ------------------------------------------------
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype  = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype     = torch.float32
    cfg.optimizer.exp_avg_sq_dtype  = torch.float32

    # ---- Communication overlap (disable for single GPU) --------------------
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute               = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm  = False

    # ---- DDP ----------------------------------------------------------------
    cfg.ddp.overlap_grad_reduce           = True
    cfg.ddp.overlap_param_gather          = True
    cfg.ddp.check_for_nan_in_grad         = True
    cfg.ddp.use_distributed_optimizer     = True
    cfg.ddp.use_megatron_fsdp             = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # ---- Checkpointing ------------------------------------------------------
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save    = False

    # ---- MoE load balancing -------------------------------------------------
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_aux_loss_coeff              = 1e-3

    return cfg

def param2_17b_8gpu_config(
    hf_path: str = HF_MODEL_PATH,
    mock: bool = True,
    data_paths=None,
    train_data_path=None,
    valid_data_path=None,
    test_data_path=None,
    dir: str = None,
    name: str = "param2_17b_8gpu",
    load_weights: bool = False,
) -> ConfigContainer:
    """
    8-GPU recipe.  Recommended: 8×A100-80G or 8×H100-80G in one node.
    EP=8 shards the 64 experts across 8 GPUs (8 experts/GPU).
    """
    cfg = param2_17b_single_gpu_config(
        hf_path=hf_path, mock=mock,
        data_paths=data_paths,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        test_data_path=test_data_path,
        dir=dir, name=name, load_weights=load_weights,
    )
    mcore_cfg, _ = _build_model_provider(hf_path, tp=1, pp=1, ep=8,
                                         load_weights=load_weights)
    cfg.model = mcore_cfg
    # Re-apply MoE settings
    cfg.model.tensor_model_parallel_size     = 1
    cfg.model.pipeline_model_parallel_size   = 1
    cfg.model.expert_model_parallel_size     = 8
    cfg.model.sequence_parallel              = False
    cfg.model.seq_length                     = 4096
    cfg.model.moe_token_dispatcher_type      = "alltoall"
    cfg.model.moe_flex_dispatcher_backend    = "deepep"  # use DeepEP for EP>1
    cfg.model.moe_hybridep_num_sms           = 16
    cfg.model.moe_grouped_gemm               = True
    cfg.model.moe_shared_expert_overlap      = True
    cfg.model.moe_router_fusion              = False
    cfg.model.moe_permute_fusion             = True
    cfg.model.transformer_impl               = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion      = True
    cfg.model.cross_entropy_fusion_impl      = "te"
    cfg.model.cuda_graph_impl                = "none"
    cfg.model.recompute_granularity          = "full"
    cfg.model.recompute_method               = "uniform"
    cfg.model.recompute_num_layers           = 1

    cfg.train.global_batch_size  = 64
    cfg.train.micro_batch_size   = 1

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True  # EP comms overlap

    cfg.ddp.overlap_grad_reduce        = True
    cfg.ddp.overlap_param_gather       = True
    cfg.ddp.use_distributed_optimizer  = True

    return cfg

def param2_17b_32gpu_config(
    hf_path: str = HF_MODEL_PATH,
    mock: bool = True,
    data_paths=None,
    train_data_path=None,
    valid_data_path=None,
    test_data_path=None,
    dir: str = None,
    name: str = "param2_17b_32gpu",
    load_weights: bool = False,
) -> ConfigContainer:
    """
    32-GPU recipe (4 nodes, 8 GPUs each).

    Parallelism:
      TP=1, PP=2, EP=16
      num_layers=21 → PP split: [11 layers | 10 layers] (manual layout)
      64 experts / 16 EP ranks = 4 experts per GPU (well-distributed)
    """
    cfg = param2_17b_8gpu_config(
        hf_path=hf_path, mock=mock,
        data_paths=data_paths,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        test_data_path=test_data_path,
        dir=dir, name=name, load_weights=load_weights,
    )
    mcore_cfg, _ = _build_model_provider(hf_path, tp=1, pp=2, ep=16,
                                         load_weights=load_weights)
    cfg.model = mcore_cfg

    cfg.model.tensor_model_parallel_size          = 1
    cfg.model.pipeline_model_parallel_size        = 2
    cfg.model.expert_model_parallel_size          = 16
    cfg.model.pipeline_dtype                      = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size               = 1
    cfg.model.sequence_parallel                   = False
    cfg.model.seq_length                          = 4096

    # PP layout: split 21 layers as evenly as possible [11, 10]
    cfg.model.pipeline_model_parallel_layout = [
        ["embedding"] + ["decoder"] * 10,
        ["decoder"] * 10 + ["output_layer"],
    ]

    cfg.model.moe_token_dispatcher_type   = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms        = 20
    cfg.model.moe_grouped_gemm            = True
    cfg.model.moe_shared_expert_overlap   = True
    cfg.model.moe_router_fusion           = False
    cfg.model.moe_permute_fusion          = True
    cfg.model.transformer_impl            = "transformer_engine"
    cfg.model.cross_entropy_loss_fusion   = True
    cfg.model.cross_entropy_fusion_impl   = "te"
    cfg.model.cuda_graph_impl             = "none"
    cfg.model.recompute_granularity       = "full"
    cfg.model.recompute_method            = "uniform"
    cfg.model.recompute_num_layers        = 1

    cfg.train.global_batch_size  = 256
    cfg.train.micro_batch_size   = 1

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.comm_overlap.delay_wgrad_compute               = True

    cfg.ddp.overlap_grad_reduce       = True
    cfg.ddp.overlap_param_gather      = True
    cfg.ddp.use_distributed_optimizer = True

    cfg.checkpoint.save_interval = 1000
    cfg.checkpoint.async_save    = True

    return cfg