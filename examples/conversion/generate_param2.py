"""
generate_param2.py
------------------
Inference for Param2MoE / BailingMoeV2 via Megatron-Bridge.

THREE loading paths supported:
  1. HF hub model (downloads safetensors on first run)
  2. Local safetensors dir (e.g. /fsxnew/.../989999/)
  3. NeMo distributed checkpoint (17B_PT2_829k_nemo/weights/)

Place this file at:
  examples/conversion/generate_param2.py

Launch commands
---------------
# Path 1/2 — HF or local safetensors, 2 GPUs TP=2:
CUDA_VISIBLE_DEVICES=6,7 \\
PYTHONPATH=/workdir/src:/workdir/3rdparty/Megatron-LM \\
python -m torch.distributed.run --nproc_per_node=2 \\
    examples/conversion/generate_param2.py \\
    --hf_model_path /fsxnew/checkpoints-pretraining/17B_PT2/1M_decay_1_1/989999 \\
    --prompt "What is the BharatGen mission?" \\
    --tp 2

# Path 3 — NeMo distributed checkpoint, 2 GPUs TP=2:
CUDA_VISIBLE_DEVICES=6,7 \\
PYTHONPATH=/workdir/src:/workdir/3rdparty/Megatron-LM \\
python -m torch.distributed.run --nproc_per_node=2 \\
    examples/conversion/generate_param2.py \\
    --hf_model_path /fsxnew/checkpoints-pretraining/17B_PT2/1M_decay_1_1/989999 \\
    --megatron_model_path ~/17B_PT2_829k_nemo/weights \\
    --prompt "What is the BharatGen mission?" \\
    --tp 2

# 8 GPUs, EP=8 (most memory-efficient — each GPU holds 8 experts):
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
PYTHONPATH=/workdir/src:/workdir/3rdparty/Megatron-LM \\
python -m torch.distributed.run --nproc_per_node=8 \\
    examples/conversion/generate_param2.py \\
    --hf_model_path /fsxnew/checkpoints-pretraining/17B_PT2/1M_decay_1_1/989999 \\
    --megatron_model_path ~/17B_PT2_829k_nemo/weights \\
    --prompt "What is the BharatGen mission?" \\
    --tp 1 --ep 8

How NeMo checkpoint loading works
----------------------------------
17B_PT2_829k_nemo/
├── context/          ← NeMo metadata (model_config.yaml, tokenizer)
└── weights/          ← Megatron Core distributed checkpoint
                         (.distcp shards, loadable by bridge.load_megatron_model)

When --megatron_model_path is given:
  - hf_model_path is used ONLY for tokenizer + config
  - weights are loaded from the NeMo weights/ dir directly
  - No conversion needed — NeMo 2.0 weights/ IS Megatron Core format
"""

import argparse
import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

# ── Register both architecture names before AutoBridge is called ──────────────
# This import fires the @register_bridge decorator for:
#   "Param2MoEForCausalLM"    (HF Thinking model)
#   "BailingMoeV2ForCausalLM" (local NeMo checkpoint)
import megatron.bridge.models.param2moe  # noqa: F401

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import (
    disable_mtp_for_inference,
    get_last_rank,
    print_rank_0,
)


# ── Single-batch iterator required by forward_backward_func ──────────────────

class SingleBatchIterator:
    def __init__(self, input_ids, position_ids):
        self.batch    = dict(tokens=input_ids, position_ids=position_ids)
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def text_forward_step(data_iterator, model, **kwargs):
    batch       = next(data_iterator)
    forward_args = {
        "input_ids":    batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }
    return model(**forward_args), lambda x, **kw: x


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(args):
    """
    Load model via Megatron-Bridge.

    Two paths:
      A) No --megatron_model_path  → stream safetensors through bridge
         (works for HF hub OR local safetensors directory)
      B) --megatron_model_path set → load NeMo distributed checkpoint directly
         (hf_model_path used only for config + tokenizer)
    """
    tp  = args.tp
    pp  = args.pp
    ep  = args.ep
    etp = args.etp

    # AutoBridge reads config.json → dispatches to Param2MoEBridge
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model_path,
        trust_remote_code=True,
    )

    if args.megatron_model_path:
        # ── Path B: NeMo distributed checkpoint ──────────────────────────────
        # Build provider without loading weights (we'll load from NeMo ckpt)
        print_rank_0(f"Loading NeMo checkpoint from: {args.megatron_model_path}")
        model_provider = bridge.to_megatron_provider(load_weights=False)
        model_provider.tensor_model_parallel_size   = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size   = ep
        model_provider.expert_tensor_parallel_size  = etp
        model_provider.pipeline_dtype               = torch.bfloat16
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)

        # load_megatron_model loads Megatron Core dist checkpoint directly.
        # The NeMo weights/ dir IS a Megatron Core distributed checkpoint.
        model = bridge.load_megatron_model(
            args.megatron_model_path,
            mp_overrides={
                "tensor_model_parallel_size":   tp,
                "pipeline_model_parallel_size": pp,
                "expert_model_parallel_size":   ep,
                "expert_tensor_parallel_size":  etp,
                "pipeline_dtype":               torch.bfloat16,
            },
            wrap_with_ddp=False,
        )
    else:
        # ── Path A: HF safetensors (hub or local dir) ─────────────────────────
        print_rank_0(f"Loading HF weights from: {args.hf_model_path}")
        model_provider = bridge.to_megatron_provider(load_weights=True)
        model_provider.tensor_model_parallel_size   = tp
        model_provider.pipeline_model_parallel_size = pp
        model_provider.expert_model_parallel_size   = ep
        model_provider.expert_tensor_parallel_size  = etp
        model_provider.pipeline_dtype               = torch.bfloat16
        model_provider.finalize()
        model_provider.initialize_model_parallel(seed=0)
        model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    return model, bridge


# ── Tokenizer loading ─────────────────────────────────────────────────────────

def load_tokenizer(args):
    """
    Load tokenizer. Tries three locations in order:
      1. NeMo context/tokenizer/ (if --megatron_model_path given)
      2. hf_model_path (local safetensors dir or HF hub)
    """
    # Try NeMo context dir first
    if args.megatron_model_path:
        nemo_root = os.path.dirname(args.megatron_model_path.rstrip("/"))
        nemo_tok  = os.path.join(nemo_root, "context", "tokenizer")
        if os.path.isdir(nemo_tok):
            print_rank_0(f"Loading tokenizer from NeMo context: {nemo_tok}")
            try:
                tok = AutoTokenizer.from_pretrained(nemo_tok, trust_remote_code=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                return tok
            except Exception as e:
                print_rank_0(f"  NeMo tokenizer failed ({e}), falling back to hf_model_path")

    # Fall back to hf_model_path
    print_rank_0(f"Loading tokenizer from: {args.hf_model_path}")
    tok = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── Greedy generation ─────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, max_new_tokens: int):
    input_ids     = tokenizer.encode(prompt, return_tensors="pt").cuda()
    position_ids  = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        .unsqueeze(0).expand_as(input_ids)
    )
    generated_ids = input_ids.clone()
    stop_tokens   = [tokenizer.eos_token_id]

    print_rank_0(f"Input length: {input_ids.shape[1]} tokens")

    for step in range(max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Step {step + 1}/{max_new_tokens}")

            fwd_bwd_fn = get_forward_backward_func()
            output     = fwd_bwd_fn(
                forward_step_func=text_forward_step,
                data_iterator=SingleBatchIterator(input_ids, position_ids),
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )

        if isinstance(output, list) and len(output) > 0:
            output = output[0]

        if parallel_state.is_pipeline_last_stage():
            # All-gather across TP ranks to reconstruct full vocab logits
            world_size = parallel_state.get_tensor_model_parallel_world_size()
            gathered   = [torch.zeros_like(output) for _ in range(world_size)]
            dist.all_gather(
                gathered, output,
                group=parallel_state.get_tensor_model_parallel_group(),
            )
            output         = torch.cat(gathered, dim=2)
            next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

            # Debug info for first 5 steps
            if step < 5:
                logits           = output[0, -1, :]
                top5_vals, top5_ids = torch.topk(logits, 5)
                top5_tokens      = [tokenizer.decode([i]) for i in top5_ids]
                print_rank_0(f"  Top-5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                print_rank_0(
                    f"  → '{tokenizer.decode([next_token_ids.item()])}'"
                    f" (id={next_token_ids.item()})"
                )
        else:
            next_token_ids = torch.ones(
                (1, 1), device=generated_ids.device, dtype=generated_ids.dtype
            )

        dist.broadcast(next_token_ids, get_last_rank())

        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        input_ids     = generated_ids
        position_ids  = (
            torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
            .unsqueeze(0).expand_as(input_ids)
        )

        if next_token_ids.item() in stop_tokens:
            print_rank_0("EOS reached.")
            break

    return tokenizer.decode(list(generated_ids[0]), skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print_rank_0("=" * 60)
    print_rank_0("  Param2MoE / BailingMoeV2 Inference via Megatron-Bridge")
    print_rank_0(f"  HF path    : {args.hf_model_path}")
    print_rank_0(f"  NeMo path  : {args.megatron_model_path or '(not set — using HF weights)'}")
    print_rank_0(f"  TP={args.tp}  PP={args.pp}  EP={args.ep}")
    print_rank_0("=" * 60)

    model, bridge = load_model(args)

    # Move to GPU, set eval mode
    model = [m.cuda() for m in model]
    for m in model:
        m.eval()
        disable_mtp_for_inference(m)

    tokenizer = load_tokenizer(args)

    generated = generate(model, tokenizer, args.prompt, args.max_new_tokens)

    print_rank_0("\n" + "=" * 60)
    print_rank_0(f"Prompt    : {args.prompt}")
    print_rank_0(f"Generated : {generated}")
    print_rank_0("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--hf_model_path",
        required=True,
        help=(
            "HF hub model ID or local path to safetensors checkpoint dir. "
            "Always used for config + tokenizer. Also used for weights unless "
            "--megatron_model_path is set."
        ),
    )
    parser.add_argument(
        "--megatron_model_path",
        default=None,
        help=(
            "Path to NeMo distributed checkpoint weights/ directory "
            "(e.g. ~/17B_PT2_829k_nemo/weights). "
            "When set, weights are loaded from here instead of HF safetensors."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="The future of artificial intelligence is",
        help="Input prompt",
    )
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--tp",  type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp",  type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep",  type=int, default=1, help="Expert parallel size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size")

    args = parser.parse_args()
    main(args)

    if dist.is_initialized():
        dist.destroy_process_group()