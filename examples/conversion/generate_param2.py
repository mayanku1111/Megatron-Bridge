import argparse

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

# ── This import triggers @register_bridge for BailingMoeV2ForCausalLM ────────
# If you get "No bridge registered", this import is the fix.
import megatron.bridge.models.param2moe  # noqa: F401  — registers Param2MoEBridge

from megatron.bridge import AutoBridge
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import (
    disable_mtp_for_inference,
    get_last_rank,
    print_rank_0,
)


# ── Data iterator for a single generation step ───────────────────────────────

class SingleBatchIterator:
    """Yields exactly one batch then stops. Required by forward_backward_func."""

    def __init__(self, input_ids, position_ids):
        self.batch = dict(tokens=input_ids, position_ids=position_ids)
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def text_forward_step(data_iterator, model, **kwargs) -> torch.Tensor:
    """Forward step for greedy generation. No loss computation."""
    batch = next(data_iterator)
    forward_args = {
        "input_ids":     batch["tokens"],
        "position_ids":  batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    tp  = args.tp
    pp  = args.pp
    ep  = args.ep
    etp = args.etp

    hf_path = args.hf_model_path

    print_rank_0(f"Checkpoint path : {hf_path}")
    print_rank_0(f"Parallelism     : TP={tp}  PP={pp}  EP={ep}  ETP={etp}")
    print_rank_0(f"Prompt          : {args.prompt!r}")

    # ── Build bridge from local checkpoint ──────────────────────────────────
    # AutoBridge reads config.json → sees "BailingMoeV2ForCausalLM"
    # → dispatches to Param2MoEBridge (registered above)
    print_rank_0("Building bridge from checkpoint …")
    bridge = AutoBridge.from_hf_pretrained(
        hf_path,
        trust_remote_code=True,   # required for custom modeling files
    )

    # ── Build model provider ─────────────────────────────────────────────────
    # load_weights=True streams safetensors weights through the mapping_registry
    # (q_proj+k_proj+v_proj → linear_qkv, gate+up → linear_fc1, etc.)
    print_rank_0("Building Megatron provider and loading weights …")
    model_provider = bridge.to_megatron_provider(load_weights=True)
    model_provider.tensor_model_parallel_size   = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size   = ep
    model_provider.expert_tensor_parallel_size  = etp
    model_provider.pipeline_dtype               = torch.bfloat16

    # finalize() runs post-init validation and derives dependent fields
    model_provider.finalize()
    # initialize_model_parallel() sets up NCCL process groups for TP/PP/EP
    model_provider.initialize_model_parallel(seed=0)

    # provide_distributed_model() instantiates GPTModel sharded across ranks
    model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    # Move to GPU and set eval mode
    model = [m.cuda() for m in model]
    for m in model:
        m.eval()
        disable_mtp_for_inference(m)   # disables MTP heads (Param2 has none, harmless)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    # Loaded from the same checkpoint dir (tokenizer.json is there)
    print_rank_0("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Tokenise prompt ──────────────────────────────────────────────────────
    input_ids  = tokenizer.encode(args.prompt, return_tensors="pt").cuda()
    position_ids = (
        torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        .unsqueeze(0).expand_as(input_ids)
    )
    generated_ids = input_ids.clone()
    stop_tokens   = [tokenizer.eos_token_id]

    print_rank_0(f"Input tokens: {input_ids.shape[1]}")

    # ── Greedy generation loop ────────────────────────────────────────────────
    for step in range(args.max_new_tokens):
        with torch.no_grad():
            print_rank_0(f"Generation step {step + 1}/{args.max_new_tokens}")

            fwd_bwd_fn = get_forward_backward_func()
            iterator   = SingleBatchIterator(input_ids, position_ids)

            output = fwd_bwd_fn(
                forward_step_func=text_forward_step,
                data_iterator=iterator,
                model=model,
                num_microbatches=1,
                forward_only=True,
                seq_length=input_ids.size(1),
                micro_batch_size=1,
                collect_non_loss_data=True,
            )

            if isinstance(output, list) and len(output) > 0:
                output = output[0]

            # All-gather across TP ranks on the last PP stage
            if parallel_state.is_pipeline_last_stage():
                world_size     = parallel_state.get_tensor_model_parallel_world_size()
                gathered       = [torch.zeros_like(output) for _ in range(world_size)]
                dist.all_gather(
                    gathered, output,
                    group=parallel_state.get_tensor_model_parallel_group(),
                )
                output = torch.cat(gathered, dim=2)

                next_token_ids = torch.argmax(output[:, -1], dim=-1, keepdim=True)

                # Debug info for first 5 steps
                if step < 5:
                    print_rank_0(f"  output shape={output.shape}, var={output.var():.4f}")
                    logits = output[0, -1, :]
                    top5_vals, top5_ids = torch.topk(logits, 5)
                    top5_tokens = [tokenizer.decode([i]) for i in top5_ids]
                    print_rank_0(f"  Top-5: {list(zip(top5_tokens, top5_vals.tolist()))}")
                    print_rank_0(
                        f"  Selected: '{tokenizer.decode([next_token_ids.item()])}'"
                        f" (id={next_token_ids.item()})"
                    )
            else:
                next_token_ids = torch.ones(
                    (1, 1), device=generated_ids.device, dtype=generated_ids.dtype
                )

            # Broadcast selected token from last rank to all ranks
            dist.broadcast(next_token_ids, get_last_rank())

        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        input_ids     = generated_ids
        position_ids  = (
            torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
            .unsqueeze(0).expand_as(input_ids)
        )

        if next_token_ids.item() in stop_tokens:
            print_rank_0("EOS token generated — stopping.")
            break

    # ── Decode and print ─────────────────────────────────────────────────────
    generated_text = tokenizer.decode(list(generated_ids[0]), skip_special_tokens=True)
    print_rank_0("\n" + "=" * 50)
    print_rank_0(f"Prompt    : {args.prompt}")
    print_rank_0(f"Generated : {generated_text}")
    print_rank_0("=" * 50)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference with BailingMoeV2 / Param2-17B via Megatron-Bridge"
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Local path to checkpoint dir (contains config.json + safetensors)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum new tokens to generate",
    )
    parser.add_argument("--tp",  type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp",  type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--ep",  type=int, default=1, help="Expert parallel size")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size")

    args = parser.parse_args()
    main(args)

    if dist.is_initialized():
        dist.destroy_process_group()