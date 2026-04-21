# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from functools import partial

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.param2moe.param2moe_provider import Param2MoEModelProvider


try:
    import transformer_engine  # noqa: F401
    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


# ── Primary registration: Param2MoEForCausalLM (HF Thinking model) ───────────
# source is a string because this is a trust_remote_code model —
# it cannot be imported directly at module load time.
@MegatronModelBridge.register_bridge(
    source="Param2MoEForCausalLM",
    target=GPTModel,
    provider=Param2MoEModelProvider,
    model_type="param2moe",
)
class Param2MoEBridge(MegatronModelBridge):
    """
    Megatron Bridge for the Param2MoE / BailingMoeV2 architecture family.

    Handles two HuggingFace architecture class names that share the same
    underlying architecture:
      - "Param2MoEForCausalLM"    (bharatgenai/Param2-17B-A2.4B-Thinking)
      - "BailingMoeV2ForCausalLM" (local NeMo pretrain checkpoint)

    Architecture:
      - 21 layers: 1 dense (layer 0) + 20 MoE
      - Standard GQA: 32Q / 8KV heads, head_dim=64  (NOT MLA)
      - 64 routed experts + 2 shared experts, top-6 sigmoid routing
      - QK-Norm per head (q_norm / k_norm)
      - Tied embeddings, vocab_size=128000

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> # Works for both architecture names:
        >>> bridge = AutoBridge.from_hf_pretrained(
        ...     "bharatgenai/Param2-17B-A2.4B-Thinking",
        ...     trust_remote_code=True,
        ... )
        >>> bridge = AutoBridge.from_hf_pretrained(
        ...     "/path/to/local/bailing_moe_checkpoint",
        ...     trust_remote_code=True,
        ... )
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Param2MoEModelProvider:
        """Translate HuggingFace config → Param2MoEModelProvider.

        super().provider_bridge() auto-reads from CONFIG_MAPPING:
          num_layers, hidden_size, ffn_hidden_size, num_attention_heads,
          num_query_groups, kv_channels, rotary_base, vocab_size, layernorm_epsilon, ...
        """
        provider   = super().provider_bridge(hf_pretrained)
        hf_config  = hf_pretrained.config

        # ── Layer spec ────────────────────────────────────────────────────────
        provider.transformer_layer_spec = partial(
            get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE
        )

        # ── Attention ─────────────────────────────────────────────────────────
        provider.normalization    = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear  = False
        provider.add_qkv_bias     = False
        provider.qk_layernorm     = hf_config.use_qk_norm          # True
        provider.share_embeddings_and_output_weights = (
            hf_config.tie_word_embeddings                           # True
        )
        provider.hidden_dropout   = hf_config.embedding_dropout    # 0.0
        provider.attention_dropout = hf_config.attention_dropout   # 0.0
        provider.attention_softmax_in_fp32 = False
        provider.apply_rope_fusion = False

        # ── MoE router ────────────────────────────────────────────────────────
        provider.moe_token_dispatcher_type    = "alltoall"
        provider.moe_router_pre_softmax       = False   # sigmoid: no pre-softmax
        provider.moe_router_score_function    = "sigmoid"
        provider.moe_router_enable_expert_bias = (
            hf_config.moe_router_enable_expert_bias                # True
        )
        provider.moe_router_dtype             = getattr(hf_config, "router_dtype", "fp32")
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_shared_expert_overlap    = True
        provider.moe_grouped_gemm             = True
        provider.moe_permute_fusion           = True

        # ── Fusion flags ──────────────────────────────────────────────────────
        provider.gradient_accumulation_fusion = True
        provider.bias_activation_fusion       = True
        provider.bias_dropout_fusion          = True
        provider.cross_entropy_loss_fusion    = True
        provider.cross_entropy_fusion_impl    = "te"
        provider.masked_softmax_fusion        = True
        provider.persist_layer_norm           = True

        # ── MoE layer frequency ───────────────────────────────────────────────
        # 0 = dense FFN, 1 = MoE FFN
        # Layer 0 is dense, layers 1-20 are MoE (first_k_dense_replace=1)
        first_k_dense = hf_config.first_k_dense_replace            # 1
        num_layers    = hf_config.num_hidden_layers                 # 21
        provider.moe_layer_freq = (
            [0] * first_k_dense + [1] * (num_layers - first_k_dense)
        )

        # ── Shared expert total intermediate size ─────────────────────────────
        # MCore wants total size across all shared experts:
        #   per-expert (4096) × num_shared (2) = 8192
        provider.moe_shared_expert_intermediate_size = (
            hf_config.moe_shared_expert_intermediate_size          # 4096
            * hf_config.num_shared_experts                         # × 2
        )

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: Param2MoEModelProvider) -> dict:
        """Reverse-map Megatron provider fields → HuggingFace config dict."""
        hf_cfg = super(Param2MoEBridge, cls).megatron_to_hf_config(provider)

        # Reconstruct first_k_dense_replace from moe_layer_freq
        moe_layer_freq = getattr(provider, "moe_layer_freq", None)
        if isinstance(moe_layer_freq, list):
            hf_cfg["first_k_dense_replace"] = sum(
                1 for v in moe_layer_freq if v == 0
            )

        # Reconstruct per-expert shared intermediate size
        shared_total = getattr(provider, "moe_shared_expert_intermediate_size", None)
        num_shared   = getattr(provider, "num_shared_experts", None)
        if shared_total and num_shared:
            hf_cfg["moe_shared_expert_intermediate_size"] = shared_total // num_shared

        # Param2-specific fields not in base CONFIG_MAPPING
        hf_cfg["score_function"]           = "sigmoid"
        hf_cfg["norm_topk_prob"]           = True
        hf_cfg["n_group"]                  = 1
        hf_cfg["topk_group"]               = 1
        hf_cfg["num_nextn_predict_layers"] = 0
        hf_cfg["mtp_loss_scaling_factor"]  = 0
        hf_cfg["partial_rotary_factor"]    = 1.0
        hf_cfg["use_qkv_bias"]             = False
        hf_cfg["use_bias"]                 = False
        hf_cfg["use_rmsnorm"]              = True

        return hf_cfg

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Full parameter mapping: Megatron Core names ↔ HuggingFace names.

        This mapping is IDENTICAL for both Param2MoEForCausalLM and
        BailingMoeV2ForCausalLM — they share the same weight key naming
        convention (model.layers.*.self_attn.q_proj.weight, etc.).

        Mapping types used:
          QKVMapping      — fuses q/k/v → linear_qkv (handles GQA interleaving + TP)
          GatedMLPMapping — fuses gate+up → linear_fc1 [G;U] (handles TP splitting)
          AutoMapping     — 1:1, auto-detects ColumnParallel/RowParallel/Replicated
        """
        # ── 1:1 AutoMappings ─────────────────────────────────────────────────
        param_mappings = {
            # Embedding
            "embedding.word_embeddings.weight":
                "model.embed_tokens.weight",
            # Final norm + output head
            "decoder.final_layernorm.weight":
                "model.norm.weight",
            "output_layer.weight":
                "lm_head.weight",
            # Input layernorm (fused into linear_qkv in TE backend)
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight":
                "model.layers.*.input_layernorm.weight",
            # Attention output projection
            "decoder.layers.*.self_attention.linear_proj.weight":
                "model.layers.*.self_attn.o_proj.weight",
            # QK-Norm (Param2/BailingMoE specific: per-head RMSNorm on Q and K)
            "decoder.layers.*.self_attention.q_layernorm.weight":
                "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight":
                "model.layers.*.self_attn.k_norm.weight",
            # Pre-MLP layernorm
            # MoE layers  → pre_mlp_layernorm (separate module in MCore)
            # Dense layer → mlp.linear_fc1.layer_norm_weight (fused in TE)
            # Both map to the same HF key; MCore resolves by module presence.
            "decoder.layers.*.pre_mlp_layernorm.weight":
                "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight":
                "model.layers.*.post_attention_layernorm.weight",
            # Dense FFN down-projection (layer 0 only)
            "decoder.layers.*.mlp.linear_fc2.weight":
                "model.layers.*.mlp.down_proj.weight",
            # MoE router weight
            "decoder.layers.*.mlp.router.weight":
                "model.layers.*.mlp.gate.weight",
            # Expert bias correction (sigmoid router, DeepSeek-V3 style)
            "decoder.layers.*.mlp.router.expert_bias":
                "model.layers.*.mlp.gate.e_score_correction_bias",
            # Routed expert down-projections (* matches expert index for EP)
            "decoder.layers.*.mlp.experts.linear_fc2.weight*":
                "model.layers.*.mlp.experts.*.down_proj.weight",
            # Shared expert down-projection
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight":
                "model.layers.*.mlp.shared_experts.down_proj.weight",
        }

        mapping_list = [
            AutoMapping(megatron_param=mc, hf_param=hf)
            for mc, hf in param_mappings.items()
        ]

        # ── QKV fusion (standard GQA, NOT MLA) ──────────────────────────────
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )

        # ── GatedMLP: [gate; up] → linear_fc1 for SwiGLU ────────────────────
        mapping_list.extend([
            # Dense layer 0
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            # Routed experts (* wildcard = expert index)
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
            ),
            # Shared experts
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                up="model.layers.*.mlp.shared_experts.up_proj.weight",
            ),
        ])

        return MegatronMappingRegistry(*mapping_list)


# ── Secondary registration: BailingMoeV2ForCausalLM ──────────────────────────
# Same class, same bridge logic, different architecture name.
# Applying the decorator as a plain function call registers the SAME class
# under a second name — no subclass, no code duplication.
#
# This handles:
#   - Local NeMo checkpoint: config.json has "BailingMoeV2ForCausalLM"
#   - HF Thinking model:     config.json has "Param2MoEForCausalLM"
# Both are the same architecture, same weight layout, same mapping.
MegatronModelBridge.register_bridge(
    source="BailingMoeV2ForCausalLM",
    target=GPTModel,
    provider=Param2MoEModelProvider,
    model_type="param2moe",
)(Param2MoEBridge)