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


@MegatronModelBridge.register_bridge(
    source="BailingMoeV2ForCausalLM",
    target=GPTModel,
    provider=Param2MoEModelProvider,
    model_type="param2moe",
)
class Param2MoEBridge(MegatronModelBridge):
    """
    Megatron Bridge for Param2MoE (bharatgenai/Param2-17B-A2.4B-Thinking).

    Architecture:
      - 21 layers: 1 dense (layer 0) + 20 MoE
      - Standard GQA: 32Q / 8KV heads, head_dim=64  ← NOT MLA
      - 64 routed experts + 2 shared experts, top-6 sigmoid routing
      - QK-Norm (q_norm / k_norm per head)
      - Tied embeddings (tie_word_embeddings=True)

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained(
        ...     "bharatgenai/Param2-17B-A2.4B-Thinking",
        ...     trust_remote_code=True,
        ... )
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Param2MoEModelProvider:
        """Translate HuggingFace Param2MoEConfig → Param2MoEModelProvider."""
        # super() reads all fields declared in MegatronModelBridge.CONFIG_MAPPING
        # (num_layers, hidden_size, ffn_hidden_size, num_attention_heads,
        #  num_query_groups, kv_channels, rotary_base, vocab_size, …)
        # and populates them automatically from hf_pretrained.config.
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # ── Layer spec ────────────────────────────────────────────────────────
        provider.transformer_layer_spec = partial(
            get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE
        )

        # ── Attention ─────────────────────────────────────────────────────────
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.qk_layernorm = hf_config.use_qk_norm                  # True
        provider.share_embeddings_and_output_weights = (
            hf_config.tie_word_embeddings                               # True
        )
        provider.hidden_dropout = hf_config.embedding_dropout           # 0.0
        provider.attention_dropout = hf_config.attention_dropout        # 0.0
        provider.attention_softmax_in_fp32 = False
        provider.apply_rope_fusion = False

        # ── MoE router ────────────────────────────────────────────────────────
        provider.moe_token_dispatcher_type = "alltoall"
        # sigmoid router: moe_router_pre_softmax MUST be False
        provider.moe_router_pre_softmax = False
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = (
            hf_config.moe_router_enable_expert_bias                     # True
        )
        provider.moe_router_dtype = getattr(hf_config, "router_dtype", "fp32")
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_shared_expert_overlap = True
        provider.moe_grouped_gemm = True
        provider.moe_permute_fusion = True

        # ── Fusion flags ──────────────────────────────────────────────────────
        provider.gradient_accumulation_fusion = True
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.cross_entropy_loss_fusion = True
        provider.cross_entropy_fusion_impl = "te"
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True

        # ── MoE layer frequency ───────────────────────────────────────────────
        # 0 = dense FFN,  1 = MoE FFN
        # Layer 0 is dense (first_k_dense_replace=1), layers 1-20 are MoE.
        first_k_dense = hf_config.first_k_dense_replace                # 1
        num_layers    = hf_config.num_hidden_layers                     # 21
        provider.moe_layer_freq = (
            [0] * first_k_dense + [1] * (num_layers - first_k_dense)
        )

        # ── Shared expert total intermediate size ─────────────────────────────
        # MCore wants the *total* size across all shared experts:
        #   per-expert size (4096) × num_shared_experts (2) = 8192
        provider.moe_shared_expert_intermediate_size = (
            hf_config.moe_shared_expert_intermediate_size               # 4096
            * hf_config.num_shared_experts                             # × 2
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
            hf_cfg["moe_shared_expert_intermediate_size"] = (
                shared_total // num_shared
            )

        # Param2-specific fields not covered by the base CONFIG_MAPPING
        hf_cfg["score_function"]            = "sigmoid"
        hf_cfg["norm_topk_prob"]            = True
        hf_cfg["n_group"]                   = 1
        hf_cfg["topk_group"]                = 1
        hf_cfg["num_nextn_predict_layers"]  = 0
        hf_cfg["mtp_loss_scaling_factor"]   = 0
        hf_cfg["partial_rotary_factor"]     = 1.0
        hf_cfg["use_qkv_bias"]              = False
        hf_cfg["use_bias"]                  = False
        hf_cfg["use_rmsnorm"]               = True

        return hf_cfg

    def mapping_registry(self) -> MegatronMappingRegistry:
        """
        Full parameter mapping: Megatron Core names ↔ HuggingFace Param2MoE names.

        Notes
        -----
        * AutoMapping(megatron_param=..., hf_param=...)
            General 1:1 mapping.  AutoMapping auto-detects whether the weight
            is ColumnParallel, RowParallel, or Replicated from the layer type.

        * QKVMapping(megatron_param=..., q=..., k=..., v=...)
            Fuses the three separate HF projections into MCore's packed
            linear_qkv.weight, handling GQA interleaving and TP splitting.

        * GatedMLPMapping(megatron_param=..., gate=..., up=...)
            Concatenates gate_proj + up_proj → linear_fc1 [G; U] for SwiGLU,
            with correct TP splitting (each rank gets [gate_shard; up_shard]).

        Param2-specific weight names vs. DeepSeek:
          - NO MLA projections (q_a_proj, kv_a_proj, etc.) — use standard GQA
          - QK-Norm: self_attn.q_norm / k_norm → q_layernorm / k_layernorm
          - expert bias: mlp.gate.e_score_correction_bias → mlp.router.expert_bias
        """
        # ── 1:1 AutoMappings ─────────────────────────────────────────────────
        param_mappings = {
            # Embedding
            "embedding.word_embeddings.weight":
                "model.embed_tokens.weight",
            # Final norm + output
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
            # QK-Norm (Param2-specific: per-head RMSNorm on Q and K)
            "decoder.layers.*.self_attention.q_layernorm.weight":
                "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight":
                "model.layers.*.self_attn.k_norm.weight",
            # Pre-MLP layernorm — MoE layers use pre_mlp_layernorm,
            # dense layer uses mlp.linear_fc1.layer_norm_weight.
            # Both map to the same HF key; MCore resolves by module presence.
            "decoder.layers.*.pre_mlp_layernorm.weight":
                "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight":
                "model.layers.*.post_attention_layernorm.weight",
            # Dense FFN output (layer 0 only)
            "decoder.layers.*.mlp.linear_fc2.weight":
                "model.layers.*.mlp.down_proj.weight",
            # MoE router
            "decoder.layers.*.mlp.router.weight":
                "model.layers.*.mlp.gate.weight",
            # Expert bias correction (sigmoid router with bias — DeepSeek-V3 style)
            "decoder.layers.*.mlp.router.expert_bias":
                "model.layers.*.mlp.gate.e_score_correction_bias",
            # Routed expert down-projections
            # The trailing * in the megatron_param matches expert indices for EP
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

        # ── GatedMLP (SwiGLU: [gate; up] → linear_fc1) ──────────────────────
        mapping_list.extend([
            # Dense layer 0: standard FFN gate/up
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            # Routed experts: * wildcard matches the expert index
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
            ),
            # Shared experts (always-active, 2 experts fused into one module)
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                up="model.layers.*.mlp.shared_experts.up_proj.weight",
            ),
        ])

        return MegatronMappingRegistry(*mapping_list)