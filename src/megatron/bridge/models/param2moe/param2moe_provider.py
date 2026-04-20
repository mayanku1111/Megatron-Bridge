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

from megatron.bridge.models.gpt_provider import GPTModelProvider


class Param2MoEModelProvider(GPTModelProvider):
    """
    Model provider for Param2MoE (bharatgenai/Param2-17B-A2.4B-Thinking).


    Key traits vs. base GPTModelProvider:
      - qk_layernorm=True          (per-head RMSNorm on Q and K)
      - tie_word_embeddings=True   (shared embed + lm_head)
      - sigmoid MoE router         (moe_router_score_function="sigmoid")
      - expert bias correction     (moe_router_enable_expert_bias=True)
      - 64 routed + 2 shared experts, top-6 per token
      - vocab_size=128008, rope_theta=1e6
    """

    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False

    # Param2-specific attention settings
    qk_layernorm: bool = True                        # use_qk_norm=True
    share_embeddings_and_output_weights: bool = True  # tie_word_embeddings=True

    # Positional encoding
    position_embedding_type: str = "rope"
    rotary_base: float = 1_000_000.0                 # rope_theta=1e6
    seq_length: int = 4096
    layernorm_epsilon: float = 1e-6

    # Dropout
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    # Vocab
    vocab_size: int = 128000

    # Precision
    bf16: bool = True

    # MoE router
    moe_grouped_gemm: bool = True
    moe_router_pre_softmax: bool = False             # sigmoid does not use pre-softmax
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_shared_expert_overlap: bool = True
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_dtype: str = "fp32"
    moe_permute_fusion: bool = True
    moe_aux_loss_coeff: float = 1e-3