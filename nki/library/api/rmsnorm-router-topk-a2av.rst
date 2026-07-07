.. meta::
    :description: Fused RMSNorm + Router TopK for MoE token generation (small T).
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.moe_block

RMSNorm Router Top-K A2AV Kernel API Reference
==============================================

Fused RMSNorm + Router TopK for MoE token generation (small T).

Both NCs duplicate RMSNorm + Router (no token sharding on compute). HBM norm_output store is sharded across NCs for bandwidth.

Background
-----------

The ``rmsnorm_router_topk_a2av`` kernel fuses RMSNorm and router top-K expert selection for MoE token generation with small token counts (T = B*S <= 128), producing the normalized hidden states, top-K expert indices, and masked affinities needed for the all-to-all-v MoE dispatch path.

API Reference
--------------

**Source code for this kernel API can be found at**: `rmsnorm_router_topk_a2av.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe_block/rmsnorm_router_topk_a2av.py>`_

rmsnorm_router_topk_a2av
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: rmsnorm_router_topk_a2av(hidden_states: nl.ndarray, gamma: nl.ndarray, router_weights: nl.ndarray, router_bias: Optional[nl.ndarray] = None, eps: float = 1e-06, top_k: int = 1, router_act_fn: RouterActFnType = RouterActFnType.SIGMOID)

   Fused RMSNorm + Router TopK for MoE token generation (small T).

   :param hidden_states: [B, S, H]@HBM, bf16/fp16 input.
   :type hidden_states: ``nl.ndarray``
   :param gamma: [1, H]@HBM, bf16/fp16 RMSNorm scale weights.
   :type gamma: ``nl.ndarray``
   :param router_weights: [H, E]@HBM, bf16/fp16 router projection.
   :type router_weights: ``nl.ndarray``
   :param router_bias: [1, E]@HBM, optional router bias.
   :type router_bias: ``Optional[nl.ndarray]``
   :param eps: RMSNorm epsilon for numerical stability.
   :type eps: ``float``
   :param top_k: Number of top experts to select per token.
   :type top_k: ``int``
   :param router_act_fn: Activation for router (SOFTMAX or SIGMOID).
   :type router_act_fn: ``RouterActFnType``
   :return: [T, H]@HBM, normalized hidden states.
   :rtype: ``nl.ndarray``
   :return: [T, K]@HBM, int32 top-K expert indices.
   :rtype: ``nl.ndarray``
   :return: [T, E]@HBM, bf16 masked affinities.
   :rtype: ``nl.ndarray``

   **Notes**:

   * T = B * S must be <= 128 (single tile processing)
   * H must be divisible by 128 (pmax)
   * Both NCs compute identical results; HBM store is sharded for bandwidth
   * Router uses ACT2 pipeline (router_pre_norm=False): top-K on raw logits, then activation applied only to selected experts

   **Dimensions**:

   * B: Batch size (typically 1 for TKG)
   * S: Sequence length (tokens per batch, S <= 128)
   * H: Hidden dimension
   * E: Number of experts

