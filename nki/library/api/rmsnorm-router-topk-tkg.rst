.. meta::
    :description: Fused RMSNorm (+ optional MX quantize) + Router TopK.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.moe_block

RMSNorm Router Top-K TKG Kernel API Reference
=============================================

Fused RMSNorm (+ optional MX quantize) + Router TopK.

Background
-----------

The ``rmsnorm_router_topk_tkg`` kernel fuses RMSNorm, optional MX (fp8) quantization, and router top-K expert selection for MoE token generation. It returns the (optionally MX-packed) normalized hidden states, top-K expert indices, and masked top-K affinities. Requires LNC=2 sharding.

API Reference
--------------

**Source code for this kernel API can be found at**: `rmsnorm_router_topk_tkg.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe_block/rmsnorm_router_topk_tkg.py>`_

rmsnorm_router_topk_tkg
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: rmsnorm_router_topk_tkg(hidden_states: nl.ndarray, gamma: nl.ndarray, router_weights: nl.ndarray, router_bias: Optional[nl.ndarray] = None, eps: float = 1e-06, top_k: int = 1, hidden_actual: Optional[int] = None, quantization_type: QuantizationType = QuantizationType.NONE, router_mm_dtype = nl.bfloat16, router_act_fn: RouterActFnType = RouterActFnType.SIGMOID)

   Fused RMSNorm (+ optional MX quantize) + Router TopK.

   :param hidden_states: [B, S, H], Input tensor on HBM.
   :type hidden_states: ``nl.ndarray``
   :param gamma: [1, H], RMSNorm weights on HBM.
   :type gamma: ``nl.ndarray``
   :param router_weights: [H, E], Router weights on HBM.
   :type router_weights: ``nl.ndarray``
   :param router_bias: [1, E], Optional router bias on HBM.
   :type router_bias: ``Optional[nl.ndarray]``
   :param eps: Epsilon for RMSNorm. Default 1e-6.
   :type eps: ``float``
   :param top_k: Number of top experts per token. Default 1.
   :type top_k: ``int``
   :param hidden_actual: Actual hidden dim for padded inputs.
   :type hidden_actual: ``Optional[int]``
   :param quantization_type: NONE or MX. Default NONE.
   :type quantization_type: ``QuantizationType``
   :param router_mm_dtype: Dtype for router matmul. Default nl.bfloat16.
   :param router_act_fn: SOFTMAX or SIGMOID. Default SIGMOID.
   :type router_act_fn: ``RouterActFnType``
   :return: [T, H] (NONE) or [T, H + H/4] FP8 packed quant‖scales (MX).
   :rtype: ``nl.ndarray``
   :return: [T, K] int32 top-K indices.
   :rtype: ``nl.ndarray``
   :return: [T, E] bfloat16 masked top-K affinities (zero elsewhere).
   :rtype: ``nl.ndarray``

   **Notes**:

   * Requires LNC=2 sharding.
   * NONE: H must be divisible by 128; T must be a multiple of 256 (DLoC tiling).
   * MX: H must be divisible by 512 (MX block size).

