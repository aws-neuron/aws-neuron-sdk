.. meta::
    :description: DeepSeek V3.2 MLP kernel (MX prequantized packed block-scale input).
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.deepseekv32_mlp

MLP DeepSeek MX Kernel API Reference
====================================

DeepSeek V3.2 MLP kernel (MX prequantized packed block-scale input).

Specialized for DeepSeek V3.2 shared-experts and first dense MLP layers. Supports MX (MXFP8) quantization only, with MX-prequantized packed block-scale hidden input; no norm / bias / fused-add. Intended for prefill: large-T small-I shapes (e.g. T=8192, H=7168, I=512) hoist all weights and token-shard across cores, while small-T large-I shared-experts shapes (e.g. T<=512, H=7168, I=2048) stream weights tiled and I-shard across cores.

Background
-----------

The ``mlp_deepseek_mx`` kernel implements the DeepSeek-V3.2 shared-experts and first dense MLP layers for prefill with MXFP8 weights and MX-prequantized packed block-scale input, as produced by ``rmsnorm_mx_prefill`` with ``pack_scales=True``.

API Reference
--------------

**Source code for this kernel API can be found at**: `mlp_deepseek_mx.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/deepseekv32_mlp/mlp_deepseek_mx.py>`_

mlp_deepseek_mx
^^^^^^^^^^^^^^^

.. py:function:: mlp_deepseek_mx(hidden_tensor: nl.NkiTensor, gate_proj_weights_tensor: nl.NkiTensor, up_proj_weights_tensor: nl.NkiTensor, down_proj_weights_tensor: nl.NkiTensor, gate_w_scale: nl.NkiTensor, up_w_scale: nl.NkiTensor, down_w_scale: nl.NkiTensor, activation_fn: ActFnType = ActFnType.SiLU, output_dtype = None, routed_expert_output: Optional[nl.NkiTensor] = None, sbm: Optional[SbufManager] = None, compact_scales: bool = False) -> list

   DeepSeek V3.2 MLP kernel (MX prequantized packed block-scale input).

   :param hidden_tensor: [B, S, H + scale_region] fp8, MX-prequantized packed hidden states where ``scale_region = ceil(H/512/4)*128`` uint8 block-scale bytes follow the H fp8 bytes.
   :type hidden_tensor: ``nl.NkiTensor``
   :param gate_proj_weights_tensor: [128, H/512, I/512, 4, 128, 4] fp8 gate weight (MX ``H_X4_INNERMOST`` layout).
   :type gate_proj_weights_tensor: ``nl.NkiTensor``
   :param up_proj_weights_tensor: [128, H/512, I/512, 4, 128, 4] fp8 up weight.
   :type up_proj_weights_tensor: ``nl.NkiTensor``
   :param down_proj_weights_tensor: [128, I/512, H, 4] fp8 down weight.
   :type down_proj_weights_tensor: ``nl.NkiTensor``
   :param gate_w_scale: MX gate weight scales. Native block-32 layout [16, H/512, I/512, 4, 128] when ``compact_scales`` is False; compact block-128 layout [H/128, I/128] when True.
   :type gate_w_scale: ``nl.NkiTensor``
   :param up_w_scale: MX up weight scales (same layouts as ``gate_w_scale``).
   :type up_w_scale: ``nl.NkiTensor``
   :param down_w_scale: MX down weight scales. Native block-32 [16, I/512, H] when ``compact_scales`` is False; compact block-128 [I/128, H/128] when True.
   :type down_w_scale: ``nl.NkiTensor``
   :param activation_fn: Gate activation (default SiLU).
   :type activation_fn: ``ActFnType``
   :param output_dtype: Output dtype (default bf16).
   :param routed_expert_output: [B, S, H] routed-expert output to add to this kernel's result. When provided (shared-experts mode), the kernel stores ``mlp_result + routed_expert_output`` (summed in fp32 inside the store DMA). When None (plain MLP mode), the kernel's own result is stored unchanged.
   :type routed_expert_output: ``Optional[nl.NkiTensor]``
   :param sbm: Optional SbufManager for the shared HBM output allocation.
   :type sbm: ``Optional[SbufManager]``
   :param compact_scales: When True, gate/up/down weight scales are DeepSeek compact block-128 (one uint8 per 128x128 weight block) and are expanded + swizzled to the native block-32 layout in-kernel. When False (default), scales are already the native block-32 layout.
   :type compact_scales: ``bool``
   :return: [B, S, H] bf16, returned as a single-element list. When ``routed_expert_output`` is given, this is the elementwise sum of the two.
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * B: Batch size
   * S: Sequence length (T = B * S total tokens)
   * H: Hidden dimension size (multiple of 512)

