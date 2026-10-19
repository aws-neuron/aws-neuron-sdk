.. meta::
    :description: Output Projection CTE kernel computes output projection optimized for Context Encoding.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.core.output_projection.output_projection_cte

Output Projection CTE Kernel API Reference
===========================================

Computes output projection (attention @ weight + bias) optimized for Context Encoding (prefill) use cases.

The kernel supports:

* Efficient projection of attention outputs
* Optional bias addition
* LNC sharding for distributed computation
* Optimized memory access patterns
* Head dimension packing for improved performance

Background
--------------

The ``Output Projection CTE`` kernel computes the operation ``out = attention @ weight + bias``, which is commonly used to project the output scores after an attention block in transformer models. This kernel is specifically optimized for Context Encoding (Prefill) use cases, where the sequence length can be large (typically ``S`` ≥ 512).

The kernel employs efficient tiling strategies and memory access patterns to maximize performance on Neuron hardware, with support for sharding across multiple Logical Neuron Cores (LNCs) to handle large hidden dimensions. When ``LNC>1``, the ``H`` dimension is sharded across the cores, which avoids the need for any inter-core collective operations as each core produces part of the output tensor.

API Reference
----------------

**Source code for this kernel API can be found at**: `output_projection_cte.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/core/output_projection/output_projection_cte/output_projection_cte.py>`_

output_projection_cte
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: output_projection_cte(attention: nl.NkiTensor, weight: nl.NkiTensor, bias: Optional[nl.NkiTensor] = None, quantization_type: QuantizationType = QuantizationType.NONE, input_scales: Optional[nl.NkiTensor] = None, weight_scales: Optional[nl.NkiTensor] = None, output_dtype: Optional[type] = None, dtype_mode: DtypeMode = DtypeMode.NON_OCP, compact_weight_scales: bool = False, attention_layout: Optional[OProjAttentionLayout] = None, out: Optional[nl.NkiTensor] = None, out_s_offset: int = 0) -> nl.NkiTensor

   Output Projection Kernel optimized for Context Encoding (Prefill) use cases.

   This kernel computes ``out = attention @ weight + bias``, typically used to project the output scores after an attention block in transformer models.

   This kernel is optimized for Context Encoding (aka Prefill) use cases where sequence length ``S`` is large. Using this kernel with ``S < 512`` may result in degraded performance.

   This kernel uses a layout also used by other Context Encoding kernels to avoid need for transposes.

   :param attention: [B, N, D, S], Input tensor in HBM from attention block.
   :type attention: ``nl.NkiTensor``
   :param weight: [N * D, H], Weight tensor in HBM.
   :type weight: ``nl.NkiTensor``
   :param bias: [1, H], Optional bias tensor in HBM.
   :type bias: ``Optional[nl.NkiTensor]``
   :param quantization_type: Type of quantization (NONE, STATIC for FP8, or MX).
   :type quantization_type: ``QuantizationType``
   :param input_scales: [128, 1], Input scale tensor for FP8 quantization.
   :type input_scales: ``Optional[nl.NkiTensor]``
   :param weight_scales: [128, 1], Weight scale tensor for FP8 quantization.
   :type weight_scales: ``Optional[nl.NkiTensor]``
   :param output_dtype: Output data type. Defaults to attention.dtype for non-MX, or nl.bfloat16 for MX quantization. Can be set to nl.float16 for higher precision.
   :type output_dtype: ``Optional[type]``
   :param dtype_mode: Quantization dtype policy for STATIC/ROW attention and weight-tile allocation. - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240). - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448). TRN3 only. - ``DtypeMode.AUTO``: ``nl.float8_e4m3fn`` on TRN3, else ``nl.float8_e4m3``.
   :type dtype_mode: ``DtypeMode``
   :param compact_weight_scales: When True (and quantization_type is MX), ``weight_scales`` is interpreted as the block-128 compact layout ``[N*D // 128, H // 128]`` uint8, with one scale per 128x128 weight block. The kernel expands the compact scales to the hardware MX layout on-device. Defaults to False (block-32 dense layout).
   :type compact_weight_scales: ``bool``
   :param attention_layout: Dimension order of ``attention``. When declared it is taken as-is, asserted to be supported by ``quantization_type``. Every layout is 4D, so a tag that does not describe the tensor passed in permutes N, D, and S rather than failing; the caller owns getting this right. - ``None`` (default): the layout ``quantization_type`` implies — [B, S, N, D] for ``QuantizationType.ROW``, [B, N, D, S] for every other type. - ``OProjAttentionLayout.BNdS``: ``attention`` is [B, N, D, S]. Supported by every ``quantization_type`` except ROW. - ``OProjAttentionLayout.BSNd``: ``attention`` is [B, S, N, D]. Supported by ``QuantizationType.NONE`` and ``QuantizationType.ROW``. - ``OProjAttentionLayout.BNSd``: ``attention`` is [B, N, S, D] — the attention CTE output with heads folded into batch and ``tp_out=False``. Supported by ``QuantizationType.NONE``.
   :type attention_layout: ``Optional[OProjAttentionLayout]``
   :param out: [B, S_total, H], caller-allocated output. When given, results are written here instead of into a freshly allocated tensor, and the same tensor is returned. Lets a caller project several S chunks into one result without a stitching copy.
   :type out: ``Optional[nl.NkiTensor]``
   :param out_s_offset: Row offset into ``out``'s S dimension for this call's chunk. Only meaningful together with ``out``. Float (non-quantized) path only.
   :type out_s_offset: ``int``
   :return: [B, S, H], Output tensor in HBM.
   :rtype: ``nl.NkiTensor``

   **Data Types**:
     This kernel supports ``nl.float32``, ``nl.float16`` and ``nl.bfloat16`` data types.
     However, for ``nl.float32``, large inputs may not fit in SBUF.

   **Dimensions**:
     * ``B``: Batch size
     * ``N``: Number of heads
     * ``S``: Sequence length
     * ``H``: Hidden dimension size
     * ``D``: Head dimension size

   **Restrictions**:

   * The contract dimension of input and weight tensors must match (``N*D == weight.shape[0]``)
   * Output projection kernel currently only supports ``H`` to be no more than 32768
   * Hidden dimension (``H``) needs to be divisible by LNC size since LNC sharding is on the weight hidden dimension
   * Head dimension (``D``) must be <= 128
   * Maximum validated ``H`` size is 20705
   * Maximum validated ``B*S`` size is 131072
   * Maximum validated ``N`` size is 17

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **Dimension Packing**: Optimizes the contraction dimension by folding ``N`` (number of heads) into ``D`` (head dimension) when beneficial, improving computational efficiency.

2. **Efficient Tiling Strategy**: Uses carefully chosen tile sizes for processing batches and sequences to maximize hardware utilization.

3. **LNC Sharding**: Supports sharding across multiple Logical Neuron Cores (LNCs) by dividing the hidden dimension, enabling processing of larger models.

4. **Memory Access Optimization**: Employs optimized memory access patterns to maximize bandwidth utilization and minimize data movement.

5. **PSUM Bank Utilization**: Efficiently utilizes PSUM banks for accumulating partial results during matrix multiplication operations.

6. **Stream Shuffle Broadcast**: Uses stream shuffle broadcast for bias tensors to efficiently distribute them across processing elements.

7. **Specialized Engine Selection**: Alternates between scalar and vector engines for tensor copy operations to balance workload and improve performance.

See Also
-----------

* :doc:`Output Projection TKG Kernel API Reference </nki/library/api/output-projection-tkg>`
* :doc:`QKV Kernel API Reference </nki/library/api/qkv>`
