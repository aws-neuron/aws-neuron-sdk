.. meta::
    :description: Tensor-wise static FP8 quantization.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.core.quantization

FP8 Quantize Kernel API Reference
==================================

Tensor-wise static FP8 quantization.

Multiplies input by quant_scale (= 1/dequant_scale), clips to [-FP8_MAXVAL, FP8_MAXVAL]. The caller can pre-compute quant_scale via nisa.reciprocal to avoid redundant computation when the same dequant_scale is reused (e.g., gate and up projections share gate_up_in_scale).

Background
-----------

The ``static_quantization`` kernel performs tensor-wise static FP8 quantization by multiplying input values by a quantization scale and clipping to the FP8 representable range.

API Reference
--------------

**Source code for this kernel API can be found at**: `fp8_quantize.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/quantization/fp8_quantize.py>`_

static_quantization
^^^^^^^^^^^^^^^^^^^

.. py:function:: static_quantization(hidden_state, input_dequant_scale, quant_scale = None, dtype = nl.float8_e4m3fn, sbm: Optional[BufferManager] = None, quantized = None)

   Tensor-wise static FP8 quantization.


row_quantization
^^^^^^^^^^^^^^^^

.. py:function:: row_quantization(hidden_state, dtype = nl.float8_e4m3fn, sbm: Optional[BufferManager] = None, output_dtype = None, quantized = None, dequant_scale = None)

   Row-wise dynamic FP8 quantization.


pre_combine_dequant_scales
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pre_combine_dequant_scales(input_dequant_scale, weight_dequant_scale, sbm: Optional[BufferManager] = None)

   Pre-combine input and weight dequant scales: combined = w_dequant * in_dequant.


