.. meta::
    :description: RMSNorm-Quant kernel performs optional RMS normalization followed by fp8 quantization.
    :date-modified: 10/28/2025

.. currentmodule:: nkilib.core.rmsnorm_quant.rmsnorm_quant

RMSNorm-Quant Kernel API Reference
==================================

Performs optional RMS normalization followed by quantization to fp8.

The kernel supports:

* Optional RMS normalization before quantization
* 8-bit quantization along the last dimension of the input tensor
* Single program multiple data (SPMD) sharding for distributed computation
* Flexible input tensor shapes (minimum 2 dimensions)
* Input validation with configurable dimension limits
* Lower bound clipping for numerical stability

Background
--------------

The ``RMSNorm-Quant`` kernel processes tensors along their last dimension (processing dimension), with all other dimensions collapsed into a single outer dimension. This design allows for efficient processing of tensors with arbitrary shapes, as long as they have at least 2 dimensions.

For detailed information about the mathematical operations and implementation details, refer to the :doc:`RMSNorm-Quant Kernel Design Specification </nki/library/specs/design-rmsnorm-quant>`.

API Reference
----------------

**Source code for this kernel API can be found at**: `rmsnorm_quant.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/rmsnorm/rmsnorm_quant.py>`_

rmsnorm_quant_kernel
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: rmsnorm_quant_kernel(hidden: nl.ndarray, ln_w: nl.ndarray, kargs: RmsNormQuantKernelArgs, input_dequant_scale: nl.ndarray = None)

   Entrypoint NKI kernel that performs one of the following:
   
   1. Perform RMSNorm and quantize the normalized hidden over the hidden dimension (``H``, or ``axis=-1``).
   2. Quantize hidden over dimension ``H``.

   The kernel supports no specialization, or specialization along 1 dimension (1D SPMD grid).

   :param hidden: Input hidden states tensor with minimum 2 dimensions. For 3D inputs, expected layout is ``[B, S, H]``. For 2D inputs, layout is ``[outer_dim, processing_dim]`` where outer_dim is the product of all major dimensions.
   :type hidden: ``nl.ndarray``
   :param ln_w: Gamma multiplicative bias vector with ``[H]`` or ``[1, H]`` layout. Required when RMS normalization is enabled.
   :type ln_w: ``nl.ndarray``
   :param kargs: Kernel arguments specifying normalization type, bounds, and epsilon values. See :py:class:`RmsNormQuantKernelArgs` for details.
   :type kargs: ``RmsNormQuantKernelArgs``
   :param input_dequant_scale: Optional dequantization scale for input tensor.
   :type input_dequant_scale: ``nl.ndarray``, optional
   :return: Output tensor with shape ``[..., H + 4]`` on HBM where the last dimension is extended by 4 elements. The first H elements store the possibly normalized and quantized tensor, while the last 4 elements store fp8 floats that can be reinterpreted as fp32 dequantization scales.
   :rtype: ``nl.ndarray``

   **Constraints**:

   * Input tensor must have at least 2 dimensions
   * For 3D inputs: batch dimension ≤ MAX_B, sequence length ≤ MAX_S, hidden dimension ≤ MAX_H
   * For 2D inputs: processing dimension ≤ MAX_H, outer dimension ≤ MAX_B × MAX_S
   * When RMS normalization is enabled, ln_w must have shape [H] or [1, H] where H matches the processing dimension

RmsNormQuantKernelArgs
^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: RmsNormQuantKernelArgs

   RMS Norm Quantization Kernel arguments.

   .. py:attribute:: lower_bound
      :type: float

      Non-negative float used for clipping input values and scale.

   .. py:attribute:: norm_type
      :type: NormType
      :value: NormType.RMS_NORM

      Normalization type to use [``RMS_NORM``, ``NO_NORM``]

   .. py:attribute:: quantization_type
      :type: QuantizationType
      :value: QuantizationType.ROW

      Quantization type to use [``ROW``, ``STATIC``]

   .. py:attribute:: eps
      :type: float
      :value: 1e-6

      Epsilon value for numerical stability, model hyperparameter

   .. py:method:: needs_rms_normalization() -> bool

      Returns True if RMS normalization should be applied, False otherwise.

   .. py:method:: has_lower_bound() -> bool

      Returns True if a positive lower bound is specified, False otherwise.

   **Raises**:

   * **AssertionError** – Raised when unsupported normalization types are used, negative bounds are provided, or invalid epsilon values are specified.
   * Supports 1D SPMD grid or no specialization

   .. note::
      The autocast argument may NOT be respected properly. The kernel automatically handles dimension validation and provides detailed error messages for constraint violations.

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **Input Tensor Outer Dimension Collapse**: All major dimensions are collapsed into one for simplification, allowing the kernel to process along the minor dimension efficiently.

2. **Tiling**: The kernel is tiled on the major dimension by a size equal to the hardware's maximum partition dimension, ensuring full utilization of the hardware engines' input width.

3. **SBUF/PSUM Allocation**: Uses Stack Allocator for consistent and deterministic memory allocations within the kernel scope.

4. **SPMD Sharding**: Supports splitting computation across the constituent cores of a Logical Neuron Core by sharding on the outer-most dimension with automatic load balancing for non-divisible dimensions.

5. **Gamma Broadcast**: Improves pipeline parallelism by distributing work to the TensorEngine through matrix multiplication against a vector of ones.

6. **Activation Reduce**: Uses specialized instructions to perform reduce-add operations efficiently along with square operations.

7. **Optimized Batch Processing**: Processes tiles in batches of 8 for improved efficiency, with remainder handling for non-divisible cases.

8. **Input Validation**: Comprehensive validation of tensor dimensions against hardware limits (MAX_B, MAX_S, MAX_H) with detailed error messages.

9. **Numerical Stability**: Implements lower bound clipping and minimum dequantization scale clamping to prevent numerical instabilities.

See Also
-----------

* :doc:`RMSNorm-Quant Kernel Design Specification </nki/library/specs/design-rmsnorm-quant>`
