.. meta::
    :description: Backward pass of separable 3D average pooling.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.pooling

Average Pooling 3D Backward Kernel API Reference
================================================

Backward pass of separable 3D average pooling.

Spreads each output gradient back over the input window it was averaged from, applying the 1/prod(kernel_size) scale, using a separable per-axis spread across D, H and W.

Background
-----------

The ``avg_pooling_3d_backward`` kernel computes the backward pass of 3D average pooling. It spreads each output gradient back over its input window with the ``1/prod(kernel_size)`` scale, one axis at a time across D, H and W.

API Reference
--------------

**Source code for this kernel API can be found at**: `avg_pooling_3d_backward.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/pooling/avg_pooling_3d_backward.py>`_

avg_pooling_3d_backward
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: avg_pooling_3d_backward(grad_output: nl.NkiTensor, input_tensor: nl.NkiTensor, kernel_size: Union[int, tuple], stride: Optional[Union[int, tuple]] = None, padding: Union[int, tuple] = 0, data_format: str = 'NCDHW', output_format: Optional[str] = None) -> nl.NkiTensor

   Backward pass of separable 3D average pooling.

   :param grad_output: Gradient w.r.t. the forward output, in HBM, output_format layout, forward-output shape.
   :type grad_output: ``nl.NkiTensor``
   :param input_tensor: The forward input, data_format layout. Only its shape and dtype are read an average pool's gradient does not depend on the forward values.
   :type input_tensor: ``nl.NkiTensor``
   :param kernel_size: Pooling window (kD, kH, kW).
   :type kernel_size: ``Union[int, tuple]``
   :param stride: Stride (sD, sH, sW); defaults to kernel_size.
   :type stride: ``Optional[Union[int, tuple]]``
   :param padding: Implicit zero padding (pD, pH, pW).
   :type padding: ``Union[int, tuple]``
   :param data_format: Layout of input_tensor and grad_input, "NCDHW" or "NDHWC".
   :type data_format: ``str``
   :param output_format: Layout of grad_output; defaults to data_format.
   :type output_format: ``Optional[str]``
   :return: Gradient w.r.t. the forward input, in HBM, data_format layout, same spatial extents as input_tensor.
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * B: Batch size
   * C: Number of channels
   * D: Input depth (spatial)
   * H: Input height (spatial)
   * W: Input width (spatial)
   * Do: Output depth (pooled)
   * Ho: Output height (pooled)

