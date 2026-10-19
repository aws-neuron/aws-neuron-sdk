.. meta::
    :description: Backward pass of separable 3D max pooling that uses the input and output of 3D max pooling forward and the backward grad
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.pooling

Max Pooling 3D Backward Kernel API Reference
============================================

Backward pass of separable 3D max pooling that uses the input and output of 3D max pooling forward and the backward gradient.

Background
-----------

The ``max_pooling_3d_backward`` kernel computes the backward pass of 3D max pooling from the forward input, the forward output, and the output gradient.

API Reference
--------------

**Source code for this kernel API can be found at**: `max_pooling_3d_backward.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/pooling/max_pooling_3d_backward.py>`_

max_pooling_3d_backward
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: max_pooling_3d_backward(grad_output: nl.NkiTensor, input_tensor: nl.NkiTensor, output_tensor: nl.NkiTensor, kernel_size: Union[int, tuple], stride: Optional[Union[int, tuple]] = None, padding: Union[int, tuple] = 0, data_format: str = 'NCDHW', output_format: Optional[str] = None) -> nl.NkiTensor

   Backward pass of separable 3D max pooling that uses the input and output of 3D max pooling forward and the backward gradient.

   :param grad_output: Gradient w.r.t. the forward output, in HBM, output_format layout, forward-output shape.
   :type grad_output: ``nl.NkiTensor``
   :param input_tensor: The forward input, data_format layout.
   :type input_tensor: ``nl.NkiTensor``
   :param output_tensor: The forward output (per-window maxima), output_format layout, same shape as grad_output.
   :type output_tensor: ``nl.NkiTensor``
   :param kernel_size: Pooling window (kD, kH, kW) (int broadcasts).
   :type kernel_size: ``Union[int, tuple]``
   :param stride: Stride (sD, sH, sW); defaults to kernel_size.
   :type stride: ``Optional[Union[int, tuple]]``
   :param padding: Implicit -inf padding (pD, pH, pW).
   :type padding: ``Union[int, tuple]``
   :param data_format: Layout of input_tensor and grad_input, "NCDHW" or "NDHWC".
   :type data_format: ``str``
   :param output_format: Layout of grad_output / output_tensor; defaults to data_format.
   :type output_format: ``Optional[str]``
   :return: Gradient w.r.t. the forward input, in HBM, data_format layout, same spatial extents as input_tensor.
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * B: batch.

