.. meta::
    :description: Compute first-order linear recurrence along the last dimension.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.scan

Linear Scan Kernel API Reference
================================

Compute first-order linear recurrence along the last dimension.

This kernel computes result[t] = decay[t] * result[t-1] + data[t] along the last dimension of the input tensors using nisa.tensor_tensor_scan. Supports arbitrary batch dimensions which are collapsed internally.

Background
-----------

The ``linear_scan`` kernel computes a first-order linear recurrence along the last dimension, where each output element is the sum of the current input and the product of a decay coefficient with the previous output.

API Reference
--------------

**Source code for this kernel API can be found at**: `linear_scan.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/scan/linear_scan.py>`_

linear_scan
^^^^^^^^^^^

.. py:function:: linear_scan(decay: nl.ndarray, data: nl.ndarray, initial: nl.ndarray = None) -> tuple

   Compute first-order linear recurrence along the last dimension.

   :param decay: Input HBM tensor of shape (..., P, L) containing multiplicative decay coefficients. dtype can be any NKI-supported type.
   :type decay: ``nl.ndarray``
   :param data: Input HBM tensor of shape (..., P, L) containing additive input values. Must have same shape as decay.
   :type data: ``nl.ndarray``
   :param initial: Initial state tensor of shape (..., P, 1). If None, initial state is zero. Default: None.
   :type initial: ``nl.ndarray``
   :return: (result, final_state) - result: HBM tensor with same shape as inputs, containing the scan output. - final_state: HBM tensor of shape (..., P, 1) containing the last state for each sequence, in float32.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Only supports scan along the last dimension
   * Uses float32 accumulation internally for numerical stability
   * decay and data must have identical shapes and rank >= 2
   * For long sequences (>2048), the scan is tiled with carry propagation

   **Dimensions**:

   * P: Partition dimension (second-to-last), tiled at P_MAX=128
   * L: Free dimension (last), tiled at F_TILE_SIZE=2048

