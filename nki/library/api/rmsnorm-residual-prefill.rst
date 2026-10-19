.. meta::
    :description: Pre-norm residual add followed by RMSNorm, for prefill-shaped inputs (T >= 128).
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.subkernels

RMSNorm Residual Prefill Kernel API Reference
=============================================

Pre-norm residual add followed by RMSNorm, for prefill-shaped inputs (T >= 128).

When residual is given the add runs in the DMA engines via dma_compute, so the sum is produced during the load; it is also spilled to HBM because it is the next block's residual. Without it the kernel is a plain RMSNorm and returns only out.

Background
-----------

The ``rmsnorm_residual_prefill`` kernel computes a pre-norm residual add followed by RMSNorm for prefill-shaped inputs (T >= 128). The residual add runs on the DMA engines during the load.

API Reference
--------------

**Source code for this kernel API can be found at**: `rmsnorm_residual_prefill.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/subkernels/rmsnorm_residual_prefill.py>`_

rmsnorm_residual_prefill
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: rmsnorm_residual_prefill(hidden: nl.NkiTensor, gamma: nl.NkiTensor, residual: Optional[nl.NkiTensor] = None, eps: float = 1e-06, hidden_actual: Optional[int] = None) -> List[nl.NkiTensor]

   Pre-norm residual add followed by RMSNorm, for prefill-shaped inputs (T >= 128).

   :param hidden: [T, H] or [B, S, H], Hidden states on HBM
   :type hidden: ``nl.NkiTensor``
   :param gamma: [1, H], Gamma tensor used in normalization on HBM
   :type gamma: ``nl.NkiTensor``
   :param residual: [T, H] or [B, S, H], Residual to add before normalizing, on HBM. Omit for a plain RMSNorm
   :type residual: ``Optional[nl.NkiTensor]``
   :param eps: Epsilon to maintain numerical stability. Default is 1e-6
   :type eps: ``float``
   :param hidden_actual: True hidden width when H is padded, used as the mean divisor. Defaults to H
   :type hidden_actual: ``Optional[int]``
   :return: [T, H], Normalized output
   :rtype: ``nl.NkiTensor``
   :return: [T, H], hidden + residual, only when residual is given
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * T: Token count (batch x sequence, flattened)

