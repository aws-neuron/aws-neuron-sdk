.. meta::
    :description: Ring Attention Backward SPMD kernel.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.attention

Ring Attention Bwd Kernel API Reference
=======================================

Ring Attention Backward SPMD kernel.

Computes gradients dQ, dK, dV for ring attention using collective permute operations to circulate Q, dY, LSE, and dy_o_sum across workers while keeping K, V local. Supports causal masking and striped attention.

Background
-----------

The ``ring_attention_spmd_bwd`` kernel computes the backward pass for ring attention, producing gradients dQ, dK, and dV using collective permute operations to circulate tensors across workers while keeping K and V local to each worker.

API Reference
--------------

**Source code for this kernel API can be found at**: `ring_attention_bwd.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/attention/ring_attention_bwd.py>`_

ring_attention_spmd_bwd
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: ring_attention_spmd_bwd(q_ref: nl.ndarray, k_ref: nl.ndarray, v_ref: nl.ndarray, o_ref: nl.ndarray, dy_ref: nl.ndarray, lse_ref: nl.ndarray, use_causal_mask: bool = False, mixed_precision: bool = True, softmax_scale: float = None, num_workers: int = 1, lnc_size: int = 1, replica_groups: tuple = None, striped_attention: bool = False)

   Ring Attention Backward SPMD kernel.

   :param q_ref: [B, N, D, S], Query tensor in HBM.
   :type q_ref: ``nl.ndarray``
   :param k_ref: [B, N, D, S], Key tensor in HBM.
   :type k_ref: ``nl.ndarray``
   :param v_ref: [B, N, D, S], Value tensor in HBM.
   :type v_ref: ``nl.ndarray``
   :param o_ref: [B, N, D, S], Forward output tensor in HBM.
   :type o_ref: ``nl.ndarray``
   :param dy_ref: [B, N, D, S], Upstream gradient tensor in HBM.
   :type dy_ref: ``nl.ndarray``
   :param lse_ref: [B, N, 128, S//128], Log-sum-exp from forward pass in HBM.
   :type lse_ref: ``nl.ndarray``
   :param use_causal_mask: Whether to apply causal masking. Default: False.
   :type use_causal_mask: ``bool``
   :param mixed_precision: Whether to use mixed precision (fp32 accumulators). Default: True.
   :type mixed_precision: ``bool``
   :param softmax_scale: Softmax scale factor. Default: 1/sqrt(D).
   :type softmax_scale: ``float``
   :param num_workers: Number of workers in the ring. Default: 1.
   :type num_workers: ``int``
   :param lnc_size: LNC size (number of logical cores). Default: 1.
   :type lnc_size: ``int``
   :param replica_groups: Replica groups for collective communication. Default: None.
   :type replica_groups: ``tuple``
   :param striped_attention: Whether to use striped attention layout. Default: False.
   :type striped_attention: ``bool``
   :return: [B, N, D, S], Query gradient in HBM (float32).
   :rtype: ``nl.ndarray``
   :return: [B, N, D, S], Key gradient in HBM (float32).
   :rtype: ``nl.ndarray``
   :return: [B, N, D, S], Value gradient in HBM (float32).
   :rtype: ``nl.ndarray``

   **Notes**:

   * Sequence length S must be divisible by 128.
   * striped_attention requires use_causal_mask=True.
   * When B is not divisible by lnc_size, the last batch is handled with duplicate work.

   **Dimensions**:

   * B: Batch size
   * N: Number of attention heads
   * D: Head dimension

