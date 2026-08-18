.. meta::
    :description: Fine-grained ring-based all-gather kernel for TRN2.
    :date-modified: 04/09/2026

.. currentmodule:: nkilib.experimental.collectives

Fine-Grained All-Gather Kernel API Reference
=============================================

Performs fine-grained ring-based all-gather across ranks for TRN2.

The kernel supports:

* Ring-based collective permute with double buffering
* Both SBUF and HBM communication paths with automatic selection based on tensor sizes
* Overlapped communication and data movement

Background
-----------

The ``fine_grained_allgather`` kernel performs all-gather on the input tensor across ranks along the row dimension. It uses ring-based collective permute with double buffering to overlap communication and data movement.

API Reference
--------------

**Source code for this kernel API can be found at**: `fg_allgather.py <https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/experimental/collectives/fg_allgather.py>`_

fine_grained_allgather
^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: fine_grained_allgather(lhs: nl.ndarray, tp_degree: int, num_groups: int, force_hbm_cc: bool = False) -> nl.ndarray

   Fine-grained ring-based all-gather kernel for TRN2.

   :param lhs: [m, K], Input tensor, row-sharded across ranks.
   :type lhs: ``nl.ndarray``
   :param tp_degree: Tensor parallelism degree (number of ranks). Must be even. Supported values: 4, 8, 16, 32, 64, 128.
   :type tp_degree: ``int``
   :param num_groups: Number of replica groups for collective communication.
   :type num_groups: ``int``
   :param force_hbm_cc: If True, force HBM collective communication path even when SBUF path is feasible.
   :type force_hbm_cc: ``bool``
   :return: [RANK_N, ...], Fully gathered tensor in shared HBM. Shape depends on communication path (SBUF vs HBM).
   :rtype: ``nl.ndarray``

   **Notes**:

   * tp_degree must be even.
   * M must be divisible by (RANK_N * LNC_N * CHANNEL_N).
   * Platform target is TRN2 only.

   **Dimensions**:

   * m: Local rows per rank (before all-gather).
   * M: Total rows after all-gather (m * tp_degree).

