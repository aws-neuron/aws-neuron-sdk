.. meta::
    :description: Fine grained all-gather and matrix multiplication (FGCC) kernel for TRN2.
    :date-modified: 04/09/2026

.. currentmodule:: nkilib.experimental.collectives

FGCC (All-Gather + Matmul) Kernel API Reference
=================================================

Performs fused all-gather and matrix multiplication (FGCC) for TRN2.

The kernel supports:

* All-gather on left-hand side tensor across ranks
* Matrix multiplication with column-sharded right-hand side tensor
* Ring-based collective permute overlapped with compute
* Both SBUF and HBM communication paths with automatic selection

Background
-----------

The ``allgather_compute_matmul`` kernel performs all-gather on the left-hand side tensor across ranks, then computes matrix multiplication with a column-sharded right-hand side tensor. Communication is overlapped with compute using ring-based collective permute.

API Reference
--------------

**Source code for this kernel API can be found at**: `fgcc.py <https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/experimental/collectives/fgcc.py>`_

allgather_compute_matmul
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: allgather_compute_matmul(lhs: nl.ndarray, rhs: nl.ndarray, tp_degree: int, num_groups: int, force_hbm_cc: bool = False) -> nl.ndarray

   Fine grained all-gather and matrix multiplication (FGCC) kernel for TRN2.

   :param lhs: [m, K], Left-hand side tensor, row-sharded across ranks.
   :type lhs: ``nl.ndarray``
   :param rhs: [K, N], Right-hand side tensor, column-sharded per rank.
   :type rhs: ``nl.ndarray``
   :param tp_degree: Tensor parallelism degree (number of ranks). Must be even.
   :type tp_degree: ``int``
   :param num_groups: Number of replica groups for collective communication.
   :type num_groups: ``int``
   :param force_hbm_cc: If True, force HBM collective communication path even when SBUF path is feasible.
   :type force_hbm_cc: ``bool``
   :return: [RANK_N, ...], Column-sharded result tensor in shared HBM. Shape depends on communication path (SBUF vs HBM).
   :rtype: ``nl.ndarray``

   **Notes**:

   * tp_degree must be even.
   * lhs and rhs must have matching K dimension.
   * M must be divisible by (RANK_N * LNC_N * CHANNEL_N).
   * Platform target is TRN2 only.

   **Dimensions**:

   * m: Local rows per rank (before all-gather).
   * M: Total rows after all-gather (m * tp_degree).
   * K: Shared (contraction) dimension.

