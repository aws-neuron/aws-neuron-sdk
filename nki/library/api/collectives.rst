.. meta::
    :description: Sum tensors across all ranks.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.collectives

Collective Communication Kernels API Reference
==============================================

HBM-based collective communication kernels for cross-rank data exchange.

Example with replica_group=[[0,1]], input shape (2, 3) -> output shape (2, 3) for ``all_reduce_hbm_kernel``::

   rank0: [[1,2,3], [4,5,6]] -> [[2,4,6], [8,10,12]]
   rank1: [[1,2,3], [4,5,6]] -> [[2,4,6], [8,10,12]]

Background
-----------

This module provides a suite of HBM-based collective communication kernels for exchanging data across ranks: ``all_reduce_hbm_kernel`` (sum tensors across all ranks), ``all_gather_hbm_kernel`` (gather tensors from all ranks along dim 0), ``reduce_scatter_hbm_kernel`` (sum then scatter chunks along dim 0), ``all_to_all_hbm_kernel`` (exchange chunks across ranks along dim 0), and the ``rank_id_kernel`` / ``dma_copy_rank_id_kernel`` helpers for per-rank slice selection using ``rank_id`` as a scalar offset.

API Reference
--------------

**Source code for this kernel API can be found at**: `collectives.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/collectives/collectives.py>`_

all_reduce_hbm_kernel
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: all_reduce_hbm_kernel(input: nl.ndarray, replica_group: ReplicaGroup) -> nl.ndarray

   Sum tensors across all ranks.


all_gather_hbm_kernel
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: all_gather_hbm_kernel(input: nl.ndarray, replica_group: ReplicaGroup, num_ranks: int) -> nl.ndarray

   Gather tensors from all ranks along dim 0.


reduce_scatter_hbm_kernel
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: reduce_scatter_hbm_kernel(input: nl.ndarray, replica_group: ReplicaGroup, num_ranks: int) -> nl.ndarray

   Sum then scatter chunks along dim 0. Dim 0 is split into num_ranks chunks.


all_to_all_hbm_kernel
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: all_to_all_hbm_kernel(input: nl.ndarray, replica_group: ReplicaGroup) -> nl.ndarray

   Exchange chunks across ranks along dim 0. Each rank sends input[i,:] to rank[i].


rank_id_kernel
^^^^^^^^^^^^^^

.. py:function:: rank_id_kernel(in_tensor: nl.ndarray) -> nl.ndarray

   Select per-rank slice using rank_id as scalar_offset.


dma_copy_rank_id_kernel
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: dma_copy_rank_id_kernel(in_tensor: nl.ndarray, rank_id_lookup: nl.ndarray) -> nl.ndarray

   Load rank_id into SBUF via lookup table, then use as scalar_offset.


