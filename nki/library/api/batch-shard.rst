.. meta::
    :description: QKV batch shard kernel: all_gather on dim 0 + reshape to separate gqa_group_size heads + slice batch.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.collectives

QKV Batch Shard Kernel API Reference
====================================

QKV batch shard kernel: all_gather on dim 0 + reshape to separate gqa_group_size heads + slice batch.

Implements the Q projection transition from TP64 to TP8DP8 for batch sharding attention. Each rank starts with n Q heads for all B batches, ends with gqa_group_size*n Q heads for B/gqa_group_size batches. The pattern works when either: (1) heads are in dim=0 (NBSd), or (2) n=1 (any layout). It performs all_gather on dim 0, reshapes to ``(G, dim0, ...)``, then slices the batch (where G = gqa_group_size)::

   Input (per rank)         all_gather dim 0       reshape                 slice batch             reshape back
   (n, B, S, d)        ->   (G*n, B, S, d)    ->   (G, n, B, S, d)    ->   (G, n, B/G, S, d)  ->   (G*n, B/G, S, d)
   (d, B, n=1, S)      ->   (G*d, B, n=1, S)  ->   (G, d, B, n=1, S)  ->   (G, d, B/G, n=1, S)->   (n=G, d, B/G, S)

   Example with G=8, n=1, B=32, S=1, d=64:
   NBSd: (1,32,1,64) -> (8,32,1,64) -> (8,1,32,1,64) -> (8,1,4,1,64) -> (8,4,1,64)
   dBnS: (64,32,1,1) -> (512,32,1,1) -> (8,64,32,1,1) -> (8,64,4,1,1) -> (8,64,4,1)

Background
-----------

The ``attn_q_batch_shard`` kernel implements the Q projection transition from a tensor-parallel layout (e.g. TP64) to a tensor-and-data-parallel layout (e.g. TP8DP8) for batch-sharded attention. Each rank starts with ``n`` Q heads for all ``B`` batches and ends with ``gqa_group_size * n`` Q heads for ``B / gqa_group_size`` batches.

API Reference
--------------

**Source code for this kernel API can be found at**: `batch_shard.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/collectives/batch_shard.py>`_

attn_q_batch_shard
^^^^^^^^^^^^^^^^^^

.. py:function:: attn_q_batch_shard(input: nl.ndarray, iota_workers: nl.ndarray, gathered_buf: nl.ndarray, gqa_group_size: int, replica_group: ReplicaGroup, layout: AttnQBatchShardLayout = AttnQBatchShardLayout.NBSd, rank_id_in: Optional[nl.ndarray] = None) -> nl.ndarray

   QKV batch shard kernel: all_gather on dim 0 + reshape to separate gqa_group_size heads + slice batch.

   :param input: Input Q tensor from TP64 projection. First dim is gathered across gqa_group_size ranks. NBSd: (n_heads, B, S, d) dBnS: (d, B, n_heads, S)
   :type input: ``nl.ndarray``
   :param iota_workers: Lookup table mapping rank_id -> batch_offset for scalar_offset DMA. Shape: (1, collective_ranks), values: [(r % gqa_group_size) * B_per_rank for r in range(collective_ranks)] Needed because NKI compiler doesn't support arithmetic on rank_id.
   :type iota_workers: ``nl.ndarray``
   :param gathered_buf: Workspace buffer for all_gather result (must be input tensor for scalar_offset)
   :type gathered_buf: ``nl.ndarray``
   :param gqa_group_size: GQA group size (e.g., 8 for TP8DP8, 2 for TP2DP2)
   :type gqa_group_size: ``int``
   :param replica_group: ReplicaGroup defining the collective topology
   :type replica_group: ``ReplicaGroup``
   :param layout: Output layout - NBSd or dBnS
   :type layout: ``AttnQBatchShardLayout``
   :param rank_id_in: Optional rank_id as input tensor (1,1) int32. If None, uses ncc.rank_id().
   :type rank_id_in: ``Optional[nl.ndarray]``
   :return: (gqa_group_size*n_heads, B/gqa_group_size, S, d)
   :rtype: ``nl.ndarray``
   :return: (n=gqa_group_size, d, B/gqa_group_size, S)
   :rtype: ``nl.ndarray``

