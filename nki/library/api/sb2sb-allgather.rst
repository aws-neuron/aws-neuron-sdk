.. meta::
    :description: SBUF-to-SBUF all-gather kernel for gathering tensors across ranks.
    :date-modified: 04/09/2026

.. currentmodule:: nkilib.experimental.collectives

SBUF-to-SBUF All-Gather Kernel API Reference
=============================================

Performs SBUF-to-SBUF all-gather for gathering tensors across ranks.

The kernel provides two variants:

* ``allgather_sb2sb`` — Optimized for small tensors that fit entirely in SBUF
* ``allgather_sb2sb_tiled`` — Adds tiling and LNC support for larger tensors

Background
-----------

The ``allgather_sb2sb`` kernels gather input tensors from all ranks along the last dimension (K dimension). Each rank contributes its local tensor, and all ranks receive the concatenated result.

API Reference
--------------

**Source code for this kernel API can be found at**: `sb2sb_allgather.py <https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/experimental/collectives/sb2sb_allgather.py>`_

allgather_sb2sb
^^^^^^^^^^^^^^^

.. py:function:: allgather_sb2sb(inp: nl.ndarray, replica_groups: ReplicaGroup, tp_degree: int) -> nl.ndarray

   SBUF-to-SBUF all-gather kernel for gathering tensors across ranks.

   :param inp: [H, W], Input tensor on HBM, where W is the local width per rank.
   :type inp: ``nl.ndarray``
   :param replica_groups: ReplicaGroup defining which ranks participate in the collective.
   :type replica_groups: ``ReplicaGroup``
   :param tp_degree: Tensor parallelism degree (number of ranks in the group).
   :type tp_degree: ``int``
   :return: [H, K], Output tensor on shared HBM containing gathered data from all ranks.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Input tensor must fit in SBUF (H * W * dtype_size <= SBUF capacity)
   * Output is stored in shared_hbm for cross-rank visibility
   * All ranks receive identical output after the collective

   **Dimensions**:

   * H: Height dimension (partition dimension, typically <= 128)
   * W: Width dimension per rank (local width before gather)

allgather_sb2sb_tiled
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: allgather_sb2sb_tiled(inp: nl.ndarray, replica_groups: ReplicaGroup, tp_degree: int) -> nl.ndarray

   SBUF-to-SBUF all-gather with tiling and LNC support for larger tensors.

   :param inp: [M, K], Input tensor on HBM, where K is the local width per rank.
   :type inp: ``nl.ndarray``
   :param replica_groups: ReplicaGroup defining which ranks participate in the collective.
   :type replica_groups: ``ReplicaGroup``
   :param tp_degree: Tensor parallelism degree (number of ranks in the group).
   :type tp_degree: ``int``
   :return: [M, K * tp_degree], Output tensor on shared HBM containing gathered data.
   :rtype: ``nl.ndarray``

   **Notes**:

   * TILE_M is capped at 128 (SBUF partition size limit)
   * When launched with LNC grid [lnc], tiles are distributed across LNC cores
   * Each LNC core processes TILES_PER_CORE = NUM_M_TILES // n_prgs tiles
   * Assumes M is evenly divisible by 128 when M > 128

   **Dimensions**:

   * M: Height dimension (tiled along this dimension)
   * K: Width dimension per rank (local width before gather)
   * TILE_M: Tile size along M dimension (capped at 128)

