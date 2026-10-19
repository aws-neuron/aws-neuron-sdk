.. meta::
    :description: Permute tokens by destination EP rank and dispatch via all-to-all-v.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.collectives.a2av_train

Permute A2AV Kernel API Reference
=================================

Permute tokens by destination EP rank and dispatch via all-to-all-v.

Gathers tokens per destination EP rank into a packed send buffer at ``cumsum(send_counts)`` row offsets, builds the ``(4, EP)`` uint32 metadata tensor, then exchanges via ``ncc.all_to_all_v``. The receive-side layout is determined by the runtime (packed cumsum of recv_counts).

Background
-----------

The ``permute_a2av`` kernel is the MoE training dispatch step: it permutes tokens by their destination expert-parallel (EP) rank and exchanges them across ranks via ``ncc.all_to_all_v``.

API Reference
--------------

**Source code for this kernel API can be found at**: `permute_a2av.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/collectives/a2av_train/permute_a2av.py>`_

permute_a2av
^^^^^^^^^^^^

.. py:function:: permute_a2av(hidden_states: nl.NkiTensor, packed_source_indices: nl.NkiTensor, send_counts: nl.NkiTensor, recv_capacity: int, replica_group: ReplicaGroup) -> tuple[nl.NkiTensor, nl.NkiTensor]

   Permute tokens by destination EP rank and dispatch via all-to-all-v.

   :param hidden_states: [N, H]@HBM routed input rows.
   :type hidden_states: ``nl.NkiTensor``
   :param packed_source_indices: [N]@HBM, int32 permutation of source rows in destination-major order. Every entry must be in [0, N), with peer segments defined by send_counts.
   :type packed_source_indices: ``nl.NkiTensor``
   :param send_counts: [1, EP]@HBM, int32/uint32 row counts.
   :type send_counts: ``nl.NkiTensor``
   :param recv_capacity: Static number of rows in the packed receive output. For top-k routing with unique expert IDs and ``L`` local experts per destination rank, a dropless bound is ``global_token_count * min(top_k, L)``. This reduces to ``global_token_count`` when ``L == 1``. Duplicate expert IDs within one token invalidate this bound.
   :type recv_capacity: ``int``
   :param replica_group: EP replica group.
   :type replica_group: ``ReplicaGroup``
   :return: [C, H]@HBM. Source-major packed rows followed by deterministic zero padding.
   :return: [4, EP]@HBM, uint32. Send counts/displacements and receive counts/displacements, all measured in elements.
   :rtype: ``tuple[nl.NkiTensor, nl.NkiTensor]``

   **Dimensions**:

   * T: number of local tokens (SP sharded).
   * H: hidden dimension.
   * EP: number of expert-parallel ranks (== replica_group size).

