.. meta::
    :description: Permute tokens by destination EP rank and dispatch via all-to-all-v.
    :date-modified: 06/11/2026

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

**Source code for this kernel API can be found at**: `permute_a2av.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/collectives/a2av_train/permute_a2av.py>`_

permute_a2av
^^^^^^^^^^^^

.. py:function:: permute_a2av(hidden_states: nl.ndarray, send_indices: nl.ndarray, send_counts: nl.ndarray, replica_group: ReplicaGroup) -> tuple[nl.ndarray, nl.ndarray]

   Permute tokens by destination EP rank and dispatch via all-to-all-v.

   :param hidden_states: [T, H]@HBM. Input tokens (bf16/fp16/fp32).
   :type hidden_states: ``nl.ndarray``
   :param send_indices: [T, EP]@HBM, int32. Source-row indices per destination rank. Entries with value = T are skipped (via ``oob_mode.skip``), so slots beyond ``send_counts[d]`` for destination ``d`` are no-ops.
   :type send_indices: ``nl.ndarray``
   :param send_counts: [1, EP]@HBM, int32/uint32. Tokens sent per dest rank.
   :type send_counts: ``nl.ndarray``
   :param replica_group: EP replica group.
   :type replica_group: ``ReplicaGroup``
   :return: [EP*T, H]@HBM. Received tokens packed by the runtime at ``cumsum(recv_counts)`` offsets.
   :rtype: ``nl.ndarray``
   :return: [4, EP]@HBM, uint32. After the collective, row 2 contains the runtime-populated ``recv_counts * H``.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * T: number of local tokens (SP sharded).
   * H: hidden dimension.
   * EP: number of expert-parallel ranks (== replica_group size).

