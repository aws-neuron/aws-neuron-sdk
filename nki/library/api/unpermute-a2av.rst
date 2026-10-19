.. meta::
    :description: All-to-all-v combine and unpermute to original token order.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.collectives.a2av_train

Unpermute A2AV Kernel API Reference
===================================

All-to-all-v combine and unpermute to original token order.

Accepts fixed-stride expert output (source ``s`` at rows ``[s*T, s*T + recv_counts[s])``), re-permutes into a packed send buffer using ``cumsum(recv_counts)`` offsets, builds the ``(4, EP)`` metadata, exchanges via ``ncc.all_to_all_v``, then scatter-adds received rows to original token positions via ``send_indices``.

Background
-----------

The ``unpermute_a2av`` kernel is the MoE training combine step: it exchanges expert output via ``ncc.all_to_all_v`` and unpermutes tokens back to their original order. It supports top-k > 1 because the scatter accumulates across all EP contributions.

API Reference
--------------

**Source code for this kernel API can be found at**: `unpermute_a2av.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/collectives/a2av_train/unpermute_a2av.py>`_

unpermute_a2av
^^^^^^^^^^^^^^

.. py:function:: unpermute_a2av(output: nl.NkiTensor, packed_source_indices: nl.NkiTensor, send_counts: nl.NkiTensor, recv_counts: nl.NkiTensor, replica_group: ReplicaGroup) -> nl.NkiTensor

   All-to-all-v combine and unpermute to original token order.

   :param output: [C, H]@HBM packed expert output.
   :type output: ``nl.NkiTensor``
   :param packed_source_indices: [N]@HBM, int32 permutation mapping packed return rows to original routed-row indices, in the same destination-major order used for dispatch.
   :type packed_source_indices: ``nl.NkiTensor``
   :param send_counts: [1, EP]@HBM original dispatch send counts.
   :type send_counts: ``nl.NkiTensor``
   :param recv_counts: [1, EP]@HBM original dispatch receive counts.
   :type recv_counts: ``nl.NkiTensor``
   :param replica_group: EP replica group.
   :type replica_group: ``ReplicaGroup``
   :return: [N, H]@HBM in original routed-row order.
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * Supports top-k > 1 because the scatter accumulates across all EP contributions.
   * With LNC=2, EP is partitioned across cores and partial results are combined via ``nisa.sendrecv`` before core 0 writes the final output.

   **Dimensions**:

   * T: number of original local tokens (SP sharded).
   * H: hidden dimension.

