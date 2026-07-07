.. meta::
    :description: All-to-all-v combine and unpermute to original token order.
    :date-modified: 06/11/2026

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

**Source code for this kernel API can be found at**: `unpermute_a2av.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/collectives/a2av_train/unpermute_a2av.py>`_

unpermute_a2av
^^^^^^^^^^^^^^

.. py:function:: unpermute_a2av(output: nl.ndarray, send_indices: nl.ndarray, recv_counts: nl.ndarray, replica_group: ReplicaGroup) -> nl.ndarray

   All-to-all-v combine and unpermute to original token order.

   :param output: [EP*T, H]@HBM. Expert output in fixed-stride layout: source ``s`` contributes at rows ``[s*T, s*T + recv_counts[s])``.
   :type output: ``nl.ndarray``
   :param send_indices: [T, EP]@HBM, int32. MoE routing table shared with dispatch: ``send_indices[t, d]`` is the original local-token row that source rank ``d`` produces the ``t``-th contribution for. On combine we scatter-add the received rows into those original positions. Entries with value = T are skipped (via ``oob_mode.skip``).
   :type send_indices: ``nl.ndarray``
   :param recv_counts: [1, EP]@HBM, int32/uint32. Original dispatch ``recv_counts`` — used as combine's send-counts (metadata row 0).
   :type recv_counts: ``nl.ndarray``
   :param replica_group: EP replica group.
   :type replica_group: ``ReplicaGroup``
   :return: [T, H]@HBM. Output in original token order.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Supports top-k > 1 because the scatter accumulates across all EP contributions.
   * With LNC=2, EP is partitioned across cores and partial results are combined via ``nisa.sendrecv`` before core 0 writes the final output.

   **Dimensions**:

   * T: number of original local tokens (SP sharded).
   * H: hidden dimension.

