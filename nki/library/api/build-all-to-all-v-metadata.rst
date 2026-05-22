.. meta::
    :description: Build metadata buffer for all_to_all_v collective from MoE routing decisions.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.subkernels

Build All To All V Metadata Kernel API Reference
================================================

Build metadata buffer for all_to_all_v collective from MoE routing decisions.

Computes per-rank send counts and displacements from expert assignments.

Background
-----------

The ``build_all_to_all_v_metadata`` kernel builds the metadata buffer required by the all_to_all_v collective operation from MoE routing decisions, computing per-rank send counts and displacements from expert assignments.

API Reference
--------------

**Source code for this kernel API can be found at**: `build_all_to_all_v_metadata.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/subkernels/build_all_to_all_v_metadata.py>`_

build_all_to_all_v_metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: build_all_to_all_v_metadata(expert_index: nl.ndarray, replica_group_size: int, E: int, recv_counts_known: bool = False, has_rdispls: bool = False)

   Build metadata buffer for all_to_all_v collective from MoE routing decisions.

   :param expert_index: [T, K] int32 HBM tensor indicating the K experts each token is routed to.
   :type expert_index: ``nl.ndarray``
   :param replica_group_size: Size of replica group for all_to_all_v collective.
   :type replica_group_size: ``int``
   :param E: Number of global experts.
   :type E: ``int``
   :param recv_counts_known: Not currently supported; when True, metadata includes recv counts.
   :type recv_counts_known: ``bool``
   :param has_rdispls: Not currently supported; when True, metadata includes recv displacements.
   :type has_rdispls: ``bool``
   :return: [n_rows, replica_group_size] uint32 HBM tensor. n_rows is 4 when has_rdispls=True, 3 otherwise. Row 0: send counts, Row 1: send displacements, Row 2: recv counts (zeros), Row 3 (optional): recv displacements (zeros).
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * T: Number of tokens.
   * K: Top-K experts per token.

