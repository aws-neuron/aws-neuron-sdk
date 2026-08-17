.. meta::
    :description: Build metadata for all2all dispatch using NKI.
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.experimental.moe_block

Build All2All Dispatch Metadata Kernel API Reference
====================================================

Build metadata for all2all dispatch using NKI.

Computes per-rank send_counts (with token deduplication) and send_displs from expert_index. Equivalent to the ``scatter_``-based PyTorch implementation but avoids XLA tracing issues.

Background
-----------

The ``build_all2all_dispatch_metadata`` kernel builds metadata for all2all dispatch using NKI.

API Reference
--------------

**Source code for this kernel API can be found at**: `build_all2all_dispatch_metadata.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe_block/build_all2all_dispatch_metadata.py>`_

build_all2all_dispatch_metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: build_all2all_dispatch_metadata(expert_index, num_experts, num_elements_per_token, replica_group_size)

   Build metadata for all2all dispatch using NKI.

   :param expert_index: [T, K] int32 tensor of expert indices per token.
   :param num_experts: Total number of experts.
   :param num_elements_per_token: Elements per token (e.g. H_CONCAT).
   :param replica_group_size: Number of destination ranks.

