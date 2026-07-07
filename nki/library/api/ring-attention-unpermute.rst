.. meta::
    :description: Unpermute striped ring attention output to contiguous sequence order.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.attention

Ring Attention Unpermute Kernel API Reference
=============================================

Unpermute striped ring attention output to contiguous sequence order.

Each rank holds tokens at striped positions ``[rank, rank+nw, rank+2*nw, ...]``. After unpermute, each rank holds a contiguous chunk of the global sequence: rank 0 gets ``[0, spr)``, rank 1 gets ``[spr, 2*spr)``, etc. Algorithm: all_gather all ranks' data, then use strided DMA copies with rank-dependent offset to extract and interleave this rank's contiguous chunk. Requires ``spr % nw == 0`` (seqlen_per_rank evenly divisible by num_workers).

Background
-----------

The ``ring_attention_unpermute`` kernel reorders striped ring-attention output back to contiguous sequence order, so that each rank ends up owning a contiguous chunk of the global sequence after context-parallel attention.

API Reference
--------------

**Source code for this kernel API can be found at**: `ring_attention_unpermute.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/attention/ring_attention_unpermute.py>`_

ring_attention_unpermute
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: ring_attention_unpermute(x: nl.ndarray, replica_groups: tuple = None, num_workers: int = 1)

   Unpermute striped ring attention output to contiguous sequence order.

   :param x: [bs, seqlen_per_rank, d] — this rank's striped tokens (fp16/bf16).
   :type x: ``nl.ndarray``
   :param replica_groups: Replica group specification for collective communication.
   :type replica_groups: ``tuple``
   :param num_workers: Number of CP ranks in the ring.
   :type num_workers: ``int``
   :return: [bs, seqlen_per_rank, d] — this rank's contiguous chunk.
   :rtype: ``nl.ndarray``

