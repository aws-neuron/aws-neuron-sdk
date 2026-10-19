.. meta::
    :description: Top-k over ``inp[BxS, vocab]`` for a vocab wider than nisa.topk's ceiling.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.topk

Chunked Top-K Kernel API Reference
==================================

Top-k over ``inp[BxS, vocab]`` for a vocab wider than nisa.topk's ceiling.

Background
-----------

The ``chunked_topk`` kernel computes top-k over a vocabulary wider than the 65,535-element limit of ``nisa.topk`` on a single core. It splits the vocabulary into chunks, takes a top-k of each, and reduces the candidates with a second top-k, with no collective.

API Reference
--------------

**Source code for this kernel API can be found at**: `chunked_topk.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/topk/chunked_topk.py>`_

chunked_topk
^^^^^^^^^^^^

.. py:function:: chunked_topk(inp: nl.NkiTensor, config: ChunkedTopkConfig) -> Tuple[nl.NkiTensor, nl.NkiTensor]

   Top-k over ``inp[BxS, vocab]`` for a vocab wider than nisa.topk's ceiling.

   :param inp: ``[BxS, vocab]`` bfloat16 tensor in HBM -- or, when ``config.interleaved_batch_groups > 1`` (jit-in-jit callers only), a LIST of batch-group slab tensors in the slabbed receive layout, one per producing exchange, so stage-1 tiles data-depend per slab.
   :type inp: ``nl.NkiTensor``
   :param config: ChunkedTopkConfig. ``config.chunks == 1`` delegates to gpsimd_topk.
   :type config: ``ChunkedTopkConfig``

   **Dimensions**:

   * BxS: number of logical rows (tokens)
   * vocab: full reduction length; may exceed 65,535

