.. meta::
    :description: FlashAttention-2 forward: ``O = softmax(Q @ K^T * scale) @ V``.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.attention.flash_attention2_trn3

Flash Attention 2 Kernel API Reference
======================================

FlashAttention-2 forward: ``O = softmax(Q @ K^T * scale) @ V``.

Unmasked bidirectional attention for long key sequences -- the regime where Sk dominates and the softmax must stay online. Set ``sq_tiles_per_blk`` to ``ceil(Sq / 128)`` when all of Q fits SBUF: K and V are then streamed once for the whole of Q instead of once per Sq block, which is a significant win once Sk is long.

Background
-----------

The ``flash_attention_2`` kernel implements the FlashAttention-2 forward pass, ``O = softmax(Q @ K^T * scale) @ V``, for unmasked bidirectional attention over long key sequences on Trn3.

API Reference
--------------

**Source code for this kernel API can be found at**: `flash_attention_2.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/attention/flash_attention2_trn3/flash_attention_2.py>`_

flash_attention_2
^^^^^^^^^^^^^^^^^

.. py:function:: flash_attention_2(q: nl.NkiTensor, k: nl.NkiTensor, v: nl.NkiTensor, scale: float, config: Optional[FlashAttentionConfig] = None) -> nl.NkiTensor

   FlashAttention-2 forward: ``O = softmax(Q @ K^T * scale) @ V``.

   :param q: ``[B, Sq, D]`` in HBM. Must be a 2-byte dtype (transposed load uses ``dma_transpose``).
   :type q: ``nl.NkiTensor``
   :param k: ``[Bkv, D, Sk]`` in HBM.
   :type k: ``nl.NkiTensor``
   :param v: ``[Bkv, Sk, D]`` in HBM.
   :type v: ``nl.NkiTensor``
   :param scale: Attention-score scale.
   :type scale: ``float``
   :param config: Tuning knobs; defaults are used if omitted.
   :type config: ``Optional[FlashAttentionConfig]``

   **Notes**:

   * * **Unmasked.** Computes full bidirectional attention. There is no causal or padding mask, so this is not a drop-in prefill kernel for decoder models. * **Trn3 B1 only** (NeuronCore-v4 sub-version 1). ``nisa.exponential`` is a VectorE instruction that does not exist on Trn3 A0 (``trn3pre``, sub-version 0) or on any earlier generation, so ``trn3_a0`` and ``trn3_pds_a0`` are not supported targets. * ``Sq`` and ``Sk`` are arbitrary. Only ``D`` is constrained (a positive multiple of 128, at most 512, because ``D`` bounds MM2's moving free size). * **Compile with** ``--internal-skip-backend-allocation-opt-nki``. The kernel hand-places its PSUM banks, and the backend allocation optimization rearranges them, which measurably regresses the kernel. This is a leaked implementation detail and should become unnecessary once the underlying pass is fixed.

