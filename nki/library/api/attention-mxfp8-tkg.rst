.. meta::
    :description: MXFP8 flash decode attention with separate KV blocks and packed-Q eviction.
    :date-modified: 08/17/2026

.. currentmodule:: nkilib.experimental.attention_mxfp8

Attention MXFP8 TKG Kernel API Reference
========================================

MXFP8 flash decode attention with separate KV blocks and packed-Q eviction.

Token-generation (decode) attention over an MXFP8-quantized block KV cache on Trainium 3. Optimized for long contexts (bucket_size >= 2048 tokens, i.e. at least one full chunk); requires q_head == 64 and d_head == 128. All configuration is derived from input tensor shapes: bs, q_head, d_head from q.shape = [bs, q_head, 1, d_head] bucket_size from k_prior.shape = [num_blocks, 32, 160] Note: Tensor layouts differ from attention_tkg. This kernel uses H in the partition dim for packed-Q eviction, while attention_tkg uses d in partitions.

Background
-----------

The ``attention_mxfp8_tkg`` kernel performs MXFP8 flash decode attention with separate KV blocks and packed-Q eviction.

API Reference
--------------

**Source code for this kernel API can be found at**: `attention_mxfp8_tkg.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/attention_mxfp8/attention_mxfp8_tkg.py>`_

attention_mxfp8_tkg
^^^^^^^^^^^^^^^^^^^

.. py:function:: attention_mxfp8_tkg(q: nl.NkiTensor, k_active: nl.NkiTensor, v_active: nl.NkiTensor, k_prior: nl.NkiTensor, v_prior: nl.NkiTensor, mask: nl.NkiTensor, identity_hbm: Optional[nl.NkiTensor] = None, active_blocks_table: Optional[nl.NkiTensor] = None, sbm: Optional[SbufManager] = None) -> nl.NkiTensor

   MXFP8 flash decode attention with separate KV blocks and packed-Q eviction.

   :param q: Query tensor [B, H, 1, d] bfloat16.
   :type q: ``nl.NkiTensor``
   :param k_active: Active key [B, d] bfloat16.
   :type k_active: ``nl.NkiTensor``
   :param v_active: Active value [B, d] bfloat16.
   :type v_active: ``nl.NkiTensor``
   :param k_prior: MXFP8 K cache [num_blocks, 32, 160] float32. Each block = 128 tokens.
   :type k_prior: ``nl.NkiTensor``
   :param v_prior: MXFP8 V cache [num_blocks, 32, 160] float32. Each block = 128 tokens.
   :type v_prior: ``nl.NkiTensor``
   :param mask: Pre-computed chunk masks [B * num_chunks, 128, score_free] uint8.
   :type mask: ``nl.NkiTensor``
   :param identity_hbm: [128, 128] bfloat16 identity matrix for PE reduction.
   :type identity_hbm: ``Optional[nl.NkiTensor]``
   :param active_blocks_table: Block indices [B, num_blocks] int32.
   :type active_blocks_table: ``Optional[nl.NkiTensor]``
   :param sbm: Optional SbufManager for SBUF allocation. None = auto-alloc mode.
   :type sbm: ``Optional[SbufManager]``
   :return: [B, H, d] bfloat16 attention output.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * B: Batch size.
   * H: Number of query heads (must be 64).
   * d: Head dimension (must be 128).
   * num_blocks: KV cache blocks; each block covers 128 tokens as [32, 160] MXFP8.
   * num_chunks: bucket_size / 2048 online-softmax iterations.

