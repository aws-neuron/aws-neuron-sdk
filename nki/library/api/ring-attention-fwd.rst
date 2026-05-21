.. meta::
    :description: Ring attention forward using attention_cte with HBM I/O.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.attention

Ring Attention Fwd Kernel API Reference
=======================================

Ring attention forward using attention_cte with HBM I/O.

Implements ring attention for context parallelism across multiple workers. All Q/K/V/O stay in HBM. After each attention_cte call, the output is corrected via a tiled HBM roundtrip (load one group at a time into SBUF, apply online softmax correction, write back). The collective permute runs in parallel with the correction for latency hiding. LNC sharding and batch iteration are handled internally by attention_cte, so this kernel does not manage LNC explicitly for the attention compute. However, the K/V DMA transfers (initial copy and per-step swap) are sharded across NCs so each NC handles only its assigned batches, parallelizing the DMA work under LNC2.

Background
-----------

The ``ring_attention_spmd_fwd`` kernel implements ring attention for context parallelism across multiple workers. All Q/K/V/O tensors reside in HBM. After each attention step, the output is corrected via online softmax rescaling, while a collective permute runs in parallel for latency hiding.

API Reference
--------------

**Source code for this kernel API can be found at**: `ring_attention_fwd.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/attention/ring_attention_fwd.py>`_

ring_attention_spmd_fwd
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: ring_attention_spmd_fwd(q: nl.ndarray, k: nl.ndarray, v: nl.ndarray, replica_groups: tuple = None, num_workers: int = 1, softmax_scale: float = None, use_causal_mask: bool = False, striped_input: bool = False, training: bool = False, lse_dtype: nki.dtype = nl.float32, tp_q: bool = False, tp_k: bool = False)

   Ring attention forward using attention_cte with HBM I/O.

   :param q: Query tensor. Shape depends on tp_q: [b, h, seqlen, d] when tp_q=True (non-transposed, kernel transposes internally via dma_transpose). [b, h, d, seqlen] when tp_q=False (pre-transposed layout).
   :type q: ``nl.ndarray``
   :param k: Key tensor. Shape depends on tp_k: [b, h, seqlen, d] when tp_k=True (non-transposed, kernel transposes internally via dma_transpose). [b, h, d, seqlen] when tp_k=False (pre-transposed layout).
   :type k: ``nl.ndarray``
   :param v: [b, h, seqlen, d], Value tensor (non-transposed layout).
   :type v: ``nl.ndarray``
   :param replica_groups: Replica groups for collective communication.
   :type replica_groups: ``tuple``
   :param num_workers: Number of workers in the ring.
   :type num_workers: ``int``
   :param softmax_scale: Softmax scale factor. Default: 1/sqrt(d). When use_causal_mask=True, the caller must pre-scale Q by softmax_scale before calling this kernel, and pass softmax_scale=1.0 (attention_cte requires scale=1.0 in CP mode).
   :type softmax_scale: ``float``
   :param use_causal_mask: Whether to apply causal masking.
   :type use_causal_mask: ``bool``
   :param striped_input: Whether input is striped (requires use_causal_mask=True).
   :type striped_input: ``bool``
   :param training: Whether to output LSE for backward pass. Default: False.
   :type training: ``bool``
   :param lse_dtype: Data type for the LSE output tensor. Default: nl.float32.
   :type lse_dtype: ``nki.dtype``
   :param tp_q: Query tensor transpose flag. When True, q is in non-transposed layout (batch, seqlen, d) and attention_cte will transpose internally via dma_transpose. When False (default), q must be pre-transposed to (batch, d, seqlen).
   :type tp_q: ``bool``
   :param tp_k: Key tensor transpose flag. When True, k is in non-transposed layout (batch, seqlen, d) and attention_cte will transpose internally via dma_transpose. When False (default), k must be pre-transposed to (batch, d, seqlen). The ring transfer buffers match the input k layout, so collective permute transfers k in whichever layout the caller provides.
   :type tp_k: ``bool``
   :return: [b, h, seqlen, d], Attention output.
   :rtype: ``nl.ndarray``
   :return: [b, h, 128, seqlen//128], Log-sum-exp (if training).
   :rtype: ``nl.ndarray``

   **Notes**:

   * Requires Trainium2 or later (not supported on trn1)
   * MHA only (q_heads == kv_heads, must broadcast before calling)
   * Supports LNC1 and LNC2 (sharded on batch * heads by attention_cte)
   * Supports non-causal and causal attention
   * Supports striped causal masking (via cp_offset adjustment: 0 or -1)
   * Tested with up to 16k seqlen per rank (1M total sequence on trn2/trn3)

   **Dimensions**:

   * b: Batch size
   * h: Number of attention heads
   * d: Head dimension (must be <= 128)

