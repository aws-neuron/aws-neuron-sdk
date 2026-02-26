.. meta::
    :description: Attention CTE kernel implements attention optimized for Context Encoding (prefill) use cases.
    :date-modified: 11/28/2025

.. currentmodule:: nkilib.core.attention.attention_cte

Attention CTE Kernel API Reference
===================================

Implements attention optimized for Context Encoding (prefill) use cases with long sequence lengths.

The kernel supports:

* Efficient attention computation for long sequence lengths
* Causal masking
* Sliding window attention
* Context parallelism for distributed computation
* Prefix caching for efficient inference
* Sink tokens for streaming attention
* Native Grouped Query Attention (GQA) support
* Softmax caching for training

Background
--------------

The ``Attention CTE`` kernel is designed specifically for context encoding (prefill) scenarios where the sequence length is large (typically > 256). It performs the standard attention operation ``Attention(Q, K, V) = softmax(scale * Q @ K^T) @ V`` with optimizations for long sequence lengths.

The kernel employs efficient tiling strategies and memory access patterns to maximize performance on Neuron hardware. It supports various optimizations including flash attention for long sequences, LNC sharding, and context parallelism.

API Reference
----------------

**Source code for this kernel API can be found at**: `attention_cte.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_cte.py>`_


attention_cte
^^^^^^^^^^^^^^^

.. py:function:: attention_cte(q: nl.ndarray, k: nl.ndarray, v: nl.ndarray, scale: float = 1.0, causal_mask: bool = True, k_prior: Optional[nl.ndarray] = None, v_prior: Optional[nl.ndarray] = None, prior_used_len: Optional[nl.ndarray] = None, sink: Optional[nl.ndarray] = None, sliding_window: Optional[int] = None, tp_q: bool = True, tp_k: bool = False, tp_out: bool = False, cache_softmax: bool = False, softmax_dtype=nl.float32, cp_offset: Optional[nl.ndarray] = None, global_cp_deg: int = None, cp_strided_q_slicing: bool = False)

   Entrypoint NKI kernel that supports multiple attention variants.

   The kernel can be invoked with 1D SPMD grid for LNC2 or without grid.

   :param q: Query tensor with layout dependent on ``tp_q`` parameter
   :type q: ``nl.ndarray``
   :param k: Key tensor with layout dependent on ``tp_k`` parameter
   :type k: ``nl.ndarray``
   :param v: Value tensor with shape ``(batch_size_kv, seqlen, d)``
   :type v: ``nl.ndarray``
   :param scale: Scaling factor for attention scores. Must be 1.0 when using sliding window, context parallel, or prefix caching.
   :type scale: ``float``, optional
   :param causal_mask: Whether to use causal mask
   :type causal_mask: ``bool``, optional
   :param k_prior: (Prefix caching) Prior key tensor with layout dependent on ``tp_k`` parameter
   :type k_prior: ``nl.ndarray``, optional
   :param v_prior: (Prefix caching) Prior value tensor with shape ``(batch_size_kv, seqlen_prior, d)``
   :type v_prior: ``nl.ndarray``, optional
   :param prior_used_len: (Prefix caching) Actual used length in prior with shape ``(1,)``
   :type prior_used_len: ``nl.ndarray``, optional
   :param sink: Sink token tensor
   :type sink: ``nl.ndarray``, optional
   :param sliding_window: Sliding window size for attention, ``None`` or ``0`` denotes no sliding window mask
   :type sliding_window: ``int``, optional
   :param tp_q: Query tensor transpose flag
   :type tp_q: ``bool``, optional
   :param tp_k: Key tensor transpose flag
   :type tp_k: ``bool``, optional
   :param tp_out: Output tensor transpose flag
   :type tp_out: ``bool``, optional
   :param cache_softmax: Whether to cache softmax intermediate values
   :type cache_softmax: ``bool``, optional
   :param softmax_dtype: Data type for softmax computations
   :type softmax_dtype: ``nl.dtype``, optional
   :param cp_offset: Context parallel offset tensor
   :type cp_offset: ``nl.ndarray``, optional
   :param global_cp_deg: Global context parallel degree
   :type global_cp_deg: ``int``, optional
   :param cp_strided_q_slicing: Whether to use strided Q slicing for context parallelism. Default: False.
   :type cp_strided_q_slicing: ``bool``
   :return: Output tensor with attention results. Shape depends on ``tp_out`` parameter. If ``cache_softmax`` is ``True``, returns tuple of ``(output, out_neg_max, out_sum_recip)``.
   :rtype: ``nl.ndarray`` or ``tuple``

   **IO Shapes**:

   * q:
     ``(batch_size, seqlen_q, d)`` when ``tp_q`` is ``True``
     ``(batch_size, d, seqlen_q)`` when ``tp_q`` is ``False``
   * k:
     ``(batch_size_kv, seqlen_kv, d)`` when ``tp_k`` is ``True``
     ``(batch_size_kv, d, seqlen_kv)`` when ``tp_k`` is ``False``
   * v: ``(batch_size_kv, seqlen_kv, d)``
   * returns:
     ``(batch_size, d, seqlen_q)`` if ``tp_out`` is ``True``
     ``(batch_size, seqlen_q, d)`` if ``tp_out`` is ``False``

   **Constraints**:

   * Head dimension (``d``) must be <= 128
   * ``scale`` must be 1.0 when using sliding window, context parallel, or prefix caching
   * Context parallelism currently only supports causal attention
   * Sliding window attention currently only supports causal attention

Features
-----------

1. **Causal Masking (causal_mask=True)**:
   
   * Masks upper triangle of attention scores: ``S[i,j] = -inf`` when ``i < j``
   * Enables compute skipping: skip MM1/MM2 for upper triangle tiles

2. **Sliding Window Attention (SWA, when sliding_window > 0)**:
   
   * Local attention: each query only attends to nearby keys within a window
   * Masks attention scores: ``S[i,j] = -inf`` when ``|i - j| > sliding_window``
   * Currently only works with causal: masks both upper triangle AND positions outside window
   * When used with CP: loads only required KV slice to save memory

3. **Context Parallelism (CP, global_cp_deg > 1, cp_offset != None)**:
   
   * Distributes long sequence computation across multiple devices/ranks
   * Each rank (kernel call) processes a slice of Q sequence with full K/V
   * ``cp_offset`` indicates which Q slice this rank handles (runtime value)
   * Requires dynamic masking since offset unknown at compile time
   * Currently only supports causal attention

4. **Prefix Caching (k_prior/v_prior provided)**:
   
   * K/V split into two parts: prior (cached) and active (current)
   * ``prior_used_len`` specifies how much of prior to use (dynamic mask)
   * Causal mask not required for prior portion (although SWA still applies if enabled)

5. **Sink Tokens (sink provided)**:
   
   * Add additional sink token to softmax denominator

6. **Grouped Query Attention (GQA, batch_size_kv < batch_size)**:
   
   * Kernel handles GQA natively without explicit K/V replication

7. **Support for training**:
   
   * Kernel can optionally return maximum attention score and softmax denominator (per row) for backpropagation

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **LNC2 Sharding**: Shards computation across 2 NeuronCores with primary sharding on batch dimension and secondary sharding on sequence length for odd batch sizes.

2. **Flash Attention**: For K/V length > 10K tokens, divides into 8K-token sections and processes one section at a time to fit in SBUF memory.

3. **Software Pipelining**: Overlaps operations across Q groups (``i``, ``i+1``, ``i+2``) for efficient hardware utilization:
   
   * Group ``i``: PV computation, writeback
   * Group ``i+1``: Exp computation
   * Group ``i+2``: Q load, QK computation

4. **Modular Allocation**: Uses efficient buffer reuse with modular allocation for intermediate tensors.

5. **Dynamic Masking**: Implements efficient masking strategies for causal, sliding window, and context parallel scenarios.

6. **Optimized Memory Access**: Employs careful memory access patterns to optimize data movement between HBM and SBUF.



See Also
-----------

* :doc:`Attention TKG Kernel API Reference </nki/library/api/attention-tkg>`
