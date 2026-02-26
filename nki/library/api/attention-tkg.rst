.. meta::
    :description: Attention TKG kernel implements attention optimized for Token Generation (decode) use cases.
    :date-modified: 11/28/2025

.. currentmodule:: nkilib.core.attention_tkg

Attention TKG Kernel API Reference
===================================

Implements attention optimized for Token Generation (decode) use cases with small active sequence lengths.

The kernel supports:

* Efficient attention computation for small active sequence lengths
* Flexible tensor placement in SBUF or HBM
* Adaptive LNC2 sharding strategies
* In-kernel mask generation
* Fused RoPE (Rotary Position Embedding)
* Block KV cache for efficient long-context inference
* Attention sink for streaming attention
* GPSIMD optimizations for inter-core communication

Background
--------------

The ``Attention TKG`` kernel is designed specifically for token generation (decoding) scenarios where the active sequence length is small (typically ≤ 7). It performs the standard attention operation ``Attention(Q, K, V) = softmax(Q @ K^T) @ V`` with optimizations for small active sequence lengths and large KV caches.

The kernel employs efficient tiling strategies and memory access patterns to maximize performance on Neuron hardware. It supports various optimizations including LNC sharding, block KV cache, and attention sink for streaming attention.

API Reference
----------------

**Source code for this kernel API can be found at**: `attention_tkg.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/attention/attention_tkg.py>`_

AttnTKGConfig
^^^^^^^^^^^^^^^

.. py:class:: AttnTKGConfig

   Configuration for token-generation attention kernel.

   This dataclass contains shape parameters and performance optimization flags
   for the attention_tkg kernel, which is optimized for small active sequence lengths.

   .. py:attribute:: bs
      :type: int
      :value: 0

      Batch size

   .. py:attribute:: q_head
      :type: int
      :value: 0

      Number of query heads

   .. py:attribute:: s_active
      :type: int
      :value: 0

      Active sequence length (>1 means speculative decoding)

   .. py:attribute:: curr_sprior
      :type: int
      :value: 0

      Current prior sequence length (KV cache length for this execution)

   .. py:attribute:: full_sprior
      :type: int
      :value: 0

      Full prior sequence length (maximum KV cache capacity)

   .. py:attribute:: d_head
      :type: int
      :value: 0

      Head dimension (embedding size per head)

   .. py:attribute:: block_len
      :type: int
      :value: 0

      Block length for block KV cache (0 if not using block KV)

   .. py:attribute:: tp_k_prior
      :type: bool
      :value: False

      Specifies that k_prior is transposed (shape ``[B, 1, d, s_prior]`` instead of ``[B, 1, s_prior, d]``)

   .. py:attribute:: strided_mm1
      :type: bool
      :value: True

      Use strided memory access for first matmul to improve cache locality

   .. py:attribute:: use_pos_id
      :type: bool
      :value: False

      Generate attention mask from position IDs in-kernel instead of loading pre-generated mask

   .. py:attribute:: fuse_rope
      :type: bool
      :value: False

      Fuse RoPE (Rotary Position Embedding) computation into the kernel

   .. py:attribute:: use_gpsimd_sb2sb
      :type: bool
      :value: True

      Use GPSIMD instructions for SBUF-to-SBUF data transfers (LNC2 sharding)

   .. py:attribute:: qk_in_sb
      :type: bool
      :value: False

      Query and key tensors are already in SBUF instead of HBM

   .. py:attribute:: k_out_in_sb
      :type: bool
      :value: False

      Output key tensor after RoPE should be stored in SBUF instead of HBM

   .. py:attribute:: out_in_sb
      :type: bool
      :value: False

      Output tensor should be stored in SBUF instead of HBM

attention_tkg
^^^^^^^^^^^^^^^

.. py:function:: attention_tkg(q: nl.ndarray, k_active: nl.ndarray, v_active: nl.ndarray, k_prior: nl.ndarray, v_prior: nl.ndarray, mask: nl.ndarray, out: nl.ndarray, cfg: AttnTKGConfig, sbm: SbufManager, inv_freqs: Optional[nl.ndarray] = None, rope_pos_ids: Optional[nl.ndarray] = None, sink: Optional[nl.ndarray] = None, active_blocks_table: Optional[nl.ndarray] = None, k_out: Optional[nl.ndarray] = None, DBG_TENSORS: Optional[tuple] = None) -> Tuple[nl.ndarray, Optional[nl.ndarray]]

   Attention specifically optimized for token-gen (where s_active is small). Can optionally fuse RoPE at the start.

   :param q: Query tensor. Shape depends on ``cfg.qk_in_sb``: If ``True``: ``[d, B * H * s_active]``, else: ``[B, d, H, s_active]``
   :type q: ``nl.ndarray``
   :param k_active: Active key tensor. Shape depends on ``cfg.qk_in_sb``: If ``True``: ``[d, B * s_active]``, else: ``[B, d, s_active]``
   :type k_active: ``nl.ndarray``
   :param v_active: Active value tensor. Shape: ``[B, 1, s_active, d]``
   :type v_active: ``nl.ndarray``
   :param k_prior: Prior key tensor from KV cache. Shape: ``[B+, 1, s_prior, d]`` if ``cfg.tp_k_prior`` else ``[B+, 1, d, s_prior]``. For block KV cache, shape is ``[B+ * block_count, block_len, d]``
   :type k_prior: ``nl.ndarray``
   :param v_prior: Prior value tensor from KV cache. Shape: ``[B+, 1, s_prior, d]``. For block KV cache, shape is ``[B+ * block_count, block_len, d]``
   :type v_prior: ``nl.ndarray``
   :param mask: Attention mask. Shape: ``[s_active, B, H, s_active]`` if ``cfg.use_pos_id`` else ``[s_prior, B, H, s_active]``
   :type mask: ``nl.ndarray``
   :param out: Output tensor. Shape depends on ``cfg.out_in_sb``: If ``True``: ``[d, B * H * s_active]``, else: ``[B, H, d, s_active]``
   :type out: ``nl.ndarray``
   :param cfg: Kernel configuration with shapes and performance flags
   :type cfg: ``AttnTKGConfig``
   :param sbm: SBUF memory manager for allocating temporary buffers
   :type sbm: ``SbufManager``
   :param inv_freqs: Inverse frequencies for RoPE. Shape: ``[d // 2, 1]``. Required when ``cfg.fuse_rope`` is ``True``
   :type inv_freqs: ``nl.ndarray``, optional
   :param rope_pos_ids: Position IDs for RoPE. Shape: ``[B, s_active]``. Required when ``cfg.fuse_rope`` or ``cfg.use_pos_id`` is ``True``
   :type rope_pos_ids: ``nl.ndarray``, optional
   :param sink: Sink attention tokens. Shape: ``[H, 1]`` for streaming attention sink tokens
   :type sink: ``nl.ndarray``, optional
   :param active_blocks_table: Table of active blocks for block KV cache. Shape: ``[B, num_blocks]``. Required when using block KV cache
   :type active_blocks_table: ``nl.ndarray``, optional
   :param k_out: Output key tensor after RoPE. Shape depends on ``cfg.k_out_in_sb``: If ``True``: ``[d, B * s_active]``, else: ``[B, 1, d, s_active]``
   :type k_out: ``nl.ndarray``, optional
   :param DBG_TENSORS: Optional tuple of 4-5 debug tensors with shared HBM type for intermediate value inspection
   :type DBG_TENSORS: ``tuple``, optional
   :return: Tuple of ``(out, k_out)`` where ``out`` is the attention output tensor and ``k_out`` is the key output tensor (if ``cfg.fuse_rope`` is ``True``)
   :rtype: ``tuple``

   **Constraints**:

   * Optimized for ``s_active <= 7`` and ``d_head <= 128``
   * ``cfg.qk_in_sb=True`` is required when skipping fused RoPE
   * Block KV cache requires ``cfg.qk_in_sb=True``
   * In-kernel mask generation (``cfg.use_pos_id=True``) is not supported with batch sharding or block KV cache

Features
-----------

1. **Flexible Tensor Placement**:
   
   * ``q``, ``k``, ``k_out``, and ``out`` tensors can be placed in either SBUF or HBM
   * When ``qk_in_sb=True``, q and k tensors are pre-loaded in SBUF (required for block KV cache)
   * ``out_in_sb`` and ``k_out_in_sb`` flags control output tensor placement for reduced memory transfers
   * Use this feature for performance improvement when integrating this kernel into a larger kernel

2. **Adaptive LNC2 Sharding**:
   
   * Automatically selects sharding strategy based on tensor dimensions
   * Batch sharding: Used when batch is even AND (``s_prior < 256`` OR ``b*q_head*s_active > 128``)
   * Sequence sharding: Used when ``s_prior >= 256`` and batch sharding criteria not met
   * Balances computation across 2 NeuronCores for improved throughput

3. **Mask Generation**:
   
   * ``use_pos_id=False``: Pre-generated mask loaded from HBM
   * ``use_pos_id=True``: Mask generated in-kernel from position IDs
   * In-kernel generation reduces memory bandwidth but requires position ID input

4. **Fused RoPE (Rotary Position Embedding)**:
   
   * ``fuse_rope`` integrates RoPE computation directly into the attention kernel
   * Applies rotary embeddings to Q and K tensors, scaling Q by ``1/sqrt(d_head)``
   * Reduces memory traffic by avoiding separate RoPE passes

5. **Block KV Cache**:
   
   * Supports block-sparse KV cache with configurable ``block_len``
   * Uses ``active_blocks_table`` to track which cache blocks are active per batch
   * Enables efficient long-context inference with sparse memory access patterns

6. **K_prior Transpose Handling**:
   
   * ``tp_k_prior`` flag indicates whether K_prior is pre-transposed in memory
   * Optimizes memory layout: ``[B, 1, d, s_prior]`` when ``tp_k_prior=True`` vs ``[B, 1, s_prior, d]`` when False
   * Reduces transpose operations during computation and improves interoperability with other kernels

7. **Strided Memory Access (strided_mm1)**:
   
   * Enables strided read patterns for K in first matmul
   * When enabled, allows MM2 to use sequential V reads for better DMA throughput
   * Trades off MM1 memory access for MM2 optimization

8.  **Attention Sink**:
   
   * Supports streaming attention with sink tokens for infinite context
   * Sink tokens maintain fixed attention scores across all positions
   * Integrated into softmax reduction for minimal overhead

9.  **GPSIMD SBUF-to-SBUF Transfers**:
    
   * ``use_gpsimd_sb2sb`` enables high-performance GPSIMD instructions for inter-core communication
   * Optimizes LNC2 sharding by using extended instructions for SBUF-to-SBUF data transfers

10. **Context Length Management**:
    
    * ``curr_sprior``: Current prior sequence length (actual KV cache content for this invocation)
    * ``full_sprior``: Full prior sequence length (maximum KV cache capacity allocated)
    * Allows progressive filling of KV cache during autoregressive generation

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **Efficient Tiling Strategy**: Uses carefully chosen tile sizes for processing batches, sequences, and heads to maximize hardware utilization.

2. **Cascaded Reduction**: Implements cascaded max and sum reduction operations for softmax computation to maintain numerical stability.

3. **Memory Access Optimization**: Employs careful memory access patterns to optimize data movement between HBM and SBUF.

4. **Block KV Cache Support**: Implements efficient block-sparse KV cache with dynamic block size adjustment to ensure optimal hardware utilization.

5. **Attention Sink Integration**: Efficiently integrates attention sink tokens into the softmax computation for streaming attention.

6. **Fused RoPE Implementation**: Implements efficient rotary position embeddings with optimized trigonometric computations.

7. **Adaptive Sharding**: Dynamically selects between batch and sequence sharding based on tensor dimensions to optimize performance.

8. **GPSIMD Optimization**: Uses GPSIMD instructions for high-performance SBUF-to-SBUF data transfers in LNC2 sharding.

9. **Debug Support**: Provides comprehensive debug tensor support for intermediate value inspection.

10. **Stack-based SBUF Allocation**: Uses SbufManager for efficient on-chip memory management with hierarchical scoping.



See Also
-----------

* :doc:`Output Projection TKG Kernel API Reference </nki/library/api/output-projection-tkg>`
