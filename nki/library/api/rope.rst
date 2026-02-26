.. meta::
    :description: RoPE kernel applies Rotary Position Embedding to input embeddings.
    :date-modified: 01/21/2026

.. currentmodule:: nkilib.core.rope

RoPE Kernel API Reference
==========================

Applies Rotary Position Embedding (RoPE) to input embeddings, encoding positional information by rotating embedding dimension pairs using precomputed sine/cosine frequencies.

The kernel supports:

* Efficient position encoding without absolute position embeddings
* Optional LNC sharding for parallelization across cores
* Flexible memory layouts (contiguous or interleaved)
* Layout conversion strategies (DMA strided access or SBUF matmul)
* Standalone operation with HBM I/O
* SBUF-only operation for megakernel fusion

Background
--------------

The ``RoPE`` kernel implements Rotary Position Embedding, which encodes positional information by rotating pairs of embedding dimensions using precomputed sine/cosine frequencies. This approach enables position-aware attention mechanisms without requiring absolute position embeddings.

The kernel applies the following transformation:

* ``out[even] = x[even] * cos - x[odd] * sin``
* ``out[odd] = x[odd] * cos + x[even] * sin``

The kernel supports two memory layouts for the head dimension: contiguous (first half, second half) and interleaved (even, odd, even, odd). Layout conversion can be performed using either strided DMA access or SBUF matmul operations.

API Reference
----------------

**Source code for this kernel API can be found at**: `rope.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/embeddings/rope.py>`_

RoPE
^^^^^^^^^^^^^^^

.. py:function:: RoPE(x_in, cos, sin, lnc_shard=False, contiguous_layout=True, relayout_in_sbuf=False)

   Apply Rotary Position Embedding (RoPE) to input embeddings.
   Standalone kernel with HBM I/O and optional LNC sharding.

   :param x_in: Input embeddings tensor with shape ``[d_head, B, n_heads, S]`` in HBM
   :type x_in: ``nl.ndarray``
   :param cos: Cosine frequencies tensor with shape ``[d_head//2, B, S]`` in HBM
   :type cos: ``nl.ndarray``
   :param sin: Sine frequencies tensor with shape ``[d_head//2, B, S]`` in HBM
   :type sin: ``nl.ndarray``
   :param lnc_shard: Parallelize across LNC cores by tiling sequence dimension. Default is ``False``.
   :type lnc_shard: ``bool``, optional
   :param contiguous_layout: Memory layout in d_head dimension. ``True`` for ``[first_half, second_half]`` (default, more efficient), ``False`` for ``[even, odd, even, odd, ...]`` (interleaved).
   :type contiguous_layout: ``bool``, optional
   :param relayout_in_sbuf: Use SBUF matmul for layout conversion (only for small tensors). Default is ``False``.
   :type relayout_in_sbuf: ``bool``, optional
   :return: RoPE applied output tensor with shape ``[d_head, B, n_heads, S]`` in HBM
   :rtype: ``nl.ndarray``

   **Constraints**:

   * Head dimension (``d_head``) must be 64 or 128
   * Batch size (``B``) must be in range (0, 64]
   * Sequence length (``S``) must be in range (0, 512]
   * Number of heads (``n_heads``) must be in range (0, 16]
   * When ``lnc_shard=True``, sequence length must be divisible by number of programs
   * SBUF relayout (``relayout_in_sbuf=True``) requires ``B * n_heads * S <= gemm_moving_fmax``

RoPE_sbuf
^^^^^^^^^^^^^^^

.. py:function:: RoPE_sbuf(x_in_sb, cos_sb, sin_sb, x_out_sb, convert_from_interleaved=False)

   Apply RoPE on tensors in SBUF (for megakernel fusion).
   Helper function that operates entirely in SBUF without HBM I/O.

   :param x_in_sb: Input embeddings tensor with shape ``[d_head, B, n_heads, S]`` in SBUF
   :type x_in_sb: ``nl.ndarray``
   :param cos_sb: Cosine frequencies tensor with shape ``[d_head//2, B, S]`` in SBUF
   :type cos_sb: ``nl.ndarray``
   :param sin_sb: Sine frequencies tensor with shape ``[d_head//2, B, S]`` in SBUF
   :type sin_sb: ``nl.ndarray``
   :param x_out_sb: Output buffer tensor with shape ``[d_head, B, n_heads, S]`` in SBUF
   :type x_out_sb: ``nl.ndarray``
   :param convert_from_interleaved: Convert from interleaved to contiguous layout (only for small tensors: ``B * n_heads * S <= gemm_moving_fmax``). Default is ``False``.
   :type convert_from_interleaved: ``bool``, optional
   :return: Output tensor with RoPE applied (modified in-place)
   :rtype: ``nl.ndarray``

   **Constraints**:

   * Assumes contiguous layout unless ``convert_from_interleaved=True``
   * For large tensors with interleaved layout, use ``RoPE()`` with strided DMA
   * Input and output tensors must have matching dtypes

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **Layout Conversion Strategies**: Supports two methods for converting between contiguous and interleaved layouts:
   
   * **DMA Strided Access**: Uses strided DMA operations with step=2 to gather/scatter even and odd indices separately. Suitable for all tensor sizes.
   * **SBUF Matmul**: Uses matrix multiplication with a permutation matrix for layout conversion. Limited to small tensors where ``B * n_heads * S <= gemm_moving_fmax``.

2. **LNC Sharding**: Supports parallelization across Logical NeuronCore (LNC) cores by tiling the sequence dimension. Each core processes a tile of size ``S // n_prgs``.

3. **Efficient Tensor Operations**: Uses ``tensor_tensor`` operations with TensorView broadcasting to efficiently apply cos/sin coefficients across the n_heads dimension.

4. **Memory Management**: Carefully manages SBUF allocations for intermediate buffers including separate storage for odd half elements to satisfy tensor_tensor alignment requirements.

5. **Permutation Matrix Generation**: For SBUF layout conversion, generates a permutation matrix using strided access on an identity matrix, enabling efficient transformation via matrix multiplication.



See Also
-----------

* :doc:`RoPE HuggingFace Kernel API Reference </nki/library/api/rope-hf>`
