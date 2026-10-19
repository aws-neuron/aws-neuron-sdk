.. meta::
    :description: Router Top-K kernel computes router logits and performs top-K selection for MoE models.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.core.router_topk

Router Top-K Kernel API Reference
==================================

Computes router logits, applies activation functions, and performs top-K selection with expert affinity scattering for Mixture of Experts (MoE) models.

The kernel supports:

* Router logits computation (x @ w + bias)
* Activation functions (SOFTMAX, SIGMOID)
* Top-K expert selection (K ≤ 8)
* Expert affinity scattering (one-hot or indirect DMA)
* Multiple layout configurations and optimization modes
* Column tiling for small token counts
* LNC sharding across token dimension
* Pre-norm and post-norm activation pipelines
* L1 normalization of top-K probabilities

Background
--------------

The ``Router Top-K`` kernel is a core component of Mixture of Experts (MoE) models, responsible for routing tokens to the most relevant experts. The kernel computes router logits by multiplying input tokens with a weight matrix, applies activation functions, selects the top-K experts for each token, and scatters the expert affinities to the full expert dimension.

The kernel is optimized for token counts T ≤ 2048, expert counts E ≤ 512, hidden dimensions H that are multiples of 128, and K ≤ 8 top experts per token. It supports both context encoding (CTE) with larger T and token generation (TKG) with T ≤ 128.

**Pipeline Configurations**:

The kernel supports multiple pipeline configurations:

1. **(topK, ACT2, Scatter)**: Standard pipeline with post-topK activation
2. **(ACT1, topK)**: Pre-norm activation before topK selection
3. **(ACT1, topK, Norm, Scatter)**: Pre-norm with L1 normalization and scatter

API Reference
----------------

**Source code for this kernel API can be found at**: `router_topk.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/core/router_topk/router_topk.py>`_

router_topk
^^^^^^^^^^^^^^^

.. py:function:: router_topk(x: nl.NkiTensor, w: nl.NkiTensor, w_bias: nl.NkiTensor, router_logits: nl.NkiTensor, expert_affinities: nl.NkiTensor, expert_index: nl.NkiTensor, act_fn: RouterActFnType, k: int, x_hbm_layout: int, x_sb_layout: int, router_pre_norm: bool = True, norm_topk_prob: bool = False, use_column_tiling: bool = False, use_indirect_dma_scatter: bool = False, return_eager_affi: bool = False, use_PE_broadcast_w_bias: bool = False, shard_on_tokens: bool = False, skip_store_expert_index: bool = False, skip_store_router_logits: bool = False, skip_store_expert_affinities: bool = False, expert_affinities_eager_out: nl.NkiTensor = None)

   Router top-K kernel for Mixture of Experts (MoE) models.

   Computes router logits (x @ w + bias), applies activation functions, performs top-K selection,
   and scatters expert affinities. Supports multiple layout configurations, sharding strategies,
   and optimization modes.

   :param x: Input tensor. Buffer type is auto-detected. If in HBM: [H, T] or [T, H] depending on x_hbm_layout. If in SBUF: a permutation of [128, T, H/128] depending on x_sb_layout.
   :type x: ``nl.NkiTensor``
   :param w: Weight tensor [H, E] in HBM
   :type w: ``nl.NkiTensor``
   :param w_bias: Optional bias tensor [1, E] or [E] in HBM
   :type w_bias: ``nl.NkiTensor``
   :param router_logits: Output router logits [T, E] in HBM
   :type router_logits: ``nl.NkiTensor``
   :param expert_affinities: Output expert affinities [T, E] in HBM or SBUF. Buffer type is auto-detected.
   :type expert_affinities: ``nl.NkiTensor``
   :param expert_index: Output expert indices [T, K] in HBM or SBUF. Buffer type is auto-detected.
   :type expert_index: ``nl.NkiTensor``
   :param act_fn: Activation function (SOFTMAX or SIGMOID)
   :type act_fn: ``RouterActFnType``
   :param k: Number of top experts to select (must be <= 8)
   :type k: ``int``
   :param x_hbm_layout: Layout of x in HBM (0=[H,T], 1=[T,H])
   :type x_hbm_layout: ``int``
   :param x_sb_layout: Layout of x in SBUF (0-3, see router_topk_input_x_load for details)
   :type x_sb_layout: ``int``
   :param router_pre_norm: If True, apply activation before top-K (ACT1 pipeline)
   :type router_pre_norm: ``bool``
   :param norm_topk_prob: If True, normalize top-K probabilities with L1 norm
   :type norm_topk_prob: ``bool``
   :param use_column_tiling: Enable PE array column tiling for small T
   :type use_column_tiling: ``bool``
   :param use_indirect_dma_scatter: Use indirect DMA for expert affinity scatter
   :type use_indirect_dma_scatter: ``bool``
   :param return_eager_affi: If True, return top-K affinities in addition to scattered
   :type return_eager_affi: ``bool``
   :param use_PE_broadcast_w_bias: Use tensor engine for bias broadcast
   :type use_PE_broadcast_w_bias: ``bool``
   :param shard_on_tokens: Enable LNC sharding across token dimension
   :type shard_on_tokens: ``bool``
   :param skip_store_expert_index: Skip storing expert indices to HBM
   :type skip_store_expert_index: ``bool``
   :param skip_store_router_logits: Skip storing router logits to HBM
   :type skip_store_router_logits: ``bool``
   :param skip_store_expert_affinities: Skip scattering/storing the dense [T, E] expert affinities. Only valid with return_eager_affi=True, where the dense [T, k] eager affinities carry all the affinity values the caller needs. Avoids the [T, E] HBM allocation, the zero-init, and the indirect-DMA scatter.
   :type skip_store_expert_affinities: ``bool``
   :param expert_affinities_eager_out: Optional [T, k] HBM output for the dense eager affinities. When provided (requires return_eager_affi=True), each LNC core writes ONLY its own T_local slice directly to expert_affinities_eager_out[T_offset : T_offset + T_local] — the two cores' disjoint slices compose in shared HBM with no cross-core sendrecv or cross-partition assembly (mirrors the direct expert_index store). This is strictly cheaper than assembling the full [T, k] in SBUF and is the preferred path when the eager consumer reads from HBM.
   :type expert_affinities_eager_out: ``nl.NkiTensor``
   :return: [router_logits, expert_index, expert_affinities, optional: expert_affinities_topk]
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * T: Total number of tokens
   * H: Hidden dimension size
   * E: Number of experts
   * K: Number of top experts to select per token

   **Constraints**:

   * K must be ≤ 8
   * E must be ≤ 512 (gemm_moving_fmax)
   * H must be a multiple of 128
   * SIGMOID activation requires ``use_indirect_dma_scatter=True``
   * ``router_pre_norm`` requires ``use_indirect_dma_scatter=True``
   * With ``use_indirect_dma_scatter``, T must be ≤ 128 or multiple of 128
   * ``shard_on_tokens`` requires n_prgs > 1 and T divisible by 2
   * ``output_in_sbuf`` requires T ≤ 128

   **SBUF Layout Options** (``x_sb_layout``):

   * 0: ``[128, T, H/128]`` - P-dim contains H elements with stride of H/128
   * 1: ``[128, T, H/128]`` - P-dim with H/256 chunk interleaving
   * 2: ``[128, T, H/128]`` - P-dim contains consecutive H elements
   * 3: ``[128, H/128, T]`` - H-tiles in dim-1, T in dim-2

router_topk_input_x_load
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: router_topk_input_x_load(x: nl.NkiTensor, hbm_layout = XHBMLayout_H_T__0, sb_layout = XSBLayout_tp2013__1)

   Load input tensor x from HBM to SBUF with specified layout transformations.

   Performs DMA transfer from HBM to SBUF with layout conversion based on hbm_layout
   and sb_layout parameters. Supports multiple layout combinations optimized for
   different access patterns in subsequent matmul operations.

   :param x: Input tensor in HBM. Shape ``[H, T]`` if hbm_layout=0, ``[T, H]`` if hbm_layout=1
   :type x: ``nl.ndarray``
   :param hbm_layout: Layout of x in HBM (0=[H,T], 1=[T,H]). Default is 0.
   :type hbm_layout: ``int``, optional
   :param sb_layout: Target layout in SBUF (0-3). Default is 1.
   :type sb_layout: ``int``, optional
   :return: Input tensor in SBUF with transformed layout
   :rtype: ``nl.ndarray``

   **Constraints**:

   * H must be a multiple of 128
   * Supported combinations: (hbm_layout=0, sb_layout=3) and (hbm_layout=1, sb_layout=0/1/2)

router_topk_input_w_load
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: router_topk_input_w_load(w: nl.NkiTensor, x_sb_layout)

   Load weight tensor w from HBM to SBUF with layout matching x tensor.

   :param w: Weight tensor with shape ``[H, E]`` in HBM
   :type w: ``nl.ndarray``
   :param x_sb_layout: Layout of x in SBUF (determines w layout)
   :type x_sb_layout: ``int``
   :return: Weight tensor in SBUF with appropriate layout
   :rtype: ``nl.ndarray``

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **Tiled Matrix Multiplication**: Tiles computation on both H (contraction dimension) and T (token dimension) for efficient memory access and hardware utilization.

2. **PE Array Column Tiling**: For small token counts (T < 128), splits the PE array column-wise into multiple tiles (32, 64, or 128 columns) to enable parallel execution of independent matmuls.

3. **LNC Sharding**: Supports parallelization across 2 cores by sharding the token dimension. Each core processes T/2 tokens with automatic load balancing for non-divisible token counts.

4. **Bias Broadcasting**: Supports two methods for bias application:
   
   * Stream shuffle broadcast (default)
   * Tensor engine matmul with ones mask (``use_PE_broadcast_w_bias=True``)

5. **Top-K Selection**: Uses hardware-accelerated ``max8`` and ``nc_find_index8`` instructions to efficiently find top-8 values and their indices.

6. **Expert Affinity Scattering**: Supports two scattering methods:
   
   * **One-hot scatter**: Uses mask-based selection with element-wise operations
   * **Indirect DMA scatter**: Uses dynamic indexing for efficient scatter to HBM

7. **Activation Pipelines**: Supports multiple activation pipeline configurations (ACT1, ACT2) with optional L1 normalization.

8. **Memory Management**: Carefully manages SBUF allocations with modular allocation and buffer reuse for intermediate tensors.



See Also
-----------

* :doc:`/nki/library/api/index`
