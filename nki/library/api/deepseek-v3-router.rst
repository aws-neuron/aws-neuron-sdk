.. meta::
    :description: Computes the DeepSeek-V3.2 group-limited ``noaux_tc`` routing decision for one decode step: sigmoid expert scores, a cor
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.deepseek_v32.deepseek_v3_router

DeepSeek-V3 Router Kernel API Reference
=======================================

Computes the DeepSeek-V3.2 group-limited ``noaux_tc`` routing decision for one decode step: sigmoid expert scores, a correction bias applied for selection only, group-limited pruning to ``topk_group`` groups, then the ``top_k`` experts within the survivors, L1-normalized over their pre-bias scores and scaled.

Both operands accept two layouts and the choice is auto-detected rather than flagged. A caller that just produced ``normed`` on chip passes its hidden-major SBUF tile straight in, and the router weight can be pre-transposed once at load time; both skip work described in Notes.

Background
-----------

The ``deepseek_v3_router`` kernel computes the DeepSeek-V3.2 group-limited ``noaux_tc`` routing decision for one decode step: sigmoid expert scores, a selection-only correction bias, pruning to ``topk_group`` groups, and the ``top_k`` experts among the survivors, L1-normalized and scaled.

API Reference
--------------

**Source code for this kernel API can be found at**: `deepseek_v3_router.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/deepseek_v32/deepseek_v3_router/deepseek_v3_router.py>`_

deepseek_v3_router
^^^^^^^^^^^^^^^^^^

.. py:function:: deepseek_v3_router(normed: nl.NkiTensor, router_weight: nl.NkiTensor, e_score_correction_bias: nl.NkiTensor, n_group: int, topk_group: int, top_k: int, routed_scaling_factor: float, outputs: RouterOutputs = RouterOutputs.DENSE, output_in_sbuf: bool = False)

   Computes the DeepSeek-V3.2 group-limited ``noaux_tc`` routing decision for one decode step: sigmoid expert scores, a correction bias applied for selection only, group-limited pruning to ``topk_group`` groups, then the ``top_k`` experts within the survivors, L1-normalized over their pre-bias scores and scaled.

   :param normed: ``[n_tokens, hidden]`` in HBM, loaded and transposed here, or ``[P_MAX, n_h_tiles, n_tokens]`` already hidden-major in SBUF, where both the load and the transpose are skipped. A bf16 SBUF tile is widened to fp32 on the way in, because ``nc_matmul`` forbids mixing fp32 with a narrower dtype and the router weight is fp32.
   :type normed: ``nl.NkiTensor``
   :param router_weight: ``[n_experts, hidden]`` fp32 row-major, loaded contiguously and PE-transposed here, or ``[P_MAX, n_h_tiles, n_experts]`` pre-transposed in either memory space, which skips every weight transpose. Build the pre-transposed form once at load time as ``w.reshape(n_experts, hidden // P_MAX, P_MAX).transpose(2, 1, 0)``. It must be the 3-D form: pre-transposing to a flat ``[hidden, n_experts]`` would make each partition gather 1 KB runs and reintroduce exactly the degenerate DMA this layout exists to avoid.
   :type router_weight: ``nl.NkiTensor``
   :param e_score_correction_bias: ``[1, n_experts]`` fp32. Added after the sigmoid, for selection only; the returned weights come from the pre-bias scores.
   :type e_score_correction_bias: ``nl.NkiTensor``
   :param n_group: Expert groups that group-limited routing scores and prunes.
   :type n_group: ``int``
   :param topk_group: Groups kept per token.
   :type topk_group: ``int``
   :param top_k: Experts kept per token.
   :type top_k: ``int``
   :param routed_scaling_factor: Final scale applied to the normalized weights.
   :type routed_scaling_factor: ``float``
   :param outputs: Which tensors to return. See Returns.
   :type outputs: ``RouterOutputs``
   :param output_in_sbuf: When True the results are returned as the SBUF tiles they were computed in, with no HBM allocation and no store, so a fusing caller reads them in place. Only legal when this kernel is inlined into a larger one, since a top-level ``@nki.jit`` entry must return HBM tensors for the runtime to read.
   :type output_in_sbuf: ``bool``

   **Notes**:

   * The router matmul runs in fp32 regardless of the input dtype. Its logits feed a discrete selection that is tie-sensitive, so bf16 rounding can flip which experts or groups are chosen relative to the fp32 reference. Everything downstream is fp32 already. Both operands are loaded in their natural layout and transposed on the PE rather than gathered into hidden-major by the DMA. Asking the DMA for that layout directly makes each destination partition gather stride-hidden elements, degenerating to one descriptor per element: the ``[256, 7168]`` fp32 weight measured 1123 us at 6.5 GB/s, 1.2% of trn3's 528 GB/s, and was 90% of the whole kernel. ``dma_transpose`` cannot replace it either, since that path needs a 2-byte dtype. The logits matmul tiles the PE array's columns. See ``pe_column_tile`` on the shape object for why, and ``core/router_topk/router_topk.py`` for the same mechanism.

   **Dimensions**:

   * n_tokens: Decode tokens in the batch. Must be <= P_MAX.
   * hidden: Model hidden width, the contraction dimension. Must be a multiple of P_MAX.
   * n_experts: Routed experts. Must be <= 512 and divide evenly into n_group.
   * n_group: Expert groups. Must be <= MAX8_WIDTH.
   * topk_group: Groups kept per token. Must be <= MAX8_WIDTH.

