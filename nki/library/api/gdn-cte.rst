.. meta::
    :description: Chunked gated delta-rule (GDN) prefill kernel.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.gdn

GDN CTE Kernel API Reference
============================

Chunked gated delta-rule (GDN) prefill kernel.

Computes the gated delta-rule linear-attention recurrence in chunk-parallel form and returns the output hidden states plus the final recurrent state. `algorithm` selects the intra-chunk triangular-inverse method. Intended for prefill of a few hundred tokens and up; the default algorithm's chunk grouping needs at least `group_size` chunks (1024 tokens at the default of 8) to amortize, and its measured advantage grows from 1.8x at S=128 to about 3.7x at S>=2048.

Background
-----------

The ``gdn_cte`` kernel computes the gated delta-rule (GDN) linear-attention recurrence for prefill in chunk-parallel form and returns the output hidden states and the final recurrent state. The ``algorithm`` argument selects the intra-chunk triangular-inverse method.

API Reference
--------------

**Source code for this kernel API can be found at**: `gdn_cte.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/gdn/gdn_cte.py>`_

gdn_cte
^^^^^^^

.. py:function:: gdn_cte(query: nl.NkiTensor, key: nl.NkiTensor, value: nl.NkiTensor, g_log: nl.NkiTensor, beta: nl.NkiTensor, scale: float = 1.0, algorithm: str = DEFAULT_ALGORITHM, mask_pack: Optional[nl.NkiTensor] = None, group_size: int = DEFAULT_GROUP_SIZE, recurrent_state: Optional[nl.NkiTensor] = None) -> tuple[nl.NkiTensor, nl.NkiTensor]

   Chunked gated delta-rule (GDN) prefill kernel.

   :param query: [B, Hk, S, Dk], Query tensor in HBM, TOKEN-major; the permute the matmuls need rides on the load DMA. ITS DTYPE SETS THE COMPUTE DTYPE -- matmul operands, q/k/v tiles and `out`. Must be L2-normalized, see Notes.
   :type query: ``nl.NkiTensor``
   :param key: [B, Hk, S, Dk], Key tensor in HBM, TOKEN-major, L2-normalized, same dtype as `query`.
   :type key: ``nl.NkiTensor``
   :param value: [B, Hv, S, Dv], Value tensor in HBM, same dtype as `query`.
   :type value: ``nl.NkiTensor``
   :param g_log: [B, Hv, S], Per-token log-decay (negative) in HBM, fp32.
   :type g_log: ``nl.NkiTensor``
   :param beta: [B, Hv, S], Per-token update strength in (0, 1) in HBM, fp32.
   :type beta: ``nl.NkiTensor``
   :param scale: Query scaling, typically 1/sqrt(Dk). Applied ON DEVICE.
   :type scale: ``float``
   :param algorithm: Intra-chunk inverse to use, `"schur"` (default) or `"neumann"`. Trace-time constant.
   :type algorithm: ``str``
   :param mask_pack: [N_MASKS, CHUNK, CHUNK] fp32 mask pack in HBM. REQUIRED by `"schur"`; build it with `build_mask_pack()`.
   :type mask_pack: ``Optional[nl.NkiTensor]``
   :param group_size: Chunks per grouped inverse for `"schur"`.
   :type group_size: ``int``
   :param recurrent_state: [B, Hv, Dk, Dv] fp32, OPTIONAL MUTABLE IN/OUT. Supplied, it is aliased to the second output, updated IN PLACE, and its incoming contents are the initial state. Omitted, the kernel allocates the output and starts from a zero state with no load emitted. `"neumann"` supports ONLY the omitted form.
   :type recurrent_state: ``Optional[nl.NkiTensor]``
   :return: [B, Hv, S, Dv], Output hidden states in HBM in the compute dtype, HEAD-major and UN-normalized (no fused output epilogue).
   :rtype: ``nl.NkiTensor``
   :return: [B, Hv, Dk, Dv], Final recurrent state, always fp32 -- the caller's buffer when one was supplied.
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * Dq must equal Dk (they contract), and Dk, Dv <= MAX_HEAD_DIM=128.
   * S must be divisible by 128 for `"schur"` and by 64 for `"neumann"`. This dispatcher enforces only the selected algorithm's constraint.
   * Hv must be divisible by Hk; value head h reads key/query head h // (Hv//Hk). `"neumann"` additionally requires Hk == Hv (no head sharing).
   * q/k MUST arrive L2-normalized. That bounds q.k in [-1, 1], which is what makes bf16 matmul operands safe here; un-normalized inputs put the (I+A) inverse outside bf16's usable range.
   * The recurrent state is fp32 whatever the compute dtype: it accumulates across the whole sequence, so it is the one place a narrow mantissa would compound.

   **Dimensions**:

   * B: Batch size
   * Hk: Key/query head count
   * Hv: Value head count, a multiple of Hk (GQA ratio Hv // Hk)
   * Dk: Key/query/state head dimension (<= 128)
   * Dv: Value/state head dimension

