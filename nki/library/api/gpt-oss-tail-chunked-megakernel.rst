.. meta::
    :description: GPT-OSS decode sampling tail (RMSNorm, lm_head, top-k, sampler) with a single collective.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.gpt_oss_tail

GPT-OSS Tail Chunked Megakernel Kernel API Reference
====================================================

GPT-OSS decode sampling tail (RMSNorm, lm_head, top-k, sampler) with a single collective.

Background
-----------

The ``gpt_oss_tail_chunked_megakernel`` kernel implements the GPT-OSS decode sampling tail (final RMSNorm, ``lm_head``, top-k, and sampler) with a single all-to-all collective. It brings the full vocabulary onto each rank and computes the global top-k locally with ``chunked_topk``.

API Reference
--------------

**Source code for this kernel API can be found at**: `tail_chunked_megakernel.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/gpt_oss_tail/tail_chunked_megakernel.py>`_

gpt_oss_tail_chunked_megakernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: gpt_oss_tail_chunked_megakernel(hidden_states: nl.NkiTensor, gamma: nl.NkiTensor, lm_head_weight_t: nl.NkiTensor, eps: float = 1e-05, hidden_actual: Optional[int] = None, zero_interleaved_pad: bool = True, k: int = 256, sampling_params: Optional[nl.NkiTensor] = None, replica_groups: Optional[tuple] = None, hop_meshes: Optional[tuple] = None, dummy_collectives: bool = False, emit_logits: bool = True, debug_taps: bool = False, deferred_rms: bool = True, skip_nan_fold: bool = True, a2a_batch_groups: int = 1) -> Tuple[nl.NkiTensor, ...]

   GPT-OSS decode sampling tail (RMSNorm, lm_head, top-k, sampler) with a single collective.

   :param hidden_states: ``[gbs, H]`` post-MoE hidden states in HBM; fp32, fp16, or bf16. A 16-bit input is widened to fp32 through an explicit on-chip conversion copy immediately after its (half-sized) load, which is EXACT (both half formats are subsets of fp32) -- so it produces bitwise the same outputs as passing a pre-widened fp32 copy, while halving this kernel's largest input DMA and letting a 16-bit producer skip an HBM widen round trip. The copy is load-bearing: engines fed 16-bit data directly compute at input precision.
   :type hidden_states: ``nl.NkiTensor``
   :param gamma: ``[1, H]`` final RMSNorm weight, fp32.
   :type gamma: ``nl.NkiTensor``
   :param lm_head_weight_t: ``[H, V_s]`` **bfloat16** lm_head shard for this rank, ALREADY TRANSPOSED and C-contiguous. ``V_s = V/VTP``. bf16 because that is the checkpoint's own precision.
   :type lm_head_weight_t: ``nl.NkiTensor``
   :param eps: RMSNorm epsilon (1e-5 for GPT-OSS).
   :type eps: ``float``
   :param hidden_actual: ``H_u``, the unpadded hidden (2880). Defaults to H.
   :type hidden_actual: ``Optional[int]``
   :param zero_interleaved_pad: True for the mxfp4 checkpoint (interleaved padding lanes).
   :type zero_interleaved_pad: ``bool``
   :param k: candidates kept (256 in production).
   :type k: ``int``
   :param emit_logits: whether to return the fp32 ``[gbs, V_s]`` pre-collective logits. They are a DEBUG/GATING output -- nothing downstream of this kernel reads them; the collective carries the bf16 mirror instead. Keeping them costs a full extra HBM write of the shard (805 KB at the production shape, roughly half this kernel's hbm_write) plus its DMA and the fp32 SBUF tile. The tests set it True so the norm and the lm_head GEMM stay independently gated; a deployment that only wants tokens should set it False.
   :type emit_logits: ``bool``
   :param sampling_params: ``[gbs/VTP, 3]`` or ``[gbs/VTP, 4]`` fp32. Columns are ``[top_k, top_p, temperature]``; the optional 4th column is the per-row inverse-transform draw ``u`` in ``[0, 1)`` -- the RUNTIME form of the compiled-in ``_DETERMINISTIC_DRAW`` (0.5), so sweeping the draw is a data change rather than a NEFF rebuild. Rows are THIS RANK'S only -- see ``owned_row_block`` for which rows those are when more than one hop is used. When None the sampler is skipped and no token is returned.
   :type sampling_params: ``Optional[nl.NkiTensor]``
   :param replica_groups: one group listing all ``VTP`` ranks IN RANK ORDER. Shorthand for a single-hop ``hop_meshes=(replica_groups,)``; supply exactly one of the two.
   :type replica_groups: ``Optional[tuple]``
   :param hop_meshes: one replica-group mesh per all-to-all hop, from ``compose_hops``. Hop sizes must multiply to ``VTP``. Needed because an all-to-all replica group must be device-local or one-core-per-device.
   :type hop_meshes: ``Optional[tuple]``
   :param dummy_collectives: prepend a zero all-reduce fence per hop mesh so the fence, not the real hops, absorbs rank-arrival skew. A PROFILING instrument: it cannot change the answer but it adds a collective per mesh to every call, so it defaults off and profiling runs opt in. Without it a per-hop duration is mostly dispatch jitter.
   :type dummy_collectives: ``bool``
   :param debug_taps: forward ``chunked_topk``'s seven diagnostic taps as extra outputs, APPENDED after everything else so the existing output positions do not move. A DIAGNOSTIC instrument for hardware-only index defects -- see DEBUG TAPS in ``chunked_topk``. It changes this kernel's arity, so leave it off unless the consumer expects them.
   :type debug_taps: ``bool``
   :param deferred_rms: apply the per-row RMSNorm scale on the GEMM's PSUM-read copies instead of on the GEMM's input (exact algebra: ``rms`` is a per-row scalar and the GEMM is linear), which stops every matmul from waiting on the full row's sum of squares -- measured -3.8 us on the fenced gbs=128 head. Numerically the SAME one-bf16-rounding-per-element bound; only the last-ulp realization moves, so fixed-seed sampled tokens can shift at cdf crossings (same class as any resummation change). Default True; a SAFETY FALLBACK to the legacy in-chain form -- ``(x*rms)*gamma`` feeding the GEMM, the pre-deferred realization bit-for-bit -- is one trace-time flag away if a consumer needs it.
   :type deferred_rms: ``bool``
   :param skip_nan_fold: skip ``chunked_topk``'s stage-1 NaN fold. Sound here because the top-k input is this kernel's OWN GEMM output: finite hidden/gamma/weight inputs give finite bf16 products summed in fp32, and even an overflow saturates to +/-inf -- which ``nisa.topk`` orders fine (the fold only exists for NaN, which would take inf arithmetic INSIDE the GEMM, i.e. already-garbage inputs). The fold costs a full-width compare + predicated copy per stage-1 tile on the post-collective serial chain. Default True; the fold is one trace-time flag away for a caller that wants the belt-and-suspenders behavior back. Outputs are BITWISE identical either way on any NaN-free input -- unlike ``deferred_rms`` this cannot move even an ulp, because it deletes ops whose effect on finite data is the identity.
   :type skip_nan_fold: ``bool``
   :param a2a_batch_groups: split the exchange into this many batch-group collectives, one per gbs/G whole lm_head tiles (single-hop only; G must divide rows_per_rank and the geometry must support the receive-layout fold -- asserted; groups aligned to whole 8-row sort tiles overlap best but are not required). Group g's exchange fires as soon as its own lm_head tiles' mirror stores land -- under the rest of the GEMM -- pulling the exchange forward by up to (G-1)/G of its wire time minus (G-1) launch serializations. OWNERSHIP CONTRACT CHANGES with G > 1: rank r owns the STRIDED row set given by ``owned_rows`` (one extra mixed-radix digit), not ``owned_row_block``'s contiguous block -- ``sampling_params`` rows and every consumer of the per-rank outputs must use that map. Gate-validated, not bitwise: regrouping a collective changes the runtime's internal transfer realization.
   :type a2a_batch_groups: ``int``

compose_hops
^^^^^^^^^^^^

.. py:function:: compose_hops(vtp: int, hops: Tuple[int, ...])

   Replica-group meshes for a sequence of all-to-all hops whose sizes multiply to ``vtp``.


owned_row_block
^^^^^^^^^^^^^^^

.. py:function:: owned_row_block(rank: int, vtp: int, gbs: int, hops = None)

   Batch rows that ``rank`` owns after the all-to-all. Returns ``(row_start, n_rows)``.


owned_rows
^^^^^^^^^^

.. py:function:: owned_rows(rank: int, vtp: int, gbs: int, hops = None, batch_groups: int = 1)

   The GLOBAL batch rows ``rank`` owns, IN RECEIVE ORDER, as a list.


