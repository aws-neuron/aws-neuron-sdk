.. meta::
    :description: Kimi Delta Attention block for token generation: hidden states in, attention out.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.attention

KDA Block TKG Kernel API Reference
==================================

Kimi Delta Attention block for token generation: hidden states in, attention out.

Advances the KDA recurrence by n tokens for each of B requests, wrapping the recurrence in the whole surrounding block so no intermediate leaves SBUF: input RMSNorm, the fused input projections, the causal short convolutions with SiLU, the low-rank forget gate, the gated output RMSNorm, and the output projection.

Background
-----------

The ``kda_block_tkg`` kernel implements a Kimi Delta Attention (KDA) block for token generation. It advances the KDA recurrence by n tokens per request and fuses the surrounding block (input RMSNorm, input projections, short convolutions, forget gate, gated output RMSNorm, and output projection) so no intermediate leaves SBUF.

API Reference
--------------

**Source code for this kernel API can be found at**: `kda_block_tkg.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/attention/kda_block_tkg.py>`_

kda_block_tkg
^^^^^^^^^^^^^

.. py:function:: kda_block_tkg(hidden: nl.NkiTensor, gamma_in: nl.NkiTensor, W_in: nl.NkiTensor, W_fb: nl.NkiTensor, W_gb: nl.NkiTensor, conv_w: nl.NkiTensor, A_log: nl.NkiTensor, dt_bias: nl.NkiTensor, gamma_o: nl.NkiTensor, W_o: nl.NkiTensor, out: nl.NkiTensor, S: nl.NkiTensor, conv_state: nl.NkiTensor, scale: nl.NkiTensor, lower_bound: nl.NkiTensor, rms_norm_eps: float = 1e-05, l2_eps: float = 1e-06)

   Kimi Delta Attention block for token generation: hidden states in, attention out.

   :param hidden: Block input in HBM, before the input RMSNorm (the kernel applies it, so the residual the caller adds back is this tensor).
   :type hidden: ``nl.NkiTensor``
   :param Shape: [B, n, H].  Dtype: nl.float32
   :param gamma_in: Input RMSNorm weight. Replicated across ranks.
   :type gamma_in: ``nl.NkiTensor``
   :param Shape: [1, H].  Dtype: nl.float32
   :param W_in: Fused input projection, stored contraction-first so no runtime transpose is needed. Columns are this rank's shard in the order q | k | v | g_a | beta | f_a, with widths dk*num_head, dk*num_head, dv*num_head, dk, num_head, dk. The column order WITHIN each per-head block selects the layout the single transpose below produces, so it is chosen per block: - q, k channel-outer (d*num_head + h) -> transposes to [dk, BxS, num_head] - v     head-outer    (h*dv + d)       -> transposes to [num_head, BxS, dv] beta needs no transpose. f_a and g_a are the two low-rank gate latents: both are dk wide with no head axis, and each is expanded by its own up-projection (W_fb, W_gb). They ride here rather than in their own weights so they see the same normed hidden without a second norm; both are replicated across ranks, unlike the rest.
   :type W_in: ``nl.NkiTensor``
   :param Shape: [H, I].  Dtype: nl.bfloat16 (see the weight dtype note below)
   :param W_fb: Forget-gate up-projection, latent to this rank's per-head channels. Columns are HEAD-outer (h*dk + d), unlike W_in's q/k blocks: that makes each head's dk columns contiguous, so one matmul per head lands the result with dk already on the partition axis and the stage-5 transpose is not needed here.
   :type W_fb: ``nl.NkiTensor``
   :param Shape: [dk, dk * num_head].  Dtype: nl.bfloat16
   :param W_gb: Output-gate up-projection, latent to this rank's per-head channels. Columns are CHANNEL-outer (d*num_head + h), unlike W_fb: the gated norm consumes the gate head-major, so tiling the matmul over dv with num_head contiguous lands [num_head, BxS] per channel directly, with no transpose.
   :type W_gb: ``nl.NkiTensor``
   :param Shape: [dk, dv * num_head].  Dtype: nl.bfloat16
   :param conv_w: Depthwise causal convolution taps for the q, k and v streams, one filter per channel. Tap 0 is the oldest position in the window. Tap-major so that each tap is a contiguous dk*num_head run, loadable as one [dk, num_head] tile. Each stream's channel order matches the W_in block it filters, so streams 0/1 are channel-outer and stream 2 is head-outer.
   :type conv_w: ``nl.NkiTensor``
   :param Shape: [3, K, dk * num_head].  Dtype: nl.bfloat16
   :param A_log: Per-head log-gate parameter; exp(A_log) scales the gate pre-activation. A model parameter, so it carries no batch axis.
   :type A_log: ``nl.NkiTensor``
   :param Shape: [1, num_head].  Dtype: nl.float32
   :param dt_bias: Per-head, per-CHANNEL gate bias, added before the sigmoid. A model parameter, so it carries no batch axis.
   :type dt_bias: ``nl.NkiTensor``
   :param Shape: [dk, num_head].  Dtype: nl.float32
   :param gamma_o: Output RMSNorm weight, applied per head over dv and shared across heads.
   :type gamma_o: ``nl.NkiTensor``
   :param Shape: [1, dv].  Dtype: nl.float32
   :param W_o: Output projection, stored contraction-first (dv * num_head on the partition axis) so no runtime transpose is needed.
   :type W_o: ``nl.NkiTensor``
   :param Shape: [dv * num_head, H].  Dtype: nl.bfloat16
   :param out: Caller-allocated destination for this rank's output projection PARTIAL. Not a usable activation until the caller reduces it across the TP group; the residual it is added against is ``hidden``.
   :type out: ``nl.NkiTensor``
   :param Shape: [B, n, H].  Dtype: nl.float32
   :param S: Recurrent state carried in from the previous token, in HBM. IN/OUT: each head's [dk, dv] slice is read to seed its step, then overwritten with the advanced state.
   :type S: ``nl.NkiTensor``
   :param Shape: [B, num_head, dk, dv].  Dtype: nl.float32
   :param conv_state: The last K-1 PRE-convolution projected inputs for each of the q, k and v streams. IN/OUT: read as the left part of this step's convolution window, then overwritten with the window ending at the newest token. Pre-convolution is deliberate -- it is what the filter needs, and it is also the payload a future speculative replay would cache.
   :type conv_state: ``nl.NkiTensor``
   :param Shape: [B, 3, dk * num_head, K - 1].  Dtype: nl.float32
   :param scale: Query scale, normally 1/sqrt(dk); folded into the L2-normed query so the output needs no post-scale.
   :type scale: ``nl.NkiTensor``
   :param Shape: [1, 1].  Dtype: nl.float32
   :param rms_norm_eps: Epsilon for the input RMSNorm and the gated output RMSNorm. The reference implementation drives both from a single ``config.rms_norm_eps`` (1e-5 for Kimi K3), so one value covers them.
   :type rms_norm_eps: ``float``
   :param lower_bound: Gate lower bound, expected in [-5, 0] (-5 for Kimi K3). The activated gate is lower_bound * sigmoid(...), hence <= 0, so exp(gate) is in (0, 1] and the decay can never amplify the state. Also what bounds the cumulative decay range that the chunkwise prefill kernel depends on.
   :type lower_bound: ``nl.NkiTensor``
   :param Shape: [1, 1].  Dtype: nl.float32
   :return: The caller-provided output tensor, filled with this rank's output projection partial for every request in the batch. Requires a TP reduction before use.
   :rtype: ``nl.NkiTensor``
   :return: [B, n, H].  Dtype: nl.float32
   :rtype: ``nl.NkiTensor``
   :return: The caller-provided state tensor, advanced in place by n tokens for every request.
   :rtype: ``nl.NkiTensor``
   :return: [B, num_head, dk, dv].  Dtype: nl.float32
   :rtype: ``nl.NkiTensor``
   :return: The caller-provided convolution windows, advanced in place by n tokens.
   :rtype: ``nl.NkiTensor``
   :return: [B, 3, dk * num_head, K - 1].  Dtype: nl.float32
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * B: Number of requests in the batch (unconstrained)
   * n: Active tokens per request. Must be 1 today; the axis exists so a speculative verify/replay window can be added without an API change.
   * H: Model hidden size
   * num_head: Number of value heads held by THIS rank (num_head_global // TP). Must be a multiple of _HEAD_GROUP, which caps TP at 12 for Kimi K3 (96 heads): TP in {1, 2, 3, 4, 6, 12}.
   * dk: Key/query head dimension, <= 128
   * dv: Value head dimension, <= 128. Equal to dk for KDA.
   * K: Short-convolution kernel size (4 for Kimi K3)

