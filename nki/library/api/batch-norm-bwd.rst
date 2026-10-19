.. meta::
    :description: Backward pass of batch normalization over the channel dimension.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.batch_norm

Batch Norm Backward Kernel API Reference
========================================

Backward pass of batch normalization over the channel dimension.

Consumes the per-channel statistics saved by the batchnorm forward (e.g. the fused batchnorm of conv3d) and produces the input gradient plus the affine-parameter gradients. Reductions are over the batch and all spatial dimensions, so every output gradient of the affine parameters is per-channel.

Background
-----------

The ``batch_norm_bwd`` kernel computes the backward pass of batch normalization over the channel dimension. It consumes the per-channel statistics saved by the forward pass (for example, the fused batch norm in ``conv3d``) and produces the input gradient and the per-channel affine-parameter gradients.

API Reference
--------------

**Source code for this kernel API can be found at**: `batch_norm_bwd.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/batch_norm/batch_norm_bwd.py>`_

batch_norm_bwd
^^^^^^^^^^^^^^

.. py:function:: batch_norm_bwd(dy: nl.NkiTensor, x_in: nl.NkiTensor, means: nl.NkiTensor, variances: nl.NkiTensor, gamma: nl.NkiTensor, beta: nl.NkiTensor, eps: float = 1e-05, fuse_relu: Optional[bool] = False, relu_mask: Optional[nl.NkiTensor] = None) -> tuple

   Backward pass of batch normalization over the channel dimension.

   :param dy: [B, C, D, H, W], gradient of the loss w.r.t. the batchnorm output, on HBM.
   :type dy: ``nl.NkiTensor``
   :param x_in: [B, C, D, H, W], the forward pass input to the batchnorm (i.e. the pre-normalization activations), on HBM.
   :type x_in: ``nl.NkiTensor``
   :param means: [C], float32 per-channel mean saved by the forward pass.
   :type means: ``nl.NkiTensor``
   :param variances: [C], float32 per-channel biased variance (correction 0) saved by the forward pass.
   :type variances: ``nl.NkiTensor``
   :param gamma: [C, 1], float32 per-channel batchnorm scale.
   :type gamma: ``nl.NkiTensor``
   :param beta: [C, 1], float32 per-channel batchnorm shift. dbeta does not depend on its value, so it is only read when fuse_relu is True and relu_mask is
   :type beta: ``nl.NkiTensor``
   :param None: the batchnorm forward output is then recomputed to find where the fused ReLU clipped, and beta is its bias.
   :param eps: Added to the saved variance before the reciprocal square root, guarding a zero-variance channel against a divide by zero. Must match the eps the forward pass used: a different value normalizes x_hat by a different scale than the forward did, which makes every returned gradient wrong rather than merely imprecise. Defaults to 1e-5, matching torch.nn.BatchNorm2d/3d and conv3d's fused forward.
   :type eps: ``float``
   :param fuse_relu: When True, the forward pass applied a ReLU after the batchnorm, so the incoming gradient is zeroed wherever that ReLU output was zero.
   :type fuse_relu: ``Optional[bool]``
   :param relu_mask: [B, C, D, H, W], uint8 mask of the fused ReLU (1 where the forward ReLU passed its input through, 0 where it clipped). Optional even when fuse_relu is True: leaving it None makes the kernel recompute the batchnorm forward output from x_in and the saved statistics and mask on its sign, trading HBM traffic (a whole [B, C, D, H, W] tensor no longer read) for one extra activation per tile. Ignored when fuse_relu is False.
   :type relu_mask: ``Optional[nl.NkiTensor]``
   :return: [B, C, D, H, W], gradient w.r.t. x_in, same dtype as dy.
   :rtype: ``nl.NkiTensor``
   :return: [C, 1], float32 gradient w.r.t. gamma.
   :rtype: ``nl.NkiTensor``
   :return: [C, 1], float32 gradient w.r.t. beta.
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * A recomputed ReLU mask (fuse_relu=True with relu_mask=None) is not bit-exact against a float32 reference. The engines apply a per-channel scale/bias operand at about 2^-18 relative rather than float32's 2^-24, so an x_in lying closer than that to a channel's ReLU threshold can be masked the opposite way. Measured on trn2 at B=64 C=64 64x64: x_in * true_gamma for one channel returned 0.50649989 against 0.50650179 exact, flipping 179 of 16.7M dx elements. A bf16 destination, a float32 one and a cancellation-free comparison all flip the same elements, so the loss is upstream of both the destination dtype and the algebraic form.
   * Those elements sit within ~1e-6 of zero, i.e. on the ReLU's kink, where the derivative is undefined and both 0 and 1 are valid subgradients. Pass relu_mask to avoid this entirely; a supplied mask is exact. Precision: The multibuffered per-element intermediates (direct_x, var_x) and the SBUF copy of a loaded ReLU mask are bfloat16; the reductions, the per-channel statistics and all the derived coefficients stay float32. That halves the SBUF footprint of the bulk tiles and the direct_x staging traffic through HBM, at the cost of bf16 rounding (~2^-8 relative) in dx even when dy / x_in are float32. A recomputed ReLU mask (fuse_relu set without a mask) carries one extra caveat: the engines apply a per-channel scale/bias operand at about 2^-18 relative rather than float32's 2^-24, so an x_in sitting closer than that to a channel's ReLU threshold can be masked the opposite way from a float32 reference. Those elements are exactly the ones where the forward output is ~0, i.e. the ReLU's kink, where 0 and 1 are both valid subgradients. A supplied relu_mask is exact and unaffected. See the recomputation in phase 1. The dy' * x_in product is the other and stays float32: dgamma subtracts mean * dbeta from its sum, and those terms nearly cancel when x_in's mean is large relative to its spread, so a bf16 product there loses most of dgamma's significant digits (measured ~20% relative error at mean(x) ~ 5, ~69% at ~20). See the dyx_sb allocation in phase 1. Tiling Strategy: Channels map onto SBUF partitions (128 per tile) and the flattened spatial extent D * H * W onto the free dimension, tiled to f_tile elements. Every reduction is over the free dimension and the batch, i.e. within a partition, so no cross-partition reduction is needed. The kernel is DMA bound, so both phases multibuffer their per-tile SBUF tiles: each phase pre-allocates `interleave` rotating slots (as conv3d does for its input-window and stacked-input buffers) and tile i uses slot i % interleave, so tile i's loads (and its store) can run while tile i-1 is still computing. The depth and f_tile come from _build_memory_config, which keeps f_tile as large as possible (long, contiguous DMAs) and trades interleave depth away first. See BatchNormBwdMemoryConfig. The direct_x slots are shared by the two phases: heap-allocated once, before either phase's stack scope, so they survive the close_scope between them. That lets the last `interleave` tiles of phase 1 skip their HBM store entirely and be consumed straight from SBUF by phase 2, saving one store plus one load per resident tile. Both phases must therefore rotate those slots identically, which the single shared mem_cfg.interleave guarantees by construction. LNC Sharding: Channel tiles first: they are fully independent (every reduction is per-channel), so a core owns its outputs outright and no cross-core communication is needed. When there are fewer channel tiles than cores, shard the batch instead so no core sits idle. Batch sharding costs two cross-core reductions, both in phase 1 and both one float32 column per channel tile: dbeta and sum(dy' * x_in) are reductions over the batch, so each core starts with a partial sum, the pair is exchanged with lnc_sendrecv and added, and then every core finishes dgamma locally from the completed sums. direct_x needs no exchange because it is indexed by batch, so each core stages and reloads only its own slice; the same is true of dx. If neither dimension has n_prgs of work, every core computes everything redundantly and writes identical values. The pass is split into two phases with disjoint working sets, separated by an SBM close_scope so phase 2 reuses phase 1's SBUF: Phase 1 (per channel tile, over the owned batches and all spatial tiles):
   * dy' = dy * relu_mask, in place on the dy tile. With no relu_mask supplied the batchnorm forward output is rebuilt from x_in with one activation (x_in_post_bn = x_in * true_gamma + true_beta) and dy is masked on its sign instead of on a loaded mask.
   * dbeta accumulated in the Scalar Engine's reduce_regs across every tile of the channel tile (identity activation, read out on the last tile) while the Vector Engine computes dy' * x_in
   * direct_x = gamma * rstd * dy' staged to HBM while the Vector Engine reduces dy' * x_in into one partial column per (batch, spatial tile)
   * under batch sharding, dbeta and sum(dy' * x_in) are completed across cores
   * dgamma = (sum(dy' * x_in) - mean * dbeta) * rstd once the partials are complete Phase 2 (per channel tile, reloading x_in and the staged direct_x):
   * var_x = (x_in - mean) * intermediate_scale via one activation (copy with a per-channel scale and bias)
   * dx = direct_x - mean_x - var_x via one scalar_tensor_tensor, stored to HBM

   **Dimensions**:

   * B: Batch size
   * C: Number of channels
   * D: Depth
   * H: Height
   * W: Width

