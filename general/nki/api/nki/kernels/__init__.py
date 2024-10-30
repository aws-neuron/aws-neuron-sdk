import numpy as np
import ml_dtypes

def flash_attn_bwd(q_ref, k_ref, v_ref, o_ref, dy_ref, lse_ref, seed_ref, out_dq_ref, out_dk_ref, out_dv_ref, use_causal_mask=False, mixed_precision=False, dropout_p=0.0, softmax_scale=None):
  r"""
  Flash attention backward kernel. Compute the backward gradients.

  IO tensor layouts:
   - q_ref: shape (bs, nheads, head_size, seq)
   - k_ref: shape (bs, nheads, head_size, seq)
   - v_ref: shape (bs, nheads, head_size, seq)
   - o_ref: shape (bs, nheads, head_size, seq)
   - dy_ref: shape (bs, nheads, head_size, seq)
   - lse_ref: shape (bs, nheads, nl.tile_size.pmax, seq // nl.tile_size.pmax)
   - seed_ref: shape (1,)
   - out_dq_ref: shape (bs, nheads, head_size, seq)
   - out_dk_ref: shape (bs, nheads, head_size, seq)
   - out_dv_ref: shape (bs, nheads, head_size, seq)

  Detailed steps:
    1. D = rowsum(dO ◦ O) (pointwise multiply)

    2. Recompute (softmax(Q^T@K))

      2.1 Q^T@K
      2.2 Scale the QK score
      2.3 Apply causal mask
      2.4 softmax

    3. Compute the gradients of y = score @ V with respect to the loss

    4. Compute the gradients of y = softmax(x)

    5. Compute the gradients of Q^T@K

      4.1 Compute dQ
      4.2 Compute dK
  """
  ...

def flash_fwd(q, k, v, seed, o, lse=None, softmax_scale=None, use_causal_mask=True, mixed_precision=True, dropout_p=0.0, config=None):
  r"""
  Flash Attention Forward kernel

  IO tensor layouts:
    - q: shape   (bs, n_heads, d, seq_q)
    - k: shape   (bs, nk_heads, d, seq_k)
    - v: shape   (bs, nv_heads, d, seq_v) if config.should_transpose_v  else (bs, nv_heads, seq_v, d)
    - seed: shape (1,)
    - o: shape (bs, n_heads, seq_q, d)
    - lse: shape (bs, nheads, nl.tile_size.pmax, seq // nl.tile_size.pmax) if training else None
    - We use seq_q and seq_k just for clarity, this kernel requires seq_q == seq_k

  IO tensor dtypes:
    - This kernel assumes all IO tensors have the same dtype
    - If mixed_percision is True, then all Tensor Engine operation will be performed in
      bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
      will be in the same type as the inputs.

  Compile-time Constants:
    - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
    - mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`, if false, we use same precision as input types
    - causal_mask: flag to set causal masking
    - config: Instance of dataclass :class:`nki.kernels.attention.FlashConfig` with Performance config parameters for flash attention with default values
        seq_tile_size: `default=2048`, size of the kv tile size for attention computation reduction
        training: bool to indicate training vs inference `default=True`

  Performance Notes:
    For better performance, the kernel is tiled to be of size `LARGE_TILE_SZ`, and Flash attention math techniques are applied in unit
    of `LARGE_TILE_SZ`. Seqlen that is not divisible by `LARGE_TILE_SZ` is not supported at the moment.

  GQA support Notes:
    the spmd kernel for launching kernel should be on kv_heads instead of nheads

  Example usage:
    MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
      usage: `flash_fwd[b, h](q, k, v, ...)`
    GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
      usage: `flash_fwd[b, kv_h](q, k, v, ...)`
  """
  ...

def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref, out_ref, use_causal_mask=False, mixed_percision=True):
  r"""
  Fused self attention kernel for small head size Stable Diffusion workload.

  Computes softmax(QK^T)V. Decoder model can optionally include a causal mask
  application. Does not include QKV rojection, output projection, dropout,
  residual connection, etc.

  This kernel is designed to be used for Stable Diffusion models where the
  n_heads is smaller or equal to 128. Assertion is thrown if `n_heads` does
  not satisfy the requirement.

  IO tensor layouts:
   - q_ptr: shape   (bs, n_heads, seq_q)
   - k_ptr: shape   (bs, seq_k, n_heads)
   - v_ptr: shape   (bs, seq_v, n_heads)
   - out_ptr: shape (bs, seq_q, n_heads)
   - We use seq_q and seq_k just for clarity, this kernel requires seq_q == seq_k

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - If mixed_percision is True, then all Tensor Engine operation will be performed in
     bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
     will be in the same type as the inputs.
  """
  ...

def resize_nearest_fixed_dma_kernel(data_tensor, out_tensor):
  r"""
  Resize the input image to the given size using the nearest interpolation mode. This kernel is designed to be used when the scaling factor is not an integer. 

  Example:
   - Input height : 30, Input width : 20
   - Output height : 59, Output width : 38

  IO tensor layouts:
   - data_tensor: shape   (in_b, in_h, in_w, in_c)
   - out_tensor: shape   (out_b, out_h, out_w, out_c)
   - b : batch, c : channel, h : height, w : width
   - This kernel requires in_b == out_b as input batch and output batch must be identical
   - This kernel requires in_c == out_c as input channel and output channel must be identical
  
  """
  ...

def select_and_scatter_kernel(operand_tensor, source_tensor, out_tensor):
  r"""
  Implementation of a select-and-scatter kernel.

  It selects an element from each window of operand_tensor, and then scatters
  source_tensor to the indices of the selected positions to construct out_tensor
  with the same shape as the operand_tensor.

  This kernel assumes that
   - windows dimensions:  (3, 3)
   - windows strides:     (2, 2)
   - padding:             (1, 1)
   - init value:          0
   - select computation:  greater-than
   - scatter computation: add

  IO Tensor layouts:
   - operand_tensor: shape   (n, c, h, w)
   - source_tensor : shape   (n, c, src_h, src_w)
   - out_tensor    : shape   (n, c, h, w)
  
  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
  """
  ...

