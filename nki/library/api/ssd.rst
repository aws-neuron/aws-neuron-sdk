.. meta::
    :description: State Space Duality (SSD) scan for Mamba-2 models.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.scan

Ssd Kernel API Reference
========================

State Space Duality (SSD) scan for Mamba-2 models.

Performs chunk-wise parallel computation combining TensorE matmuls (intra-chunk structured attention) with VectorE cumulative sums (decay computation). For each chunk of size Q: 1. Cumulative decay: cs = cumsum(dt * A) 2. Intra-chunk: Y_intra = exp(cs) * ((CB * causal) @ (exp(-cs) * dt * x)) where CB = C @ B^T is the structured attention matrix 3. State-to-output: Y_off = exp(cs) * (C @ state) 4. State update: state = exp(cs[-1]) * state + B^T @ (dt * x * exp(cs[-1] - cs)) 5. Output: y = Y_intra + Y_off [+ D * x]

Background
-----------

The ``ssd`` kernel implements the State Space Duality (SSD) scan for Mamba-2 models, combining chunk-wise parallel TensorE matmuls for intra-chunk structured attention with VectorE cumulative sums for decay computation.

API Reference
--------------

**Source code for this kernel API can be found at**: `ssd.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/scan/ssd.py>`_

ssd
^^^

.. py:function:: ssd(x: nl.ndarray, dt: nl.ndarray, A: nl.ndarray, B: nl.ndarray, C: nl.ndarray, chunk_size: int = 128, D: nl.ndarray = None, initial_state: nl.ndarray = None, causal_mask: nl.ndarray = None) -> tuple

   State Space Duality (SSD) scan for Mamba-2 models.

   :param x: [batch, nheads, seqlen, headdim], Input tensor.
   :type x: ``nl.ndarray``
   :param dt: [batch, nheads, seqlen], Timestep tensor. Should be positive.
   :type dt: ``nl.ndarray``
   :param A: [nheads], State transition scalar per head. Should be negative.
   :type A: ``nl.ndarray``
   :param B: [batch, seqlen, dstate], Input projection. Shared across heads.
   :type B: ``nl.ndarray``
   :param C: [batch, seqlen, dstate], Output projection. Shared across heads.
   :type C: ``nl.ndarray``
   :param chunk_size: Chunk size Q. Must be <= 128 (compile-time constant).
   :type chunk_size: ``int``
   :param D: [nheads], Skip connection weights. Default: None.
   :type D: ``nl.ndarray``
   :param initial_state: [batch, nheads, dstate, headdim], Initial hidden state. Default: None (zeros).
   :type initial_state: ``nl.ndarray``
   :param causal_mask: [Q, Q], Lower triangular mask. Required. Pass np.tril(np.ones((Q, Q), dtype=np.float32)).
   :type causal_mask: ``nl.ndarray``
   :return: (y, final_state) - y (nl.ndarray): [batch, nheads, seqlen, headdim], Output tensor with same dtype as x. - final_state (nl.ndarray): [batch, nheads, dstate, headdim], Final hidden state in float32.
   :rtype: ``nl.ndarray``

   **Notes**:

   * chunk_size <= 128 (must fit in partition dimension)
   * dstate <= 128 (for nc_transpose and matmul stationary free dim)
   * headdim <= 512 (PSUM free dimension limit on gen2/3)
   * seqlen must be divisible by chunk_size
   * ngroups=1 (B/C shared across all heads)
   * Uses float32 accumulation internally for numerical stability
   * A should be negative for stable dynamics (decay < 1)
   * dt should be positive; discretization computes exp(dt * A)
   * Inter-chunk state propagation is sequential; intra-chunk uses matmuls

   **Dimensions**:

   * batch: Batch size
   * nheads: Number of attention heads
   * seqlen: Sequence length (must be divisible by chunk_size)
   * headdim: Head dimension (<= 512 for gen2/3 PSUM free dim limit)
   * dstate: SSM state dimension (<= 128 for nc_transpose and matmul)

