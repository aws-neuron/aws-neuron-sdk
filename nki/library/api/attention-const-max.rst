.. meta::
    :description: Constant-max attention: softmax(Q@K.T * softmax_scale - softmax_max) @ V.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.attention

Attention Const-Max Kernel API Reference
========================================

Constant-max attention: softmax(Q@K.T * softmax_scale - softmax_max) @ V.

Computes attention assuming a pre-known global softmax max.

Background
-----------

The ``attention_const_max`` kernel computes ``softmax(Q @ K.T * softmax_scale - softmax_max) @ V`` using a caller-supplied global softmax maximum, so the running-max rescaling of online softmax is not needed.

API Reference
--------------

**Source code for this kernel API can be found at**: `attention_const_max.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/attention/attention_const_max.py>`_

attention_const_max
^^^^^^^^^^^^^^^^^^^

.. py:function:: attention_const_max(q_hbm: nl.NkiTensor, k_hbm: nl.NkiTensor, v_hbm: nl.NkiTensor, softmax_max: Union[float, nl.NkiTensor], softmax_scale: Optional[float] = None, q_layout: AttentionLayout = AttentionLayout.HEAD_MAJOR, k_layout: AttentionLayout = AttentionLayout.HEAD_MAJOR) -> nl.NkiTensor

   Constant-max attention: softmax(Q@K.T * softmax_scale - softmax_max) @ V.

   :param q_hbm: ``[N, d, Sq]`` or ``[N, Sq, d]`` bf16 @ HBM, Query.
   :type q_hbm: ``nl.NkiTensor``
   :param k_hbm: ``[N, d, Sk]`` or ``[N, Sk, d]`` bf16 @ HBM, Key.
   :type k_hbm: ``nl.NkiTensor``
   :param v_hbm: [N, Sk, d] bf16 @ HBM, Value
   :type v_hbm: ``nl.NkiTensor``
   :param softmax_max: Upper bound on (scores * scale). Output is mathematically invariant to this value; it only affects numerical precision. A reasonable estimate suffices. May be either: - a Python scalar (compile-time constant, broadcast to all partitions), or - a [128, 1] f32 HBM tensor holding a single broadcast max replicated across the 128 partitions. Use the tensor form to supply a data-dependent max computed at runtime rather than baked in at trace time.
   :type softmax_max: ``Union[float, nl.NkiTensor]``
   :param softmax_scale: Scaling factor applied to Q. Defaults to 1/sqrt(d).
   :type softmax_scale: ``Optional[float]``
   :param q_layout: Q layout. ``SEQ_MAJOR`` for ``[N, Sq, d]``, ``HEAD_MAJOR`` for ``[N, d, Sq]``. Defaults to ``HEAD_MAJOR``.
   :type q_layout: ``AttentionLayout``
   :param k_layout: K layout. ``SEQ_MAJOR`` for ``[N, Sk, d]``, ``HEAD_MAJOR`` for ``[N, d, Sk]``. Defaults to ``HEAD_MAJOR``.
   :type k_layout: ``AttentionLayout``
   :return: [N, Sq, d] bf16 @ HBM, Normalized attention output
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * Requires std(Q) ≤ 1 and std(K) ≤ 1 for bf16 accuracy with softmax_max=4–5. This holds when QK-RMSNorm is applied after projection.
   * A softmax_max that is too low causes exp() overflow (inf/NaN in output). A softmax_max that is too high clips all scores toward zero, losing precision.
   * For inputs with significantly larger std use a standard online-softmax kernel. Choosing softmax_max from QK-RMSNorm weights: When QK-RMSNorm is applied, the max can be derived from the gamma weights: σ = sqrt(sum(γ_q² * γ_k²) / d) softmax_max ≈ 0.88 * σ * sqrt(2 * ln(Sk))     # Gumbel expected row-max As a safety check, compute the hard bound (Cauchy-Schwarz worst case): hard_bound = sqrt(d) * max|γ_q| * max|γ_k| If hard_bound - softmax_max > ~80, the score range is too wide for constant-max and the block should fall back to online-softmax attention.

   **Dimensions**:

   * N: Number of attention heads (batch of heads)
   * d: Head dimension (must be pmax=128)
   * Sq: Query sequence length

