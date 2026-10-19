.. meta::
    :description: Fused single-token gated delta-rule decode step.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.gdn

GDN TKG Kernel API Reference
============================

Fused single-token gated delta-rule decode step.

Folds l2norm(q,k), beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias), and RMSNormGated(out, z) around the recurrent delta-rule core, then advances the per-head recurrent state by one token.

Background
-----------

The ``gdn_tkg`` kernel computes a single-token gated delta-rule (GDN) decode step. It fuses the q/k L2 norm, beta and gate computation, and gated RMSNorm around the recurrence, and advances the per-head recurrent state by one token.

API Reference
--------------

**Source code for this kernel API can be found at**: `gdn_tkg.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/gdn/gdn_tkg.py>`_

gdn_tkg
^^^^^^^

.. py:function:: gdn_tkg(q: nl.NkiTensor, k: nl.NkiTensor, v: nl.NkiTensor, b: nl.NkiTensor, a: nl.NkiTensor, A_log: nl.NkiTensor, dt_bias: nl.NkiTensor, z: nl.NkiTensor, norm_weight: nl.NkiTensor, state_in: nl.NkiTensor) -> Tuple[nl.NkiTensor, nl.NkiTensor]

   Fused single-token gated delta-rule decode step.

   :param q: [BH, K] @ HBM, RAW query (GQA-expanded, not l2-normed).
   :type q: ``nl.NkiTensor``
   :param k: [BH, K] @ HBM, RAW key (GQA-expanded, not l2-normed).
   :type k: ``nl.NkiTensor``
   :param v: [BH, V] @ HBM, value.
   :type v: ``nl.NkiTensor``
   :param b: [BH] @ HBM, RAW beta pre-sigmoid.
   :type b: ``nl.NkiTensor``
   :param a: [BH] @ HBM, RAW gate input (pre-softplus).
   :type a: ``nl.NkiTensor``
   :param A_log: [num_v_heads] @ HBM, per-v-head log decay coefficient.
   :type A_log: ``nl.NkiTensor``
   :param dt_bias: [num_v_heads] @ HBM, per-v-head softplus bias.
   :type dt_bias: ``nl.NkiTensor``
   :param z: [BH, V] @ HBM, RMSNormGated gate input.
   :type z: ``nl.NkiTensor``
   :param norm_weight: [V] @ HBM, RMSNormGated weight (gamma).
   :type norm_weight: ``nl.NkiTensor``
   :param state_in: [BH, K, V] @ HBM, incoming recurrent state.
   :type state_in: ``nl.NkiTensor``
   :return: [BH, V] @ HBM, RMSNormGated decode output (bf16).
   :rtype: ``nl.NkiTensor``
   :return: [BH, K, V] @ HBM, updated recurrent state (fp32).
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * Per-head independent recurrence (no cross-head reduction), so the BH axis is sharded across the SPMD grid; each core writes only its own contiguous head slice. LNC1 is byte-identical to single-core.
   * Do NOT pre-apply exp(g) or the 1/sqrt(K) Q-scale upstream; the kernel applies both internally.

   **Dimensions**:

   * BH: Flattened batch * num_v_heads (heads GQA-expanded, cycling fastest per batch)
   * B: Batch size = BH // num_v_heads
   * num_v_heads: Number of value heads
   * K: Key/state head dimension (<= 128)

