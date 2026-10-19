.. meta::
    :description: Kimi Delta Attention (KDA) token generation: output + new state.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.attention

KDA TKG Kernel API Reference
============================

Kimi Delta Attention (KDA) token generation: output + new state.

Advances the KDA recurrence by exactly one token for each of B requests in a single invocation.

Background
-----------

The ``kda_tkg`` kernel implements Kimi Delta Attention (KDA) token generation. It advances the KDA recurrence by one token for each request and returns the attention output and the updated recurrent state.

API Reference
--------------

**Source code for this kernel API can be found at**: `kda_tkg.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/attention/kda_tkg.py>`_

kda_tkg
^^^^^^^

.. py:function:: kda_tkg(q: nl.NkiTensor, k: nl.NkiTensor, v: nl.NkiTensor, g: nl.NkiTensor, beta: nl.NkiTensor, a_log: nl.NkiTensor, dt_bias: nl.NkiTensor, out: nl.NkiTensor, S: nl.NkiTensor, scale: nl.NkiTensor, lower_bound: nl.NkiTensor)

   Kimi Delta Attention (KDA) token generation: output + new state.

   :param q: Query tensor in HBM for the current token, raw (pre-L2-norm, pre-scale).
   :type q: ``nl.NkiTensor``
   :param Shape: [B, dk, num_head].  Dtype: nl.float32
   :param k: Key tensor in HBM for the current token, raw (pre-L2-norm).
   :type k: ``nl.NkiTensor``
   :param Shape: [B, dk, num_head].  Dtype: nl.float32
   :param v: Value tensor in HBM for the current token.
   :type v: ``nl.NkiTensor``
   :param Shape: [B, num_head, dv].  Dtype: nl.float32
   :param g: Per-channel forget-gate tensor in HBM, raw (pre-activation; the kernel applies lower_bound * sigmoid(exp(a_log) * (g + dt_bias))).
   :type g: ``nl.NkiTensor``
   :param Shape: [B, dk, num_head].  Dtype: nl.float32
   :param beta: Delta-rule beta LOGITS in HBM (the kernel applies the sigmoid).
   :type beta: ``nl.NkiTensor``
   :param Shape: [B, 1, num_head].  Dtype: nl.float32
   :param a_log: Per-head log-gate parameter in HBM; exp(a_log) scales the gate pre-activation. A model parameter, so it carries no batch axis.
   :type a_log: ``nl.NkiTensor``
   :param Shape: [1, num_head].  Dtype: nl.float32
   :param dt_bias: Per-head, per-CHANNEL gate bias in HBM, added to g before the sigmoid. A model parameter, so it carries no batch axis.
   :type dt_bias: ``nl.NkiTensor``
   :param Shape: [dk, num_head].  Dtype: nl.float32
   :param out: Caller-allocated destination for this token's attention output.
   :type out: ``nl.NkiTensor``
   :param Shape: [B, num_head, dv].  Dtype: nl.float32
   :param S: Recurrent state carried in from the previous token, in HBM. IN/OUT: each item's [dk, dv] slice is read to seed its step, then overwritten with the advanced state.
   :type S: ``nl.NkiTensor``
   :param Shape: [B, num_head, dk, dv].  Dtype: nl.float32
   :param scale: Query scale, normally 1/sqrt(dk); folded into the L2-normed query so the output needs no post-scale.
   :type scale: ``nl.NkiTensor``
   :param Shape: [1, 1].  Dtype: nl.float32
   :param lower_bound: Gate lower bound, expected in [-5, 0]. The activated gate is lower_bound * sigmoid(...), hence <= 0, so exp(gate) is in (0, 1] and the decay can never amplify the state.
   :type lower_bound: ``nl.NkiTensor``
   :param Shape: [1, 1].  Dtype: nl.float32
   :return: The caller-provided output tensor, filled with this token's attention output for every request in the batch.
   :rtype: ``nl.NkiTensor``
   :return: [B, num_head, dv].  Dtype: nl.float32
   :rtype: ``nl.NkiTensor``
   :return: The caller-provided state tensor, advanced in place by one token for every request.
   :rtype: ``nl.NkiTensor``
   :return: [B, num_head, dk, dv].  Dtype: nl.float32
   :rtype: ``nl.NkiTensor``

   **Dimensions**:

   * B: Number of requests in the batch (unconstrained)
   * num_head: Number of value heads
   * dk: Key/query head dimension, <= 128

