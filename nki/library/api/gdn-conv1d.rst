.. meta::
    :description: Prefill depthwise causal conv1d + silu + conv_state update.
    :date-modified: 10/19/2026

.. currentmodule:: nkilib.experimental.gdn

GDN Conv1D Kernel API Reference
===============================

Prefill depthwise causal conv1d + silu + conv_state update.

The new ``conv_state`` is the last K tokens of the raw (pre-convolution) mixed tensor.

Background
-----------

The ``gdn_conv1d_prefill`` and ``gdn_conv1d_decode`` kernels compute the depthwise causal conv1d, SiLU activation, and ``conv_state`` update that precede the Qwen3.5 gated delta-net (GDN) linear attention.

API Reference
--------------

**Source code for this kernel API can be found at**: `gdn_conv1d.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/gdn/gdn_conv1d.py>`_

gdn_conv1d_prefill
^^^^^^^^^^^^^^^^^^

.. py:function:: gdn_conv1d_prefill(mixed: nl.NkiTensor, conv_weight: nl.NkiTensor) -> Tuple[nl.NkiTensor, nl.NkiTensor]

   Prefill depthwise causal conv1d + silu + conv_state update.

   :param mixed: [B, C, T] @ HBM, RAW pre-conv input (q|k|v concatenated on the channel axis).
   :type mixed: ``nl.NkiTensor``
   :param conv_weight: [C, K] @ HBM, depthwise weights (self.conv1d.weight squeezed).
   :type conv_weight: ``nl.NkiTensor``
   :return: [B, C, T] @ HBM, silu(causal_conv1d(mixed)).
   :rtype: ``nl.NkiTensor``
   :return: [B, C, K] @ HBM, last K RAW tokens of mixed.
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * Depthwise (groups == C); each channel is an independent FIR.
   * Channels map to the partition axis (128-wide tiles), T to the free axis.

   **Dimensions**:

   * B: Batch size
   * C: Number of channels (== conv_dim, e.g. 8192)
   * T: Number of prefill tokens

gdn_conv1d_decode
^^^^^^^^^^^^^^^^^

.. py:function:: gdn_conv1d_decode(x: nl.NkiTensor, conv_state: nl.NkiTensor, conv_weight: nl.NkiTensor) -> Tuple[nl.NkiTensor, nl.NkiTensor]

   Decode-step depthwise causal conv1d update + silu.

   :param x: [B, C, 1] @ HBM, current-token RAW mixed (q|k|v).
   :type x: ``nl.NkiTensor``
   :param conv_state: [B, C, K] @ HBM, previous conv window.
   :type conv_state: ``nl.NkiTensor``
   :param conv_weight: [C, K] @ HBM, depthwise weights.
   :type conv_weight: ``nl.NkiTensor``
   :return: [B, C, 1] @ HBM, silu(conv1d(sliding window)).
   :rtype: ``nl.NkiTensor``
   :return: [B, C, K] @ HBM, shifted window (== hs).
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * Single output position; the FIR reduces to a K-tap dot product.
   * Depthwise (groups == C); channels map to the partition axis.

   **Dimensions**:

   * B: Batch size
   * C: Number of channels (== conv_dim, e.g. 8192)

