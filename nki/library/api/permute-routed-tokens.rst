.. meta::
    :description: Sort tokens by expert and pack hidden states, affinities, and token indices into a [T*K, n_output_cols] buffer.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.subkernels

Permute Routed Tokens Kernel API Reference
==========================================

Sort tokens by expert and pack hidden states, affinities, and token indices into a [T*K, n_output_cols] buffer.

Background
-----------

The ``permute_routed_tokens`` kernel sorts tokens by their assigned expert and packs hidden states, affinities, and token indices into a contiguous [T*K, n_output_cols] buffer for efficient MoE dispatch.

API Reference
--------------

**Source code for this kernel API can be found at**: `permute_routed_tokens.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/subkernels/permute_routed_tokens.py>`_

permute_routed_tokens
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: permute_routed_tokens(hidden_input: nl.ndarray, expert_index: nl.ndarray, expert_affinities_masked: nl.ndarray)

   Sort tokens by expert and pack hidden states, affinities, and token indices into a [T*K, n_output_cols] buffer.

   :param hidden_input: [T, n_input_cols], bf16 or fp8 HBM tensor of hidden states. When hidden states are fp8, each row contains packed scales.
   :type hidden_input: ``nl.ndarray``
   :param expert_index: [T, K], int32 HBM tensor of top-K expert indices per token.
   :type expert_index: ``nl.ndarray``
   :param expert_affinities_masked: [T, E], bf16 HBM tensor of expert affinities, with zeros for non-routed token/expert pairs.
   :type expert_affinities_masked: ``nl.ndarray``
   :return: [T*K, n_output_cols], bf16 or fp8 HBM tensor where each row is [hidden_state, affinity, token_index] sorted by expert index.
   :rtype: ``nl.ndarray``

   **Notes**:

   * Requires T*K ≤ 128 (pmax) and K ∈ {1, 2, 4, 8}.

   **Dimensions**:

   * T: Number of tokens.
   * H: Hidden dimension size.
   * n_input_cols: Number of columns in hidden_input. When hidden_input is bf16, n_cols=H. When hidden_input is fp8, n_cols contains H and may contain additional columns for quantization scales.
   * n_concat_cols: Number of columns corresponding to affinities (bf16) and token index (int32), when viewed as hidden_input dtype.
   * n_output_cols: n_input_cols + n_concat_cols
   * K: Top-K experts per token.

