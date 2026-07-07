.. meta::
    :description: Compute MoE Top-K reduction across sparse all_to_all_v() collective output buffer.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.subkernels

Top-K Reduce Kernel API Reference
=================================

Computes MoE Top-K reduction across sparse ``all_to_all_v()`` collective output buffer.

The kernel supports:

* Gathering scattered rows by packed global token index
* Reduction along the K dimension
* LNC sharding on the H dimension

Background
-----------

The ``topk_reduce`` kernel gathers scattered rows by packed global token index and reduces along the K dimension. It is used to recombine expert outputs after an ``all_to_all_v()`` collective in Mixture of Experts models.

API Reference
--------------

**Source code for this kernel API can be found at**: `topk_reduce.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/subkernels/topk_reduce.py>`_

topk_reduce
^^^^^^^^^^^

.. py:function:: topk_reduce(input: nl.ndarray, T: int, K: int, token_base_index: int = 1)

   Compute MoE Top-K reduction across sparse all_to_all_v() collective output buffer.

   :param input: [TK_padded, H + 2]@HBM, bf16/fp16. Sparse input buffer containing T*K scattered outputs. Global token index is packed as int32 in the final 2x columns of each row.
   :type input: ``nl.ndarray``
   :param T: Total number of input tokens.
   :type T: ``int``
   :param K: Number of routed experts per token.
   :type K: ``int``
   :param token_base_index: Base offset for the packed global token indices (1-indexed by default). Use to handle sparse K.
   :type token_base_index: ``int``
   :return: [T, H]@HBM, bf16/fp16. Ordered and reduced output.
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * TK_padded: n_src_ranks * T, padded input row count
   * H: Hidden dimension size (must be divisible by LNC)
   * T: Total number of input tokens (up to 128)

