.. meta::
    :description: Transformer token generation forward pass megakernel.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.transformer

Transformer TKG Kernel API Reference
====================================

Implements the transformer token generation forward pass as a single megakernel.

The kernel supports:

* Configurable number of transformer layers
* Per-layer attention block (RMSNorm + QKV + RoPE + Attention + Output Projection)
* Per-layer MLP block (RMSNorm + Gate/Up + Activation + Down Projection)
* All-reduce collective communication between layers
* Residual connections
* Optional FP8 quantization with per-layer weight scales
* SBUF residual path with SB2SB all-reduce

Background
-----------

The ``transformer_tkg`` kernel performs multiple transformer layers in a single kernel invocation for token generation. Within each layer, it executes: attention block, all-reduce, MLP, all-reduce, and residual connections. This reduces kernel launch overhead and enables cross-layer optimizations.

API Reference
--------------

**Source code for this kernel API can be found at**: `transformer_tkg.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/transformer/transformer_tkg.py>`_

transformer_tkg
^^^^^^^^^^^^^^^

.. py:function:: transformer_tkg(X: nl.ndarray, W_qkvs: List[nl.ndarray], W_outs: List[nl.ndarray], W_gates: List[nl.ndarray], W_ups: List[nl.ndarray], W_downs: List[nl.ndarray], W_gamma_qkvs: List[nl.ndarray], W_gamma_mlps: List[nl.ndarray], K_caches: List[nl.ndarray], V_caches: List[nl.ndarray], RoPE_cos: nl.ndarray, RoPE_sin: nl.ndarray, attention_mask: nl.ndarray, position_ids: Optional[nl.ndarray], num_layers: int, eps: float = 1e-06, replica_groups: Optional[List[List[int]]] = None, sbuf_residual_and_cc: bool = False, clamp_bound: float = 0.0, W_gate_scales: Optional[List[nl.ndarray]] = None, W_up_scales: Optional[List[nl.ndarray]] = None, W_down_scales: Optional[List[nl.ndarray]] = None, dtype_mode: DtypeMode = DtypeMode.NON_OCP)

   Transformer token generation forward pass megakernel.

   :param X: [B, S_tkg, H], Input hidden states on HBM
   :type X: ``nl.ndarray``
   :param W_qkvs: Per-layer QKV projection weights
   :type W_qkvs: ``List[nl.ndarray]``
   :param W_outs: Per-layer output projection weights
   :type W_outs: ``List[nl.ndarray]``
   :param W_gates: Per-layer MLP gate projection weights
   :type W_gates: ``List[nl.ndarray]``
   :param W_ups: Per-layer MLP up projection weights
   :type W_ups: ``List[nl.ndarray]``
   :param W_downs: Per-layer MLP down projection weights
   :type W_downs: ``List[nl.ndarray]``
   :param W_gamma_qkvs: Per-layer RMSNorm gamma for QKV
   :type W_gamma_qkvs: ``List[nl.ndarray]``
   :param W_gamma_mlps: Per-layer RMSNorm gamma for MLP
   :type W_gamma_mlps: ``List[nl.ndarray]``
   :param K_caches: Per-layer K caches on HBM
   :type K_caches: ``List[nl.ndarray]``
   :param V_caches: Per-layer V caches on HBM
   :type V_caches: ``List[nl.ndarray]``
   :param RoPE_cos: [d_head//2, B, S_tkg], RoPE cosine embeddings
   :type RoPE_cos: ``nl.ndarray``
   :param RoPE_sin: [d_head//2, B, S_tkg], RoPE sine embeddings
   :type RoPE_sin: ``nl.ndarray``
   :param attention_mask: Attention mask covering both cached KV context and active tokens. Replaces the previous separate ``mask_cache`` and ``mask_active`` parameters.
   :type attention_mask: ``nl.ndarray``
   :param position_ids: [B, 1], KV cache write positions (None = skip cache update)
   :type position_ids: ``Optional[nl.ndarray]``
   :param num_layers: Number of transformer layers to execute
   :type num_layers: ``int``
   :param eps: RMSNorm epsilon (default 1e-6)
   :type eps: ``float``
   :param replica_groups: Replica groups for collective communication
   :type replica_groups: ``Optional[List[List[int]]]``
   :param sbuf_residual_and_cc: Use SBUF residual path with SB2SB all-reduce (default False)
   :type sbuf_residual_and_cc: ``bool``
   :param clamp_bound: FP8 quantization clipping boundary (default 0.0, 0 = no clipping)
   :type clamp_bound: ``float``
   :param W_gate_scales: Per-layer FP8 gate weight scales
   :type W_gate_scales: ``Optional[List[nl.ndarray]]``
   :param W_up_scales: Per-layer FP8 up weight scales
   :type W_up_scales: ``Optional[List[nl.ndarray]]``
   :param W_down_scales: Per-layer FP8 down weight scales
   :type W_down_scales: ``Optional[List[nl.ndarray]]``
   :return: [B, S_tkg, H], Final hidden states after all transformer layers
   :rtype: ``nl.ndarray``

   **Dimensions**:

   * B: Batch size
   * S_tkg: Token generation sequence length (number of new tokens)
   * H: Hidden dimension (must be multiple of 128)
   * H0: Partition tile size (pmax = 128)
   * H1: H // H0

