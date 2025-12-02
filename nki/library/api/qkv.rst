.. meta::
    :description: API reference for the QKV kernel included in the NKI Library .
    :date-modified: 11/12/2025

.. currentmodule:: nkilib.core.qkv

QKV Kernel API Reference
==================================

This topic provides the API reference for the ``QKV`` kernel. The kernel performs Query-Key-Value projection with optional normalization fusion.

The kernel supports:

* Optional RMSNorm/LayerNorm fusion
* Multiple output tensor layouts
* Residual connections from previous MLP and attention outputs
* Automatic selection between TKG and CTE implementations based on batch_size * seqlen threshold
* Optional RoPE (Rotary Position Embedding) fusion

Background
-----------

The ``QKV`` kernel is a critical component in transformer architectures, responsible for projecting the input hidden states into query, key, and value representations. This kernel optimizes the projection operation by fusing it with optional normalization and supporting various output layouts to accommodate different transformer implementations.

.. note::
    This kernel automatically selects between TKG (Token Generation) and CTE (Compute Tensor Engine) implementations based on sequence length threshold (currently 96), ensuring optimal performance across different use cases. CTE is used for longer sequences, while TKG is optimized for shorter sequences.

API Reference
--------------

**Source code for this kernel API can be found at**: https://github.com/aws-neuron/nki-library

qkv
^^^

.. py:function:: qkv(input: nl.ndarray, fused_qkv_weights: nl.ndarray, output_layout: QKVOutputLayout = QKVOutputLayout.BSD, bias: Optional[nl.ndarray] = None, fused_residual_add: Optional[bool] = False, mlp_prev: Optional[nl.ndarray] = None, attention_prev: Optional[nl.ndarray] = None, fused_norm_type: NormType = NormType.NO_NORM, gamma_norm_weights: Optional[nl.ndarray] = None, layer_norm_bias: Optional[nl.ndarray] = None, norm_eps: Optional[float] = 1e-6, hidden_actual: Optional[int] = None, fused_rope: Optional[bool] = False, cos_cache: Optional[nl.ndarray] = None, sin_cache: Optional[nl.ndarray] = None, d_head: Optional[int] = None, num_q_heads: Optional[int] = None, num_kv_heads: Optional[int] = None, store_output_in_sbuf: bool = False, sbm: Optional[SbufManager] = None, use_auto_allocation: bool = False, load_input_with_DMA_transpose: bool = True)

   QKV (Query, Key, Value) projection kernel with multiple optional fused operations.
    
   Performs matrix multiplication between hidden states and fused QKV weights matrix with optional
   fused operations including residual addition, normalization, bias addition, and RoPE rotation.
   Automatically selects between TKG and CTE implementations based on sequence length threshold.

   :param input: Input hidden states tensor. Shape: [B, S, H] where B=batch, S=sequence_length, H=hidden_dim.
   :type input: ``nl.ndarray``
   :param fused_qkv_weights: Fused QKV weight matrix. Shape: [H, I] where I=fused_qkv_dim=(num_q_heads + 2*num_kv_heads)*d_head.
   :type fused_qkv_weights: ``nl.ndarray``
   :param output_layout: Output tensor layout. QKVOutputLayout.BSD=[B, S, I] or QKVOutputLayout.NBSd=[num_heads, B, S, d_head]. Default: QKVOutputLayout.BSD.
   :type output_layout: ``QKVOutputLayout``
   :param bias: Bias tensor to add to QKV projection output. Shape: [1, I].
   :type bias: ``nl.ndarray``, optional
   :param fused_residual_add: Whether to perform residual addition: input = input + mlp_prev + attention_prev. Default: False.
   :type fused_residual_add: ``bool``, optional
   :param mlp_prev: Previous MLP output tensor for residual addition. Shape: [B, S, H].
   :type mlp_prev: ``nl.ndarray``, optional
   :param attention_prev: Previous attention output tensor for residual addition. Shape: [B, S, H].
   :type attention_prev: ``nl.ndarray``, optional
   :param fused_norm_type: Type of normalization (NO_NORM, RMS_NORM, RMS_NORM_SKIP_GAMMA, LAYER_NORM). Default: NormType.NO_NORM.
   :type fused_norm_type: ``NormType``
   :param gamma_norm_weights: Normalization gamma/scale weights. Shape: [1, H]. Required for RMS_NORM and LAYER_NORM.
   :type gamma_norm_weights: ``nl.ndarray``, optional
   :param layer_norm_bias: Layer normalization beta/bias weights. Shape: [1, H]. Only for LAYER_NORM.
   :type layer_norm_bias: ``nl.ndarray``, optional
   :param norm_eps: Epsilon value for numerical stability in normalization. Default: 1e-6.
   :type norm_eps: ``float``, optional
   :param hidden_actual: Actual hidden dimension for padded tensors (if H contains padding).
   :type hidden_actual: ``int``, optional
   :param fused_rope: Whether to apply RoPE rotation to Query and Key heads after QKV projection. Default: False.
   :type fused_rope: ``bool``, optional
   :param cos_cache: Cosine cache for RoPE. Shape: [B, S, d_head]. Required if fused_rope=True.
   :type cos_cache: ``nl.ndarray``, optional
   :param sin_cache: Sine cache for RoPE. Shape: [B, S, d_head]. Required if fused_rope=True.
   :type sin_cache: ``nl.ndarray``, optional
   :param d_head: Dimension per attention head. Required for QKVOutputLayout.NBSd and RoPE.
   :type d_head: ``int``, optional
   :param num_q_heads: Number of query heads. Required for RoPE.
   :type num_q_heads: ``int``, optional
   :param num_kv_heads: Number of key/value heads. Required for RoPE.
   :type num_kv_heads: ``int``, optional
   :param store_output_in_sbuf: Whether to store output in SBUF (currently unsupported, must be False). Default: False.
   :type store_output_in_sbuf: ``bool``
   :param sbm: Optional SBUF manager for memory allocation control with pre-specified bounds for SBUF usage.
   :type sbm: ``SbufManager``, optional
   :param use_auto_allocation: Whether to use automatic SBUF allocation. Default: False.
   :type use_auto_allocation: ``bool``
   :param load_input_with_DMA_transpose: Whether to use DMA transpose optimization. Default: True.
   :type load_input_with_DMA_transpose: ``bool``
   :return: QKV projection output tensor with shape determined by output_layout.
   :rtype: ``nl.ndarray``

   **Raises**:

   * **ValueError** – Raised when contract dimension mismatch occurs between ``input`` and ``fused_qkv_weights``.
   * **AssertionError** – Raised when required parameters for fused operations are missing or have incorrect shapes.

Implementation Details
-----------------------

The kernel implementation includes several key optimizations:

1. **Automatic Implementation Selection**: The kernel automatically selects between TKG (Token Generation) and CTE (Compute Tensor Engine) implementations based on sequence length threshold (currently 96). Some features like RoPE fusion and loading input with DMA transpose are only available in CTE mode. TKG mode only supports automatic allocation at the moment.

2. **Fused Operations Support**: 
   
   - **Residual Addition**: Fuses ``input`` + ``mlp_prev`` + ``attention_prev``
   - **Normalization**: Supports RMSNorm, LayerNorm, and ``RMS_NORM_SKIP_GAMMA``
   - **Bias Addition**: Adds bias to QKV projection output
   - **RoPE Fusion**: Applies Rotary Position Embedding to Query and Key heads

3. **Flexible Output Layouts**: Supports BSD (``[B, S, I]``) and NBSd (``[num_heads, B, S, d_head``]) output tensor layouts.

4. **Memory Management**: 
   
   - Optional SBUF manager for controlled memory allocation
   - DMA transpose optimization for weight loading
   - Automatic or manual SBUF allocation modes

5. **Hardware Compatibility**: Supports bf16, fp16, and fp32 data types (fp32 inputs are internally converted to bf16).

6. **Constraints**: 
   
   - H must be ≤ 24576 and divisible by 128
   - I must be ≤ 4096
   - For NBSd output: d_head must equal 128

