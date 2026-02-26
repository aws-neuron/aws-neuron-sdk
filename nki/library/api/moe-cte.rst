.. meta::
    :description: MoE CTE kernel implements Mixture of Experts MLP optimized for Context Encoding.
    :date-modified: 02/13/2026

.. currentmodule:: nkilib.core.moe_cte

MoE CTE Kernel API Reference
=============================

Implements Mixture of Experts (MoE) MLP computation optimized for Context Encoding with blockwise matrix multiplication and multiple sharding strategies.

The kernel supports:

* Unified entry point dispatching to multiple implementation variants
* Block-sharding and intermediate-dimension-sharding strategies
* Multiple quantization types (FP8 row/static, MxFP4/MxFP8)
* Expert affinity scaling (pre-scale and post-scale modes)
* Various activation functions (SiLU, GELU, ReLU)
* Optional bias terms for projections
* Clamping for gate and up projections
* Activation checkpointing for gradient computation
* Hybrid static/dynamic loop optimization for padded sequences

Background
--------------

The ``MoE CTE`` kernel is designed for Mixture of Experts models during context encoding (prefill) phase where the sequence length is typically large (T > 128). The kernel performs blockwise MoE MLP computation:

1. **Token Assignment**: Tokens are pre-assigned to blocks via ``token_position_to_id``
2. **Gate Projection**: ``gate_out = hidden @ gate_weights``
3. **Up Projection**: ``up_out = hidden @ up_weights``
4. **Activation**: ``act_gate = activation_fn(gate_out)``
5. **Element-wise Multiply**: ``intermediate = act_gate * up_out``
6. **Down Projection**: ``expert_out = intermediate @ down_weights``
7. **Affinity Scaling**: ``output = expert_out * affinity`` (if enabled)
8. **Block Accumulation**: Results are accumulated across blocks for multi-expert assignments

The unified ``moe_cte`` entry point dispatches to the appropriate implementation based on the ``spec`` parameter, which selects between block-sharding and intermediate-dimension-sharding strategies with optional MX quantization support.

API Reference
----------------

**Source code for this kernel API can be found at**: `moe_cte.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/core/moe/moe_cte/moe_cte.py>`_

moe_cte
^^^^^^^^

.. py:function:: moe_cte(hidden_states: nl.ndarray, expert_affinities_masked: nl.ndarray, gate_up_proj_weight: nl.ndarray, down_proj_weight: nl.ndarray, token_position_to_id: nl.ndarray, block_to_expert: nl.ndarray, block_size: int, spec: MoECTESpec, conditions: Optional[nl.ndarray] = None, gate_and_up_proj_bias: Optional[nl.ndarray] = None, down_proj_bias: Optional[nl.ndarray] = None, quantization_config: Optional[QuantizationConfig] = None, gate_up_activations_T: Optional[nl.ndarray] = None, down_activations: Optional[nl.ndarray] = None, activation_function: ActFnType = ActFnType.SiLU, skip_dma: SkipMode = SkipMode(False, False), compute_dtype=nl.bfloat16, is_tensor_update_accumulating: bool = True, expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE, gate_clamp_upper_limit: Optional[float] = None, gate_clamp_lower_limit: Optional[float] = None, up_clamp_upper_limit: Optional[float] = None, up_clamp_lower_limit: Optional[float] = None)

   Unified entry point for MoE CTE blockwise matrix multiplication kernels.

   Dispatches to the appropriate implementation based on ``spec.implementation``. Supports multiple
   sharding strategies and quantization modes for different hardware targets.

   :param hidden_states: Input hidden states tensor with shape ``[T+1, H]`` in HBM. T+1 because padding token position is set to T.
   :type hidden_states: ``nl.ndarray``
   :param expert_affinities_masked: Expert affinities for each token with shape ``[(T+1) * E, 1]`` in HBM
   :type expert_affinities_masked: ``nl.ndarray``
   :param gate_up_proj_weight: Concatenated gate and up projection weights with shape ``[E, H, 2, I_TP]`` in HBM
   :type gate_up_proj_weight: ``nl.ndarray``
   :param down_proj_weight: Down projection weights with shape ``[E, I_TP, H]`` in HBM
   :type down_proj_weight: ``nl.ndarray``
   :param token_position_to_id: Block index of corresponding tokens with shape ``[N * B]`` in HBM. Includes padding tokens (N * B >= T). Padding token id is set to T.
   :type token_position_to_id: ``nl.ndarray``
   :param block_to_expert: Expert indices of corresponding blocks with shape ``[N, 1]`` in HBM
   :type block_to_expert: ``nl.ndarray``
   :param block_size: Number of tokens per block (must be multiple of 256)
   :type block_size: ``int``
   :param spec: Implementation selection and configuration. Controls which sharding strategy and implementation variant to use. See ``MoECTESpec`` for details.
   :type spec: ``MoECTESpec``
   :param conditions: Block padding indicators with shape ``[N+1]``. Used by hybrid and block_mx implementations to distinguish padded vs non-padded blocks.
   :type conditions: ``nl.ndarray``, optional
   :param gate_and_up_proj_bias: Gate and up projection bias with shape ``[E, 2, I_TP]``. For SiLU, up_bias = up_bias + 1.
   :type gate_and_up_proj_bias: ``nl.ndarray``, optional
   :param down_proj_bias: Down projection bias with shape ``[E, H]``
   :type down_proj_bias: ``nl.ndarray``, optional
   :param quantization_config: Quantization scales configuration containing ``gate_up_proj_scale`` and ``down_proj_scale`` for weight dequantization. See ``QuantizationConfig`` for details.
   :type quantization_config: ``QuantizationConfig``, optional
   :param gate_up_activations_T: Pre-allocated storage for gate/up activations (for activation checkpointing). Used when ``spec.shard_on_I.checkpoint_activation=True``.
   :type gate_up_activations_T: ``nl.ndarray``, optional
   :param down_activations: Pre-allocated storage for down projection activations (for activation checkpointing). Used when ``spec.shard_on_I.checkpoint_activation=True``.
   :type down_activations: ``nl.ndarray``, optional
   :param activation_function: Activation function for MLP block. Default is ``SiLU``.
   :type activation_function: ``ActFnType``
   :param skip_dma: DMA skip mode configuration. Default is ``SkipMode(False, False)``.
   :type skip_dma: ``SkipMode``
   :param compute_dtype: Compute data type. Default is ``nl.bfloat16``.
   :type compute_dtype: ``nl.dtype``
   :param is_tensor_update_accumulating: Whether to accumulate results over multiple blocks. Default is ``True``.
   :type is_tensor_update_accumulating: ``bool``
   :param expert_affinities_scaling_mode: Post or pre scaling mode. Default is ``POST_SCALE``.
   :type expert_affinities_scaling_mode: ``ExpertAffinityScaleMode``
   :param gate_clamp_upper_limit: Upper clamp limit for gate projection
   :type gate_clamp_upper_limit: ``float``, optional
   :param gate_clamp_lower_limit: Lower clamp limit for gate projection
   :type gate_clamp_lower_limit: ``float``, optional
   :param up_clamp_upper_limit: Upper clamp limit for up projection
   :type up_clamp_upper_limit: ``float``, optional
   :param up_clamp_lower_limit: Lower clamp limit for up projection
   :type up_clamp_lower_limit: ``float``, optional
   :return: Output hidden states with shape ``[T+1, H]``. When activation checkpointing is enabled, may return a tuple including saved activations.
   :rtype: ``nl.ndarray`` or ``Tuple[nl.ndarray, ...]``

   **Dimensions**:

   * T: Total number of input tokens (after linearizing across the batch dimension)
   * H: Hidden dimension size
   * B: Number of tokens per block
   * N: Total number of blocks
   * E: Number of experts
   * I_TP: Intermediate size / tensor parallelism degree

   **Supported Data Types**:

   * Input: bfloat16, float16
   * MX implementations: float4_e2m1fn_x4 (MxFP4), float8_e4m3fn (MxFP8)

   **Constraints**:

   * Block size B: 256-1024 tokens (must be multiple of 256)
   * Total tokens T: Up to 32K tokens per call
   * Hidden dimension H: 512-8192 (optimal: 2048-4096), must be multiple of 512
   * Intermediate dimension I_TP: 2048-16384 (optimal: 8192), must be divisible by 16
   * Number of experts E: 8-64 (optimal: 8-16)
   * All input/output tensors must have the same floating point dtype
   * ``token_position_to_id`` and ``block_to_expert`` must be ``nl.int32`` tensors

Configuration Classes
-----------------------

MoECTESpec
^^^^^^^^^^^

Specification for MoE CTE kernel execution. Selects the implementation variant and provides implementation-specific configuration.

.. code-block:: python

   from nkilib.core.moe.moe_cte.moe_cte import MoECTESpec, MoECTEImplementation

   # Block sharding (default config auto-initialized)
   spec = MoECTESpec(implementation=MoECTEImplementation.shard_on_block)

   # I-sharding with activation checkpointing
   spec = MoECTESpec(
       implementation=MoECTEImplementation.shard_on_i,
       shard_on_I=ShardOnIConfig(checkpoint_activation=True),
   )

**Implementation variants**:

* ``shard_on_block``: Shards blocks across cores. Best for many blocks. (TRN2)
* ``shard_on_i``: Shards intermediate dimension across cores. (TRN2)
* ``shard_on_i_hybrid``: Shard on I with hybrid static/dynamic loop. (TRN2)
* ``shard_on_i_dropping``: Shard on I for dropping layer. (TRN2)
* ``shard_on_block_mx``: Shard on block with MxFP4/MxFP8 quantization. (TRN3)
* ``shard_on_i_mx``: Shard on I with MxFP4/MxFP8 quantization. (TRN3)
* ``shard_on_i_mx_hybrid``: Shard on I with MxFP4/MxFP8 and hybrid loop. (TRN3)

QuantizationConfig
^^^^^^^^^^^^^^^^^^^

Configuration for quantization-related parameters. Contains dequantization scales for weight tensors.

.. code-block:: python

   from nkilib.core.moe.moe_cte.moe_cte import QuantizationConfig

   # No quantization (default)
   quant_cfg = QuantizationConfig()

   # With per-tensor scales
   quant_cfg = QuantizationConfig(
       gate_up_proj_scale=gate_up_scale_tensor,
       down_proj_scale=down_scale_tensor,
   )

* ``gate_up_proj_scale`` (``nl.ndarray``, optional): Dequantization scales for gate/up projection weights.
* ``down_proj_scale`` (``nl.ndarray``, optional): Dequantization scales for down projection weights.

Implementation Details
-------------------------

The kernel implementation includes several key optimizations:

1. **Unified Dispatch**: The ``moe_cte`` entry point dispatches to the appropriate implementation based on ``spec.implementation``.

2. **Block Sharding**: Distributes blocks across cores for parallel processing. Supports PING_PONG and HI_LO distribution strategies.

3. **Intermediate Dimension Sharding**: Distributes the intermediate dimension (I_TP) across multiple cores with all-reduce operations to combine partial results.

4. **Quantization Support**: Handles multiple quantization schemes:
   
   * **FP8 Row Quantization**: Per-row scaling for weights
   * **FP8 Static Quantization**: Single scale per weight matrix
   * **MxFP4/MxFP8**: Microscaling formats with block-wise scaling (TRN3)

5. **Expert Affinity Scaling Modes**:
   
   * **PRE_SCALE**: Apply affinity scaling before activation
   * **POST_SCALE**: Apply affinity scaling after down projection (default)

6. **Hybrid Loop Optimization**: For sequences with padding, uses a hybrid static/dynamic loop where non-padded blocks are processed in a compile-time-known static loop and padded blocks in a runtime-dependent dynamic loop.

7. **Activation Checkpointing**: Optionally saves intermediate activations for gradient computation during backward pass.

8. **Optional Clamping**: Supports clamping of gate and up projection outputs for numerical stability.

Usage Examples
-----------------

Basic usage with block sharding:

.. code-block:: python

   from nkilib.core.moe.moe_cte.moe_cte import moe_cte, MoECTESpec, MoECTEImplementation

   spec = MoECTESpec(implementation=MoECTEImplementation.shard_on_block)

   output = moe_cte(
       hidden_states=hidden_states,
       expert_affinities_masked=expert_affinities,
       gate_up_proj_weight=gate_up_weights,
       down_proj_weight=down_weights,
       token_position_to_id=token_position_to_id,
       block_to_expert=block_to_expert,
       block_size=512,
       spec=spec,
   )

With quantization:

.. code-block:: python

   from nkilib.core.moe.moe_cte.moe_cte import QuantizationConfig

   quant_cfg = QuantizationConfig(
       gate_up_proj_scale=gate_up_scale,
       down_proj_scale=down_scale,
   )

   output = moe_cte(
       hidden_states=hidden_states,
       expert_affinities_masked=expert_affinities,
       gate_up_proj_weight=gate_up_weights,
       down_proj_weight=down_weights,
       token_position_to_id=token_position_to_id,
       block_to_expert=block_to_expert,
       block_size=512,
       spec=spec,
       quantization_config=quant_cfg,
   )

See Also
-----------

* :doc:`MoE TKG Kernel API Reference </nki/library/api/moe-tkg>`
* :doc:`Router Top-K Kernel API Reference </nki/library/api/router-topk>`
* :doc:`MLP Kernel API Reference </nki/library/api/mlp>`
