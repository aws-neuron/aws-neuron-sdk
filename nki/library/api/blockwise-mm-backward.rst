.. meta::
    :description: Blockwise MM Backward kernel computes backward pass for blockwise Mixture of Experts layers.
    :date-modified: 02/13/2026

.. currentmodule:: nkilib.experimental.moe.bwd

Blockwise MM Backward Kernel API Reference
============================================

**[Experimental]** Computes the backward pass for blockwise matrix multiplication in Mixture of Experts (MoE) layers, producing gradients for all parameters.

The kernel supports:

* Gradient computation for hidden states, expert affinities, gate/up weights, and down weights
* Optional bias gradient computation
* Multiple sharding strategies (hidden dimension, intermediate dimension)
* Affinity scaling on hidden or intermediate dimension
* Gradient clamping for numerical stability
* Various activation functions (SiLU, GELU, Swish)
* Dropless MoE with variable block assignments per expert

Background
-----------

The ``blockwise_mm_bwd`` kernel is the backward pass companion to the MoE CTE forward kernel. It computes gradients for all learnable parameters in a blockwise MoE layer by reversing the forward computation:

1. **Down projection backward**: Compute gradients for down projection weights and intermediate activations
2. **Activation backward**: Compute gradients through the activation function using checkpointed activations
3. **Gate/Up projection backward**: Compute gradients for gate and up projection weights
4. **Hidden states backward**: Compute gradients for input hidden states
5. **Affinity backward**: Compute gradients for expert affinities

The kernel uses activation checkpoints saved during the forward pass (``gate_up_proj_act_checkpoint_T`` and ``down_proj_act_checkpoint``) to avoid recomputation.

API Reference
--------------

**Source code for this kernel API can be found at**: `blockwise_mm_backward.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe/bwd/blockwise_mm_backward.py>`_

blockwise_mm_bwd
^^^^^^^^^^^^^^^^^

.. py:function:: blockwise_mm_bwd(hidden_states: nl.ndarray, expert_affinities_masked: nl.ndarray, gate_up_proj_weight: nl.ndarray, down_proj_weight: nl.ndarray, gate_up_proj_act_checkpoint_T: nl.ndarray, down_proj_act_checkpoint: nl.ndarray, token_position_to_id: nl.ndarray, block_to_expert: nl.ndarray, output_hidden_states_grad: nl.ndarray, block_size: int, skip_dma: SkipMode = None, compute_dtype: nki.dtype = nl.bfloat16, is_tensor_update_accumulating: bool = True, shard_option: ShardOption = ShardOption.SHARD_ON_HIDDEN, affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_H, kernel_type_option: KernelTypeOption = KernelTypeOption.DROPLESS, clamp_limits: ClampLimits = None, bias: bool = False, activation_type: ActFnType = ActFnType.SiLU, block_tile_size: int = None) -> tuple

   Compute backward pass for blockwise MoE layer.

   Computes gradients for all parameters in a Mixture of Experts layer using blockwise
   matrix multiplication. Optimized for dropless MoE with variable block assignments per expert.

   :param hidden_states: Input hidden states tensor with shape ``[T, H]`` in HBM.
   :type hidden_states: ``nl.ndarray``
   :param expert_affinities_masked: Expert affinities with shape ``[T * E, 1]`` in HBM.
   :type expert_affinities_masked: ``nl.ndarray``
   :param gate_up_proj_weight: Gate and up projection weights with shape ``[E, H, 2, I_TP]`` in HBM.
   :type gate_up_proj_weight: ``nl.ndarray``
   :param down_proj_weight: Down projection weights with shape ``[E, I_TP, H]`` in HBM.
   :type down_proj_weight: ``nl.ndarray``
   :param gate_up_proj_act_checkpoint_T: Checkpointed gate/up activations from forward pass with shape ``[N, 2, I_TP, B]``.
   :type gate_up_proj_act_checkpoint_T: ``nl.ndarray``
   :param down_proj_act_checkpoint: Checkpointed down projection activations from forward pass with shape ``[N, B, H]``.
   :type down_proj_act_checkpoint: ``nl.ndarray``
   :param token_position_to_id: Token position to block mapping with shape ``[N * B]``.
   :type token_position_to_id: ``nl.ndarray``
   :param block_to_expert: Expert index per block with shape ``[N, 1]``.
   :type block_to_expert: ``nl.ndarray``
   :param output_hidden_states_grad: Upstream gradient from output with shape ``[T, H]``.
   :type output_hidden_states_grad: ``nl.ndarray``
   :param block_size: Number of tokens per block. Must be one of: 128, 256, 512, 1024.
   :type block_size: ``int``
   :param skip_dma: DMA skip mode for OOB handling. Default: ``SkipMode(False, False)``.
   :type skip_dma: ``SkipMode``, optional
   :param compute_dtype: Computation data type. Default: ``nl.bfloat16``.
   :type compute_dtype: ``nki.dtype``
   :param is_tensor_update_accumulating: Whether to accumulate into existing gradients. Default: ``True``.
   :type is_tensor_update_accumulating: ``bool``
   :param shard_option: Sharding strategy. ``SHARD_ON_HIDDEN``: shard across hidden dimension. ``SHARD_ON_INTERMEDIATE``: shard across intermediate dimension. ``AUTO``: auto-select. Default: ``SHARD_ON_HIDDEN``.
   :type shard_option: ``ShardOption``
   :param affinity_option: Dimension for affinity scaling. ``AFFINITY_ON_H``: scale on hidden dimension. ``AFFINITY_ON_I``: scale on intermediate dimension. Default: ``AFFINITY_ON_H``.
   :type affinity_option: ``AffinityOption``
   :param kernel_type_option: Token dropping strategy. ``DROPLESS``: variable blocks per expert. ``DROPPING``: fixed blocks per expert. Default: ``DROPLESS``.
   :type kernel_type_option: ``KernelTypeOption``
   :param clamp_limits: Gradient clamping limits for numerical stability. Contains ``linear_clamp_upper_limit``, ``linear_clamp_lower_limit``, ``non_linear_clamp_upper_limit``, ``non_linear_clamp_lower_limit``.
   :type clamp_limits: ``ClampLimits``, optional
   :param bias: Whether to compute bias gradients. Default: ``False``.
   :type bias: ``bool``
   :param activation_type: Activation function type. Default: ``SiLU``.
   :type activation_type: ``ActFnType``
   :param block_tile_size: Optional tile size override for block processing.
   :type block_tile_size: ``int``, optional
   :return: Tuple of gradient tensors. When ``bias=False``: ``(hidden_states_grad, expert_affinities_masked_grad, gate_up_proj_weight_grad, down_proj_weight_grad)``. When ``bias=True``: additionally includes ``gate_and_up_proj_bias_grad`` and ``down_proj_bias_grad``.
   :rtype: ``tuple``

   **Dimensions**:

   * T: Total number of input tokens
   * H: Hidden dimension size
   * I_TP: Intermediate size / tensor parallel degree
   * E: Number of experts
   * B: Block size (tokens per block)
   * N: Number of blocks

   **Supported Data Types**:

   * Input: bfloat16, float16

   **Constraints**:

   * ``block_size`` must be one of: 128, 256, 512, 1024
   * H must be divisible by the number of shards for LNC sharding
   * Currently only supports ``DROPLESS`` kernel type
   * Requires activation checkpoints from the forward pass (``gate_up_proj_act_checkpoint_T`` and ``down_proj_act_checkpoint``)

Implementation Details
-----------------------

The kernel implementation includes several key optimizations:

1. **Sharding Strategies**: Supports sharding across hidden dimension (simpler, no H-tiling) or intermediate dimension (better memory efficiency) for LNC2 parallelism.

2. **Activation Checkpointing**: Uses saved activations from the forward pass to avoid recomputation during backward, trading memory for compute.

3. **Blockwise Processing**: Processes tokens in blocks matching the forward pass structure, enabling efficient gradient accumulation across experts.

4. **Gradient Clamping**: Optional clamping of gradients for numerical stability during training.

5. **Affinity Gradient Computation**: Computes gradients for expert routing weights, enabling end-to-end training of the router.

See Also
-----------

* :doc:`MoE CTE Kernel API Reference </nki/library/api/moe-cte>`
* :doc:`MoE TKG Kernel API Reference </nki/library/api/moe-tkg>`
