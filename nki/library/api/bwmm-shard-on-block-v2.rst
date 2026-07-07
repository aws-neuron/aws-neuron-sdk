.. meta::
    :description: Blockwise matrix multiplication kernel for context-encoding MoE layers.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.moe.moe_cte

Blockwise MM Shard-on-Block Kernel API Reference
================================================

Blockwise matrix multiplication kernel for context-encoding MoE layers.

This kernel implements blockwise matrix multiplication for mixture-of-experts (MoE) layers, processing tokens through expert-specific gate, up, and down projections. The computation combines static optimization benefits with dynamic early-exit capabilities by using a hybrid loop structure. Optimized for block-level sharding with PING_PONG strategy and supports FP8 quantization, multiple expert affinity scaling modes, and TopK > 1 accumulation patterns. Optimized for block sizes 128-512 tokens, 8-64 experts, and sequence lengths up to 32K tokens. Best performance when I_TP >= 512 and batch size * sequence length <= 4096.

Background
-----------

The ``bwmm_shard_on_block`` kernel implements blockwise matrix multiplication for context-encoding MoE layers, processing tokens through expert-specific gate, up, and down projections with block-level sharding. A hybrid static/dynamic loop structure combines static-scheduling benefits with dynamic early-exit, and the ``bwmm_shard_on_block_hybrid`` entry point exposes the hybrid path directly.

API Reference
--------------

**Source code for this kernel API can be found at**: `bwmm_shard_on_block_v2.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe/moe_cte/bwmm_shard_on_block_v2.py>`_

bwmm_shard_on_block
^^^^^^^^^^^^^^^^^^^

.. py:function:: bwmm_shard_on_block(hidden_states: nl.ndarray, expert_affinities_masked: nl.ndarray, gate_up_proj_weight: nl.ndarray, down_proj_weight: nl.ndarray, block_size: int, token_position_to_id: nl.ndarray, block_to_expert: nl.ndarray, gate_and_up_proj_bias: Optional[nl.ndarray] = None, down_proj_bias: Optional[nl.ndarray] = None, gate_up_proj_scale: Optional[nl.ndarray] = None, down_proj_scale: Optional[nl.ndarray] = None, down_activations: Optional[nl.ndarray] = None, activation_function: common_types.ActFnType = common_types.ActFnType.SiLU, skip_dma: SkipMode = SkipMode(False, False), compute_dtype: Any = nl.bfloat16, is_tensor_update_accumulating: bool = True, expert_affinities_scaling_mode: common_types.ExpertAffinityScaleMode = common_types.ExpertAffinityScaleMode.POST_SCALE, n_block_per_iter: int = 1, gate_clamp_upper_limit: Optional[float] = None, gate_clamp_lower_limit: Optional[float] = None, up_clamp_upper_limit: Optional[float] = None, up_clamp_lower_limit: Optional[float] = None, block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG, sbm: Optional[SbufManager] = None, num_static_block: Optional[int] = None, total_n_blocks: Optional[int] = None, down_bias_tp_degree: Optional[int] = None, down_bias_tp_rank: Optional[int] = None, non_overlapping_shards: bool = False)

   Blockwise matrix multiplication kernel for context-encoding MoE layers.

   :param hidden_states: [T, H], Input token embeddings in HBM
   :type hidden_states: ``nl.ndarray``
   :param expert_affinities_masked: [(T+1)*E, 1], Expert routing weights for token assignments in HBM
   :type expert_affinities_masked: ``nl.ndarray``
   :param gate_up_proj_weight: [E, H, 2, I_TP], Combined gate and up projection weights in HBM
   :type gate_up_proj_weight: ``nl.ndarray``
   :param down_proj_weight: [E, I_TP, H], Down projection weights in HBM
   :type down_proj_weight: ``nl.ndarray``
   :param block_size: Number of tokens processed per block
   :type block_size: ``int``
   :param token_position_to_id: [N*B], Mapping from block positions to token IDs in HBM
   :type token_position_to_id: ``nl.ndarray``
   :param block_to_expert: [N, 1], Expert assignment for each block in HBM
   :type block_to_expert: ``nl.ndarray``
   :param gate_and_up_proj_bias: [E, 2, I_TP], Bias terms for gate/up projections in HBM
   :type gate_and_up_proj_bias: ``Optional[nl.ndarray]``
   :param down_proj_bias: [E, 1, H], Bias terms for down projection in HBM
   :type down_proj_bias: ``Optional[nl.ndarray]``
   :param gate_up_proj_scale: [E, 1, 2*I_TP], Dequantization scales for gate/up weights in HBM
   :type gate_up_proj_scale: ``Optional[nl.ndarray]``
   :param down_proj_scale: [E, 1, H], Dequantization scales for down weights in HBM
   :type down_proj_scale: ``Optional[nl.ndarray]``
   :param down_activations: [N, B, H], Storage for intermediate activations in HBM
   :type down_activations: ``Optional[nl.ndarray]``
   :param activation_function: Activation function type (SiLU, GELU, etc.)
   :type activation_function: ``common_types.ActFnType``
   :param skip_dma: DMA skip configuration for memory optimization
   :type skip_dma: ``SkipMode``
   :param compute_dtype: Data type for internal computations (default: bfloat16)
   :type compute_dtype: ``Any``
   :param is_tensor_update_accumulating: Enable accumulation for TopK > 1 scenarios
   :type is_tensor_update_accumulating: ``bool``
   :param expert_affinities_scaling_mode: Expert affinity application mode
   :type expert_affinities_scaling_mode: ``common_types.ExpertAffinityScaleMode``
   :param n_block_per_iter: Number of blocks processed per iteration
   :type n_block_per_iter: ``int``
   :param gate_clamp_upper_limit: Upper clamp limit for gate projections
   :type gate_clamp_upper_limit: ``Optional[float]``
   :param gate_clamp_lower_limit: Lower clamp limit for gate projections
   :type gate_clamp_lower_limit: ``Optional[float]``
   :param up_clamp_upper_limit: Upper clamp limit for up projections
   :type up_clamp_upper_limit: ``Optional[float]``
   :param up_clamp_lower_limit: Lower clamp limit for up projections
   :type up_clamp_lower_limit: ``Optional[float]``
   :param block_sharding_strategy: Block distribution strategy across cores
   :type block_sharding_strategy: ``BlockShardStrategy``
   :param sbm: SBUF memory manager. If None, one is created internally.
   :type sbm: ``Optional[SbufManager]``
   :param num_static_block: Number of blocks for static loop. Defaults to N.
   :type num_static_block: ``Optional[int]``
   :param total_n_blocks: Total block count for shard partitioning. Defaults to num_static_block.
   :type total_n_blocks: ``Optional[int]``
   :param down_bias_tp_degree: TP degree for down projection bias sharding.
   :type down_bias_tp_degree: ``Optional[int]``
   :param down_bias_tp_rank: TP rank for down projection bias sharding.
   :type down_bias_tp_rank: ``Optional[int]``
   :param non_overlapping_shards: When True, shards write to the same output slot (slot 0) and skip zero-init and cross-shard reduce. Requires non-overlapping token routing across shards (e.g., HI_LO strategy with sequence-level sharding). Default: False.
   :type non_overlapping_shards: ``bool``
   :return: Expert-processed token representations in HBM. Shape depends on accumulation mode: - Single expert (is_tensor_update_accumulating=False): [T, H] - Multiple experts (is_tensor_update_accumulating=True): [T, 2, H+E] for cross-core accumulation (the trailing E columns hold expert affinities; the hidden payload occupies columns 0:H)
   :rtype: ``nl.ndarray``

   **Notes**:

   * Supports the PING_PONG and HI_LO block sharding strategies (selected via ``block_sharding_strategy``; default PING_PONG)
   * Static loop processes N-E blocks with compile-time optimizations
   * Dynamic loop handles remaining blocks with early-exit capability
   * Supports FP8 quantization with dequantization scales
   * Expert affinity scaling modes: PRE_SCALE, POST_SCALE, PRE_SCALE_DELAYED
   * Multi-shard execution requires num_shards == 2 for accumulation

   **Dimensions**:

   * T: Total number of input tokens
   * H: Hidden dimension size
   * B: Block size (tokens per block)
   * E: Number of experts
   * N: Total number of blocks (T / B)
   * I_TP: Intermediate size divided by the tensor-parallelism degree

bwmm_shard_on_block_hybrid
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: bwmm_shard_on_block_hybrid(conditions: nl.ndarray, hidden_states: nl.ndarray, expert_affinities_masked: nl.ndarray, gate_up_proj_weight: nl.ndarray, down_proj_weight: nl.ndarray, block_size: int, token_position_to_id: nl.ndarray, block_to_expert: nl.ndarray, gate_and_up_proj_bias: Optional[nl.ndarray] = None, down_proj_bias: Optional[nl.ndarray] = None, gate_up_proj_scale: Optional[nl.ndarray] = None, down_proj_scale: Optional[nl.ndarray] = None, down_activations: Optional[nl.ndarray] = None, activation_function: common_types.ActFnType = common_types.ActFnType.SiLU, skip_dma: SkipMode = SkipMode(False, False), compute_dtype: Any = nl.bfloat16, is_tensor_update_accumulating: bool = True, expert_affinities_scaling_mode: common_types.ExpertAffinityScaleMode = common_types.ExpertAffinityScaleMode.POST_SCALE, n_block_per_iter: int = 1, gate_clamp_upper_limit: Optional[float] = None, gate_clamp_lower_limit: Optional[float] = None, up_clamp_upper_limit: Optional[float] = None, up_clamp_lower_limit: Optional[float] = None, block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG, down_bias_tp_degree: Optional[int] = None, down_bias_tp_rank: Optional[int] = None, non_overlapping_shards: bool = False)

   Hybrid static/dynamic shard-on-block kernel.

   :param conditions: [ceil(N/num_shards)+1] per-shard condition vector. 1=active, 0=padded. Last entry must be 0 for loop termination. All other args: same as bwmm_shard_on_block.
   :type conditions: ``nl.ndarray``

compute_same_weights_block_parallel_hbm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: compute_same_weights_block_parallel_hbm(N: int, block_to_expert: nl.ndarray, num_shards: int, shard_id: int, shard_strat: BlockShardStrategy, sbm: Optional[SbufManager] = None) -> nl.ndarray

   Compute weight reuse mask for block-parallel execution.

   :param N: Total number of blocks
   :type N: ``int``
   :param block_to_expert: Expert assignment for each block
   :type block_to_expert: ``nl.ndarray``
   :param num_shards: Number of shards for parallel execution
   :type num_shards: ``int``
   :param shard_id: Current shard identifier
   :type shard_id: ``int``
   :param shard_strat: Block distribution strategy
   :type shard_strat: ``BlockShardStrategy``

load_down_proj_weight
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: load_down_proj_weight(down_proj_weight: nl.ndarray, block_expert: nl.ndarray, compute_dtype, skip_dma: SkipMode = SkipMode(), load_dst: Optional[list] = None, sbm: Optional[SbufManager] = None) -> list

   Load down projection weights.

   :param down_proj_weight: Weight tensor with shape [E, I_TP, H]
   :type down_proj_weight: ``nl.ndarray``
   :param block_expert: Expert index tensor with shape (1, 1) in SBUF
   :type block_expert: ``nl.ndarray``
   :param compute_dtype: Compute data type
   :param skip_dma: DMA skip configuration
   :type skip_dma: ``SkipMode``
   :param load_dst: Optional pre-allocated destination list
   :type load_dst: ``Optional[list]``

   **Notes**:

   * Assumes I_TP is divisible by 16 for vector operations
   * Partial tiles are zero-padded
   * Uses scalar_offset for dynamic expert indexing

load_gate_up_proj_weights
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: load_gate_up_proj_weights(gate_up_proj_weight: nl.ndarray, block_expert: nl.ndarray, compute_dtype, skip_dma: SkipMode = SkipMode(), load_dst: Optional[list] = None, sbm: Optional[SbufManager] = None) -> list

   Load gate and up projection weights.

   :param gate_up_proj_weight: Weight tensor with shape [E, H, 2, I_TP]
   :type gate_up_proj_weight: ``nl.ndarray``
   :param block_expert: Expert index tensor with shape (1, 1) in SBUF
   :type block_expert: ``nl.ndarray``
   :param compute_dtype: Compute data type
   :param skip_dma: DMA skip configuration
   :type skip_dma: ``SkipMode``
   :param load_dst: Optional pre-allocated destination list
   :type load_dst: ``Optional[list]``

   **Notes**:

   * Gate and up projections are interleaved in dimension 2
   * Partial tiles are zero-padded
   * Uses scalar_offset for dynamic expert indexing

compute_block_output
^^^^^^^^^^^^^^^^^^^^

.. py:function:: compute_block_output(intermediate_states, dp_weights, expert_affinity, block_old, down_activations, block_idx, H, I_TP, NUM_TILES, output_dtype, is_tensor_update_accumulating, down_bias_broadcasted = None, down_bias_raw = None, down_scale = None, sbm: Optional[SbufManager] = None, down_proj_weight_hbm = None, block_expert = None, skip_dma: SkipMode = SkipMode(), block_new_lst_pre = None, i_tp_offset = 0, down_bias_h_offset: int = 0, down_bias_h_size: Optional[int] = None)

   Compute block output with down projection and expert affinity scaling.

   :param intermediate_states: Intermediate activation states [gup_tile_count][TILE_SIZE, B]
   :param dp_weights: Down projection weights [gup_tile_count][TILE_SIZE, H]
   :param expert_affinity: Expert affinities [NUM_TILES][TILE_SIZE, 1]
   :param block_old: Previous block outputs for accumulation [NUM_TILES][TILE_SIZE, H]
   :param down_activations: Storage for intermediate activations
   :param block_idx: Current block index
   :param H: Hidden dimension size
   :param I_TP: Intermediate dimension size
   :param NUM_TILES: Number of tiles per block
   :param output_dtype: Output data type
   :param is_tensor_update_accumulating: Enable accumulation mode
   :param down_bias_broadcasted: Broadcasted bias [TILE_SIZE, H]
   :param down_scale: Dequantization scales
   :return: Block output tensors [NUM_TILES][TILE_SIZE, H]
   :rtype: ``list``

   **Notes**:

   * Supports FP8 dequantization with down_scale
   * Accumulation mode for TopK > 1 scenarios
   * Optional bias addition before affinity scaling

reduce_outputs
^^^^^^^^^^^^^^

.. py:function:: reduce_outputs(output: nl.ndarray, num_tiles: int, reduce_tile_size: int, offset: int, dim_hidden: int, sbm: Optional[SbufManager] = None)

   Synchronize across axis=0 in output by performing FMA reduce and store.

   :param output: Output tensor, size [T, 2, H+E] (hidden payload in columns 0:H; trailing E columns hold expert affinities)
   :type output: ``nl.ndarray``
   :param num_tiles: Number of tiles (iterations)
   :type num_tiles: ``int``
   :param reduce_tile_size: Size of tile size on partition dimension
   :type reduce_tile_size: ``int``
   :param offset: Output read/write offset on row
   :type offset: ``int``
   :param dim_hidden: Hidden dimension
   :type dim_hidden: ``int``
   :param sbm: Optional SBUF manager for allocation.
   :type sbm: ``Optional[SbufManager]``

load_and_transpose_gup_bias
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: load_and_transpose_gup_bias(inps: InputTensors, dims: DimensionSizes, cfg: Configs, block_expert, skip_dma, sbm: Optional[SbufManager] = None)

   Load and transpose gate/up projection bias for current expert.

   :param inps: Input tensor container
   :type inps: ``InputTensors``
   :param dims: Dimension configuration
   :type dims: ``DimensionSizes``
   :param cfg: Kernel configuration
   :type cfg: ``Configs``
   :param block_expert: Expert index for current block [1, 1]
   :param skip_dma: DMA skip configuration
   :param sbm: Optional SBUF manager for allocation.
   :type sbm: ``Optional[SbufManager]``

shard_strat2blk_idx
^^^^^^^^^^^^^^^^^^^

.. py:function:: shard_strat2blk_idx(shard_strat: BlockShardStrategy, outer_block_iter: int, inner_block_iter: int) -> int

   Convert shard strategy indices to global block index.

   :param shard_strat: Sharding strategy (HI_LO or PING_PONG)
   :type shard_strat: ``BlockShardStrategy``
   :param outer_block_iter: Outer block iteration index
   :type outer_block_iter: ``int``
   :param inner_block_iter: Inner block iteration index (0 to BLOCK_PARALLEL_FACTOR-1)
   :type inner_block_iter: ``int``
   :return: Global block index
   :rtype: ``int``

shard_strat2new_blk_idx_offset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: shard_strat2new_blk_idx_offset(shard_id: int, shard_strat: BlockShardStrategy, n_blocks_per_shard: int) -> int

   Calculate block index offset based on shard ID and strategy.

   :param shard_id: Current shard identifier (0 or 1)
   :type shard_id: ``int``
   :param shard_strat: Sharding strategy
   :type shard_strat: ``BlockShardStrategy``
   :param n_blocks_per_shard: Number of blocks per shard
   :type n_blocks_per_shard: ``int``
   :return: Block index offset for the current shard
   :rtype: ``int``

load_and_broadcast_down_bias
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: load_and_broadcast_down_bias(inps: InputTensors, dims: DimensionSizes, cfg: Configs, block_expert, skip_dma, sbm: Optional[SbufManager] = None, use_pe_broadcast: bool = False)

   Load and broadcast down projection bias for the current block.

   :param inps: Input tensor container
   :type inps: ``InputTensors``
   :param dims: Dimension configuration
   :type dims: ``DimensionSizes``
   :param cfg: Kernel configuration
   :type cfg: ``Configs``
   :param block_expert: Expert index for current block
   :param skip_dma: DMA skip configuration
   :param sbm: Optional SBUF manager for allocation.
   :type sbm: ``Optional[SbufManager]``
   :param use_pe_broadcast: Use PE matmul broadcast instead of DVE StreamShuffle.
   :type use_pe_broadcast: ``bool``

load_down_bias_raw
^^^^^^^^^^^^^^^^^^

.. py:function:: load_down_bias_raw(inps, dims, cfg, block_expert, skip_dma, sbm = None, bias_h_size = None)

   Load raw (1, bias_h_size) down bias without broadcasting. bias_h_size defaults to H.


bwmm_output_initialization
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: bwmm_output_initialization(output, shard_id = None, sbm: Optional[SbufManager] = None, expert_affinities_masked = None, E = 0, H = 0, skip_zero_init = False)

   Zero initialize buffer at `output` and optionally copy expert affinities.

   :param output: External memory, shape (T, H) or (T, 2, H+E).
   :param shard_id: Optionally provide shard ID.
   :param sbm: Optional SBUF manager for allocation.
   :type sbm: ``Optional[SbufManager]``
   :param expert_affinities_masked: Expert affinities [(T+1)*E, 1], or None.
   :param E: Number of experts.
   :param H: Hidden dimension (excluding affinity columns).
   :param skip_zero_init: Skip zero initialization of output[:, shard_id, :H]. Used with non_overlapping_shards where zero-init is unnecessary.

bwmm_load_old_block
^^^^^^^^^^^^^^^^^^^

.. py:function:: bwmm_load_old_block(output, token_indices, NUM_TILES, dtype, skip_dma: SkipMode = SkipMode(), shard_id = None, token_indices_offset = 0, sbm: Optional[SbufManager] = None)

   Loads the partially computed output hidden states for the current block's token indices.


