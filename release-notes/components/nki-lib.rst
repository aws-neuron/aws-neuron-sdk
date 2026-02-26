.. meta::
    :description: Complete release notes for the NKI Library component across all AWS Neuron SDK versions.
    :keywords: nki library, nki-lib, release notes, aws neuron sdk
    :date-modified: 02/26/2026

.. _nki-lib_rn:

Release Notes for Neuron Component: NKI Library
================================================

The release notes for the {component name} Neuron component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _nki-lib-2-28-0-rn:   

NKI Library (NKI-Lib) (Neuron 2.28.0 Release)
--------------------------------------------------------------------

What's New
~~~~~~~~~~

This release expands the NKI Library with 9 new kernels, bringing the total to 16 documented kernel APIs. New core kernels include RoPE, Router Top-K, MoE CTE, MoE TKG, and Cumsum. New experimental kernels include Attention Block TKG (fused attention block for token generation), Cross Entropy (forward and backward passes), Depthwise Conv1D, and Blockwise MM Backward for MoE training.

Existing kernels receive FP8 and MX quantization support across QKV, MLP, and both Output Projection kernels. Kernel utilities gain new TensorView methods, SbufManager logging improvements with tree-formatted allocation tracing, and new utilities including ``interleave_copy``, ``LncSubscriptable``, and ``rmsnorm_mx_quantize_tkg``. Note that several breaking changes affect kernel signatures and utility APIs — see the Breaking Changes section for details.

New Core Kernels
^^^^^^^^^^^^^^^^

* :doc:`RoPE Kernel </nki/library/api/rope>` — Applies Rotary Position Embedding to input embeddings with optional LNC sharding and flexible layout support (contiguous and interleaved).
* :doc:`Router Top-K Kernel </nki/library/api/router-topk>` — Computes router logits and top-K expert selection for Mixture of Experts models, with support for multiple layout configurations and sharding strategies.
* :doc:`MoE CTE Kernel </nki/library/api/moe-cte>` — Implements Mixture of Experts optimized for Context Encoding with multiple sharding strategies (block sharding, intermediate dimension sharding) and MxFP4/MxFP8 quantization.
* :doc:`MoE TKG Kernel </nki/library/api/moe-tkg>` — Implements Mixture of Experts optimized for Token Generation with all-expert and selective-expert modes, supporting FP8 and MxFP4 quantization.
* :doc:`Cumsum Kernel </nki/library/api/cumsum>` — Computes cumulative sum along the last dimension, optimized for batch sizes up to 2048.

New Experimental Kernels
^^^^^^^^^^^^^^^^^^^^^^^^

* :doc:`Attention Block TKG Kernel </nki/library/api/attention-block-tkg>` — Fused attention block for Token Generation that combines RMSNorm, QKV projection, RoPE, attention, and output projection in SBUF to minimize HBM traffic.
* :doc:`Cross Entropy Kernel </nki/library/api/cross-entropy>` — Memory-efficient cross entropy loss forward and backward passes for large vocabularies using online log-sum-exp algorithm, optimized for LNC2.
* :doc:`Depthwise Conv1D Kernel </nki/library/api/depthwise-conv1d>` — Depthwise 1D convolution using implicit GEMM algorithm with support for arbitrary stride and padding values, optimized for TRN2.
* :doc:`Blockwise MM Backward Kernel </nki/library/api/blockwise-mm-backward>` — Backward pass for blockwise matrix multiplication in MoE layers, computing gradients for all parameters with support for dropless MoE.

Improvements
~~~~~~~~~~~~~~~

* :doc:`QKV Kernel </nki/library/api/qkv>`: Added FP8 quantization support (``quantization_type``, ``qkv_w_scale``, ``qkv_in_scale``), fused FP8 KV cache quantization (``k_cache``, ``v_cache``, ``k_scale``, ``v_scale``, ``fp8_max``, ``fp8_min``, ``kv_dtype``), block-based KV cache layout (``use_block_kv``, ``block_size``, ``slot_mapping``), and MX quantization input swizzling (``is_input_swizzled``).
* :doc:`MLP Kernel </nki/library/api/mlp>`: Added FP8 quantization support (``quantization_type``, ``gate_w_scale``, ``up_w_scale``, ``down_w_scale``, ``gate_up_in_scale``, ``down_in_scale``, ``quant_clipping_bound``), gate/up projection clamping (``gate_clamp_upper_limit``, ``gate_clamp_lower_limit``, ``up_clamp_upper_limit``, ``up_clamp_lower_limit``), ``skip_gate_proj`` option, and fp16 support for TKG mode.
* :doc:`Output Projection CTE Kernel </nki/library/api/output-projection-cte>`: Added FP8 quantization support (``quantization_type``, ``input_scales``, ``weight_scales``).
* :doc:`Output Projection TKG Kernel </nki/library/api/output-projection-tkg>`: Added FP8 quantization support (``quantization_type``, ``weight_scale``, ``input_scale``) and removed 512 restriction on non-transpose path.
* :doc:`Attention CTE Kernel </nki/library/api/attention-cte>`: Added strided Q slicing for context parallelism (``cp_strided_q_slicing``).
* :doc:`RMSNorm-Quant Kernel </nki/library/api/rmsnorm-quant>`: Added input dequantization scale support (``input_dequant_scale``).

Kernel Utilities
^^^^^^^^^^^^^^^^

See :doc:`Kernel Utilities Reference </nki/library/kernel-utils/index>` for full documentation.

* :doc:`TensorView </nki/library/kernel-utils/tensor-view>`: Added ``rearrange`` method for flexible dimension reordering, ``has_dynamic_access`` for checking whether a view requires runtime-dependent addressing, and ``key_in_dict`` helper. The ``slice`` method now clamps the end index to dimension bounds instead of asserting.
* :doc:`TiledRange </nki/library/kernel-utils/tiled-range>`: ``TiledRangeIterator`` now exposes an ``end_offset`` attribute, enabling kernels to determine the end position of each tile without manual calculation.
* :doc:`SbufManager (Allocator) </nki/library/kernel-utils/allocator>`: Added ``get_total_space`` and ``get_used_space`` for querying SBUF utilization, ``set_name_prefix`` / ``get_name_prefix`` for scoped naming, and ``flush_logs`` to emit buffered allocation logs. SbufManager now uses ``TreeLogger`` to provide hierarchical, tree-formatted logs of SBUF allocation and deallocation events, making it easier to debug memory usage across nested scopes.
* **QuantizationType**: Added ``MX`` enum value for microscaling quantization (MxFP4/MxFP8).
* **common_types**: Added ``GateUpDim`` enum for distinguishing gate vs up projection dimensions.
* **rmsnorm_tkg / layernorm_tkg**: Both subkernels now accept a ``TensorView`` or ``nl.ndarray`` for input and require an explicit ``output`` tensor parameter, giving callers control over output placement.
* **New utilities**: Added ``rmsnorm_mx_quantize_tkg`` subkernel for fused RMSNorm with MX quantization in token generation, ``interleave_copy`` for interleaved tensor copy operations, ``LncSubscriptable`` for LNC-aware data access patterns, and ``TreeLogger`` for hierarchical allocation logging.

Breaking Changes
~~~~~~~~~~~~~~~~

* The open source repository source directory has been renamed from ``nkilib_standalone`` to ``nkilib_src``.
* :doc:`MLP Kernel </nki/library/api/mlp>`: The function has been renamed from ``mlp_kernel`` to ``mlp``. New parameters have been inserted in the middle of the signature; callers using positional arguments beyond ``normalization_type`` must update to use keyword arguments.
* :doc:`QKV Kernel </nki/library/api/qkv>`: New parameters (``quantization_type``, ``qkv_w_scale``, ``qkv_in_scale``) have been inserted after ``bias``; callers using positional arguments beyond ``bias`` must update to use keyword arguments.
* :doc:`Output Projection TKG Kernel </nki/library/api/output-projection-tkg>`: The ``bias`` parameter is now optional (default ``None``). New parameters (``quantization_type``, ``weight_scale``, ``input_scale``) have been inserted before ``TRANSPOSE_OUT``; callers using positional arguments beyond ``bias`` must update to use keyword arguments.
* **TiledRangeIterator**: The constructor now requires a fourth positional argument ``end_offset``.
* **TensorView**: The ``sizes`` attribute has been renamed to ``shape``.
* **rmsnorm_tkg**: The ``inp`` parameter has been renamed to ``input``. A new required ``output`` parameter has been added as the third argument. The ``output_in_sbuf`` parameter has been removed. New parameters ``hidden_dim_tp`` and ``single_core_forced`` have been added.
* **layernorm_tkg**: The ``inp`` parameter has been renamed to ``input``. A new required ``output`` parameter has been added as the third argument. The ``output_in_sbuf`` parameter has been removed.

Bug Fixes
~~~~~~~~~

* Fixed attention TKG compilation and non-determinism issues.
* Fixed incorrect v_active slice indices in attention TKG block KV path.
* Fixed batch sharding in gen_mask_tkg active mask loading.
* Fixed expert_affinities masking when ``mask_unselected_experts`` is True in MoE TKG.
* Fixed expert_index shape mismatch in MoE TKG for T > 128.
* Fixed MoE affinity mask handling for T not divisible by 128.
* Fixed MoE TKG MX weight generation x4 pack size.
* Fixed MLP CTE ``force_cte_mode`` parameter validation.
* Fixed output projection CTE mixed precision support.
* Fixed output projection TKG variable name typo.
* Fixed router_topk bias shape to satisfy NKI check requirements.
* Fixed tail iteration bug for sequences not a multiple of 128 in MoE CTE.
* Fixed reading extra partitions for last rank in MoE CTE.

Known Issues
~~~~~~~~~~~~

.. _nki-lib-2-27-0-rn:

NKI Library (NKI-Lib) (Neuron 2.27.0 Release)
--------------------------------------------------------------------

What's New
~~~~~~~~~~

This release introduces the NKI Library, which provides pre-built kernels you can use to optimize
the performance of your models. The NKI Library offers ready-to-use, pre-optimized kernels that
leverage the full capabilities of AWS Trainium hardware.

NKI Library kernels are published in the `NKI Library GitHub repository <https://github.com/aws-neuron/nki-library>`_.
In Neuron 2.27, these kernels are also shipped as part of neuronx-cc under the ``nkilib.*`` namespace.

Accessing NKI Library Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can access NKI Library kernels in two ways:

* **Shipped version**: Import from the ``nkilib.*`` namespace (included with neuronx-cc in Neuron 2.27)
* **Open source repository**: Clone and use kernels from the GitHub repository under the ``nkilib_standalone.nkilib.*`` namespace

New Kernels
~~~~~~~~~~~

This release includes the following pre-optimized kernels:

* **Attention CTE Kernel** — Implements attention with support for multiple variants and optimizations
* **Attention TKG Kernel** — Implements attention specifically optimized for token generation scenarios
* **MLP Kernel** — Implements a Multi-Layer Perceptron with optional normalization fusion and various optimizations
* **Output Projection CTE Kernel** — Computes the output projection operation optimized for Context Encoding use cases
* **Output Projection TKG Kernel** — Computes the output projection operation optimized for Token Generation use cases
* **QKV Kernel** — Performs Query-Key-Value projection with optional normalization fusion
* **RMSNorm-Quant Kernel** — Performs optional RMS normalization followed by quantization to fp8

NKI Library Kernel Migration to New nki.* Namespace in Neuron 2.28
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some NKI Library kernels currently use the legacy ``neuronxcc.nki.*`` namespace. Starting with
Neuron 2.28, all NKI Library kernels will migrate to the new ``nki.*`` namespace.

The new ``nki.*`` namespace introduces changes to NKI APIs and language constructs. Customers
using NKI Library kernels should review the migration guide for any required changes.

NKI Library Namespace Changes in Neuron 2.28
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Starting with Neuron 2.28, the open source repository namespace will change from
``nkilib_standalone.nkilib.*`` to ``nkilib.*``, providing a consistent namespace between
the open source repository and the shipped version.

Customers who want to add or modify NKI Library kernels can build and install them to
replace the default implementation without changing model imports.



    