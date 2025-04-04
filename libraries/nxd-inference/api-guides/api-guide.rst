.. _nxd-inference-api-guide:

NxD Inference API Reference
===========================

NeuronX Distributed (NxD) Inference (``neuronx-distributed-inference``) is
an open-source PyTorch-based inference library that simplifies deep learning
model deployment on AWS Inferentia and Trainium instances. Neuronx Distributed
Inference includes a model hub and modules that users can reference to
implement their own models on Neuron.

This API guide describes API and configuration functions and parameters that you
can use when you directly interact with the NxD Inference library.

.. note ::

   NxD Inference also supports integration with vLLM. When you use vLLM, you can
   use the ``override_neuron_config`` attribute to override defaults using the
   :ref:`NeuronConfig parameters <nxd-inference-api-guide-neuron-config>` described
   in this API guide. For more information about vLLM integration, see :ref:`nxdi-vllm-user-guide`.


.. contents:: Table of contents
   :local:
   :depth: 2

Configuration
-------------

NxD Inference defines configuration objects that enable you to control how a model
is compiled and used for inference. When you compile a model, its configuration is
serialized to a JSON file in the compiled checkpoint, so you can distribute the
compiled checkpoint to additional Neuron instances without needing to compile on
each instance.

NxD Inference supports loading HuggingFace model checkpoints and configurations.
When you run a model from a HuggingFace checkpoint, NxD Inference loads the model
configuration from the model's PretrainedConfig.

.. _nxd-inference-api-guide-neuron-config:

NeuronConfig
~~~~~~~~~~~~

NeuronConfig contains compile-time configuration options for inference on Neuron. 

Initialization
^^^^^^^^^^^^^^

Pass the NeuronConfig attributes as keyword args.

Functions
^^^^^^^^^

- ``NeuronConfig(**kwargs)`` - Initializes a NeuronConfig with
  attributes from ``kwargs``.

Attributes
^^^^^^^^^^

- General configuration

  - ``batch_size`` - The number of inputs to process in a single
    request. Defaults to ``1``.
  - ``padding_side`` - The padding side. Defaults to ``right``.
  - ``seq_len`` - The sequence length, which is typically the sum of
    ``max_context_length`` and ``max_new_tokens``. This value is the
    maximum sequence size that the model can process in a single
    request. Defaults to ``128``.
  - ``max_context_length`` - The maximum context length. Default to the
    ``seq_len``.
  - ``max_new_tokens`` - The maximum number of tokens to generate in a
    single request. Default to the difference between ``seq_len`` and
    ``max_context_length``. If the difference is zero, then
    ``max_new_tokens`` is set to ``None``.
  - ``max_length`` - The maximum length to process. Default to the
    ``seq_len``.
  - ``n_active_tokens`` - The number of active tokens to track. Defaults
    to the ``seq_len``.
  - ``n_positions`` - The number of positions to track. Defaults to the
    ``seq_len``.
  - ``torch_dtype`` - The torch data type to use for computation. Choose
    from the following options. Defaults to ``torch.bfloat16``.

    - ``torch.bfloat16``
    - ``torch.float16``
    - ``torch.float32``

  - ``rpl_reduce_dtype`` - The torch data type to use for ``all_reduce``
    operations in RowParallelLinear layers. Defaults to the
    ``torch_dtype``.
  - ``async_mode`` - Whether to use asynchronous mode for inference.
    Defaults to ``false``.
  - ``save_sharded_checkpoint`` - Whether to save the sharded weights in
    the compiled checkpoint. If this option is disabled, NxD Inference
    shards the weights during model load. Defaults to ``true``.
  - ``logical_nc_config`` - The Logical NeuronCore Configuration (LNC).
    On Trn1 and Inf2, this defaults to ``1``. On Trn2, this defaults to ``2``.
    You can also configure LNC with the ``NEURON_LOGICAL_NC_CONFIG`` environment
    variable. For more information about LNC, see :ref:`logical-neuroncore-config`.

    - Note: If you use Trn2 with NxD Inference v0.1 (Neuron 2.21), you must
      specify LNC=2 by setting ``logical_neuron_cores=2`` in NeuronConfig.
      The ``logical_neuron_cores`` attribute is deprecated in NxD Inference v0.2
      and later.

  - ``skip_sharding`` - Whether to skip weight sharding during compilation.
    You can use this option if the compiled checkpoint path already
    includes sharded weights for the model. Defaults to ``false``.
  - ``weights_to_skip_layout_optimization`` - The list of weight names
    to skip during weight layout optimization.
  - ``skip_warmup`` - Whether to skip warmup during model load. To improve
    the performance of the first request sent to a model, NxD Inference
    warms up the model during load. Defaults to ``false``.

- Distributed configuration

  - ``tp_degree`` - The number of Neuron cores to parallelize across
    using tensor parallelism. Defaults to ``1``.

    - The number of attention heads needs to be divisible by the
      tensor-parallelism degree.
    - The total data size of model weights and key-value caches needs to
      be smaller than the tensor-parallelism degree multiplied by the
      amount of HBM memory per Neuron core.

      - On trn2, each Neuron core has 24GB of memory (with
        ``logical_nc_config`` set to ``2``).
      - On inf2/trn1, each Neuron core has 16GB of memory.

    - The Neuron runtime supports the following tensor-parallelism
      degrees:

      - trn2: 1, 2, 4, 8, 16, 32, and 64 (with ``logical_nc_config``
        set to ``2``)
      - inf2: 1, 2, 4, 8, and 24
      - trn1: 1, 2, 8, 16, and 32

- Attention

  - ``flash_decoding_enabled`` - Whether to enable flash decoding.
    Defaults to ``false``.
  - ``fused_qkv`` - Whether to fuse the query (Q), key (K), and value
    (V) weights in the models attention layers. This option improves
    performance by using larger matrices. Defaults to ``false``.
  - ``sequence_parallel_enabled`` - Whether to use sequence parallelism,
    which splits tensors along the sequence dimension. Defaults to
    ``false``.
  - ``qk_layernorm`` - Whether to enable QK layer normalization.
    Defaults to ``false``.

- On-device sampling

  - ``on_device_sampling_config`` - The on-device sampling configuration
    to use. Specify this config to enable on-device sampling. This
    config is an ``OnDeviceSamplingConfig``, which has the following
    attributes:

    - ``do_sample`` - Whether to use multinomial sampling (true) or
      greedy sampling (false). Defaults to ``false``.
    - ``top_k`` - The top-k value to use for sampling. Defaults to
      ``1``.
    - ``dynamic`` - Whether to enable dynamic sampling. With dynamic
      sampling, you can pass different ``top_k``, ``top_p``, and
      ``temperature`` values to the ``forward`` call to configure
      sampling for each input in a batch. Defaults to ``false``.
    - ``deterministic`` - Whether to enable deterministic sampling.
      Defaults to ``false``.
    - ``global_topk`` - The global topK value to use. Defaults to
      ``256``.

- Bucketing

  - ``enable_bucketing`` - Whether to enable bucketing. Defaults to
    ``false``. You can specify the buckets to use with the
    ``context_encoding_buckets`` and ``token_generation_buckets``
    attributes. If you don't specify the buckets to use, NxDI
    automatically selects buckets based on the following logic.

    - Context encoding: Powers of two between 128 and the max context
      length.

      - Note: Max context length is equivalent to sequence length by
        default.

    - Token generation: Powers of two between 128 and the maximum
      sequence length.

  - ``context_encoding_buckets`` - The list of bucket sizes to use for
    the context encoding model.
  - ``token_generation_buckets`` - The list of bucket sizes to use for
    the token generation model.

- Quantization

  - ``quantized`` - Whether the model weights are quantized. Defaults to
    ``false``.
  - ``quantized_checkpoints_path`` - The path to the quantized
    checkpoint. To quantize the model and save it to this path, use
    NeuronApplicationBase's ``save_quantized_state_dict`` function.
    Specify one of the following:

    - A folder path. During quantization, NxD Inference
      saves the quantized model in safetensors format to this folder. To
      use a quantized model from a folder, it can be in safetensors or
      pickle format.
    - A file path to a quantized model file in pickle format.

  - ``quantization_dtype`` - The data type to use for quantization.
    Choose from the following options. Defaults to ``int8``.

    - ``int8`` - 8 bit int.
    - ``f8e4m3`` - 8-bit float with greater precision and less range.

      - Important: To use ``f8e4m3`` for quantization, you must set the
        ``XLA_HANDLE_SPECIAL_SCALAR`` environment variable to ``1``.

    - ``f8e5m2`` - 8-bit float with greater range and less precision.

  - ``quantization_type`` - The type of quantization to use. Choose from
    the following options. Defaults to ``per_tensor_symmetric``.

    - ``per_tensor_symmetric``
    - ``per_channel_symmetric``

  - ``modules_to_not_convert`` - Specify a list of modules to be not quantized. Also, required when running inference on custom quantized models(using external libraries) where certain layers are left in full precision. Example: ["lm_head", "layers.0.self_attn", "layers.1.mlp", ...].
    Defaults to None (meaning all modules will be quantized)

  - ``draft_model_modules_to_not_convert`` - Specify a list of modules in full precision when working with fused speculation. If no layers are required, add all layers in the list. Example: ["lm_head", "layers.0.self_attn", "layers.1.mlp", ...].
    This is only required in the case of fused speculation.

- KV cache quantization

  - ``kv_cache_quant`` - Whether to quantize the KV cache. When enabled,
    the model quantizes the KV cache to the ``torch.float8_e4m3fn`` data
    type. Defaults to ``false``.

    - Important: To use ``kv_cache_quant``, you must set the
      ``XLA_HANDLE_SPECIAL_SCALAR`` environment variable to ``1``.

- Kernels

  - ``attn_kernel_enabled`` - Whether to enable the flash attention
    kernel when supported. Defaults to ``false``.
  - ``qkv_kernel_enabled`` - Whether to enable the fused QKV kernel. To
    use this option, you must set ``fused_qkv`` to ``true`` and ``torch_dtype``
    to ``torch.bfloat16``. Defaults to ``false``.
  - ``mlp_kernel_enabled`` - Whether to enable the MLP kernel. To use this
    option, you must set ``torch_dtype`` to ``torch.bfloat16``. Defaults
    to ``false``.
  - ``quantized_mlp_kernel_enabled`` - Whether to enable the quantized
    MLP kernel, which uses FP8 compute to improve performance. To use this
    option, you must set ``mlp_kernel_enabled`` to ``true``. Defaults to ``false``.
  - ``rmsnorm_quantize_kernel_enabled`` - Whether to enable the
    quantized RMS norm kernel. Defaults to ``false``.

- Continuous batching

  - ``is_continuous_batching`` - Whether to enable continuous batching.
    Defaults to ``false``.
  - ``max_batch_size`` - The maximum batch size to use for continuous
    batching. Defaults to ``batch_size``.
  - ``ctx_batch_size`` - The maximum batch size to use for the context
    encoding model in continuous batching. Defaults to ``batch_size``.
  - ``tkg_batch_size`` - The maximum batch size to use for the token
    generation model in continuous batching. Defaults to ``batch_size``.

- Speculative decoding

  - ``speculation_length`` - The number of tokens to generate with the
    draft model before checking work with the primary model. Set this
    value to a positive integer to enable speculation. Defaults to
    ``0``.
  - ``spec_batch_size`` - The batch size to use for speculation.
    Defaults to ``batch_size``.
  - ``enable_eagle_speculation`` - Whether to enable EAGLE speculation,
    where the previous hidden state is passed to a specialized target
    model to improve performance. Defaults to ``false``.
  - ``enable_eagle_draft_input_norm`` - Whether to perform input
    normalization in the EAGLE draft model. Defaults to ``false``.
  - ``enable_fused_speculation`` - Whether to enable fused speculation,
    where the target and draft model are fused into a single compiled
    model to improve performance. Fused speculation is enabled by
    default if ``enable_eagle_speculation`` is true. Otherwise, this
    defaults to ``false``.

- Medusa decoding - Medusa is a speculation method that uses multiple
  smaller LM heads to perform speculation.

  - ``is_medusa`` - Whether to use Medusa decoding. Defaults to
    ``false``
  - ``medusa_speculation_length`` - The number of tokens to generate
    with the Medusa heads before checking work with the primary model.
    Set this value to a positive integer. Defaults to ``0``.
  - ``num_medusa_heads`` - The number of LM heads to use for Medusa.
    Defaults to ``0``.
  - ``medusa_tree`` - The Medusa tree to use. For an example, see
    ``medusa_mc_sim_7b_63.json`` in the ``examples`` folder.



- Multi-LoRA serving

  - ``lora_config`` - The multi-lora serving configuration to use. Defaults to ``none``. Specify this config to enable multi-LoRA serving. This
    config is ``LoraServingConfig``, which has the following
    attributes:

    - ``max_loras`` - The maximum number of concurrent LoRA adapters 
      in device memory. Defaults to ``1``.
    - ``lora_ckpt_paths`` - The checkpoint paths for LoRA adapters with key-value pairs. The key is the adapter ID and the value is the local path of the LoRA adapter checkpoint.
    - ``lora_memory_transpose`` - Transpose memory layout to optimize 
      inference performance. Defaults to ``True``.
    - ``lora_shard_linear_layer`` - Shard the linear layer across TP group to 
      reduce memory consumption at the cost of communication overehead. 
      Defaults to ``False``.


- Compilation configuration

  - ``cc_pipeline_tiling_factor`` - The pipeline tiling factor to use
    for collectives. Defaults to ``2``.

InferenceConfig
~~~~~~~~~~~~~~~

InferenceConfig contains a NeuronConfig and model configuration
attributes.


.. _initialization-1:

Initialization
^^^^^^^^^^^^^^

You can pass attributes through keyword args, or provide a
``load_config`` hook that is called during initialization to load the
configuration attributes.

InferenceConfig is compatible with HuggingFace ``transformers``. To use
a model from HuggingFace ``transformers``, you can populate an
InferenceConfig with the attributes from the model's PretrainedConfig,
which is stored in ``config.json`` in the model checkpoint.

::

   from neuronx_distributed_inference.models.llama import (
       LlamaInferenceConfig,
       LlamaNeuronConfig
   )
   from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

   model_path = "/home/ubuntu/models/Meta-Llama-3.1-8B"

   neuron_config = LlamaNeuronConfig()
   config = LlamaInferenceConfig(
       neuron_config,
       load_config=load_pretrained_config(model_path),
   )

.. _attributes-1:

Attributes
^^^^^^^^^^

An InferenceConfig includes ``neuron_config`` and any other attributes
that you set during initialization.

- ``neuron_config`` - The NeuronConfig for this inference config.
- ``fused_spec_config`` - The FusedSpecNeuronConfig for this inference
  config. Provide a fused spec config if using fused speculation.
- ``load_config`` - The ``load_config`` hook to run during
  initialization. You can provide a load config hook to load
  configuration attributes from another source. To load from a
  HuggingFace PretrainedConfig, pass the load config hook returned by
  ``load_pretrained_config``. The ``load_pretrained_config`` hook
  provider takes the model path as its argument.

InferenceConfig also supports an attribute map, which lets you configure
additional names or aliases for attributes. When you get or set an
attribute by an alias, you retrieve or modify the value of the original
attribute. When you initialize an InferenceConfig from a HuggingFace
PretrainedConfig, it automatically inherits the attribute map from that
PretrainedConfig.

.. _functions-1:

Functions
^^^^^^^^^

- ``InferenceConfig(neuron_config, load_config=None, **kwargs)`` -
  Initializes an InferenceConfig.
- ``load_config(self)`` - Loads the config attributes. This function
  does nothing by default; subclasses can override it to provide a
  model-specific implementation. This function is called during
  initialization unless a ``load_config`` hook is provided.
- ``get_required_attributes(self)`` - Returns the list of attribute
  names that must be present in this config for it to validate during
  initialization. This function returns an empty list by default;
  subclasses can override it to require model-specific attributes to be
  present.
- ``validate_config(self)`` - Checks that the config is valid. This
  function is called during initialization. By default, this function
  checks that the attributes returned by ``get_required_attributes`` are
  present. Subclasses can override this function to implement
  model-specific validation.
- ``save(self, model_path)`` - Serializes the config to a JSON file,
  ``neuron_config.json`` in the given model path.
- ``to_json_file(self, json_file)`` - Serializes the config to the given
  JSON file.
- ``to_json_string(self)`` - Serializes the config to a string in JSON
  format.
- ``load(cls, model_path, **kwargs)`` - Loads the config from the
  ``neuron_config.json`` file in the given model path. You can specify
  ``kwargs`` to override attributes in the config.
- ``from_json_file(cls, json_file, **kwargs)`` - Loads the config from
  the given JSON file. You can specify ``kwargs`` to override attributes
  in the config.
- ``from_json_string(cls, json_string, **kwargs)`` - Loads the config
  from the given JSON string. You can specify ``kwargs`` to override
  attributes in the config.
- ``get_neuron_config_cls(cls)`` - Returns the NeuronConfig class type
  to use for this InferenceConfig. This function returns
  ``NeuronConfig`` by default; subclasses can override this function to
  configure a specific NeuronConfig subclass to use.

MoENeuronConfig
~~~~~~~~~~~~~~~

A NeuronConfig subclass for mixture-of-experts (MoE) models. This config
includes attributes specific to MoE models. MoE model configurations, such
as DbrxNeuronConfig, are subclasses of MoENeuronConfig.

.. _initialization-2:

Initialization
^^^^^^^^^^^^^^

Pass the attributes as keyword args.

.. _functions-2:

Functions
^^^^^^^^^

- ``MoENeuronConfig(**kwargs)`` - Initializes an MoENeuronConfig with
  attributes from ``kwargs``.

.. _attributes-2:

Attributes
^^^^^^^^^^

- ``capacity_factor`` - The capacity factor to use when allocating
  tokens across experts. When an expert is at capacity, tokens allocated
  to that expert are dropped until that expert has capacity again.
  Defaults to ``None``, which means that NxDI waits until an expert has
  capacity, and no tokens are dropped.
- ``glu_mlp`` - Whether to use a Gated Linear Unit in the MLP. Defaults
  to ``false``.

FusedSpecNeuronConfig
~~~~~~~~~~~~~~~~~~~~~

A configuration for a model that uses fused speculation, which is a speculative
decoding feature where the target and draft models are compiled into a combined model to improve
performance. For more information, see :ref:`nxd-fused-speculative-decoding`.

.. _attributes-3:

Attributes
^^^^^^^^^^

- ``worker_cls`` - The model class to use for fused speculation. This
  class should be a subclass of NeuronBaseModel.
- ``draft_config`` - The InferenceConfig for the draft model.
- ``draft_model_path`` - The path to the draft model checkpoint.

Generation
----------

HuggingFaceGenerationAdapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NxD Inference supports running inference with the HuggingFace ``generate``
inference. To use HuggingFace-style generation, create a
HuggingFaceGenerationAdapter that wraps a Neuron application model.
Then, you can call ``generate`` on the adapted model.

::

   generation_model = HuggingFaceGenerationAdapter(neuron_model)
   outputs = generation_model.generate(
       inputs.input_ids,
       attention_mask=inputs.attention_mask,
       generation_config=generation_config
   )

Models
------

NxD Inference provides a :ref:`model hub<nxdi-model-reference>` with production
ready models. You can use these existing models to run inference, or use them as
reference implementations when you develop your own models on Neuron. All model
inherit from base classes that provide a basic set of functionality that
is common to all models.

NeuronApplicationBase
~~~~~~~~~~~~~~~~~~~~~

NeuronApplicationBase is the base class for all application models,
including NeuronBaseForCausalLM. NeuronApplicationBase provides
functions to compile and load models. This class extends
``torch.nn.Module``. Application models are the entry point to running
inference with NxD Inference. You can extend this class to define new
application models that implement use cases in addition to causal LM.

.. _attributes-4:

Attributes
^^^^^^^^^^

- ``config`` - The InferenceConfig for this model.
- ``neuron_config`` - The NeuronConfig for this model.
- ``model_path`` - The model path for this model.
- ``models`` - The list of models that make up this application model.
  These models are instances of ModelWrapper. Add models to this list to
  compile them with ``compile``.
- ``is_compiled`` - Whether this model is compiled.
- ``is_loaded_to_neuron`` - Whether this model is loaded to the Neuron
  device.

.. _functions-3:

Functions
^^^^^^^^^

- ``NeuronApplicationBase(self, model_path, config=None, neuron_config=None)``
  - Initializes an application model from the given model path, and
  optionally the given InferenceConfig (``config``) and NeuronConfig
  (``neuron_config``). If no InferenceConfig is provided, this function
  loads the config from the given model path.
- ``compile(self, compiled_model_path, debug=False)`` - Compiles this
  model for Neuron and saves the compiled model to the given path. This
  function compiles all models added to ``self.models``. This function
  also shards the weights for the model. To produce HLO files that have
  source annotations enabled for debugging, set ``debug`` to ``True``. When ``debug`` is enabled, HLOs contain following attributes for each computation: ``op_type``, ``op_name``, ``source_file``, and ``source_line``.
- ``load(self, compiled_model_path)`` - Loads the compiled model from
  the given path to the Neuron device. This function also loads the
  model weights to the Neuron device.
- ``load_weights(self, compiled_model_path)`` - Loads the model weights
  from the given path to the Neuron device. You can call this function
  to load new weights without reloading the entire model.
- ``shard_weights(self, compiled_model_path)`` - Shards the model's
  weights and serializes the sharded weights to the given path.
- ``forward(self, **kwargs)`` - The forward function for this
  application model. This function must be implemented by subclasses.
- ``validate_config(cls, config)`` - Checks whether the config is valid
  for this model. By default, this function requires that
  ``neuron_config`` is present. This function can be implemented by
  subclasses to provide model-specific validation.
- ``get_compiler_args(self)`` - Returns the Neuron compiler arguments to
  use when compiling this model. By default, this returns no compiler
  arguments. This function can be implemented by subclasses to use
  model-specific compiler args.
- ``to_cpu(self)`` - Allows inference to be run entirely on CPU. Use this 
  in place of the ``compile`` and ``load`` functions. Note that CPU inference 
  doesn't currently work for configurations that use kernels.
- ``get_state_dict(cls, model_path, config)`` - Gets the state dict for
  this model. By default, this function loads the state dict from the
  given model path. This function calls the class'
  ``convert_hf_to_neuron_state_dict`` function to convert the state dict
  according to the specific model. Subclasses can override this function
  to provide custom state dict loading.

  - When loading the state dict, this function replaces keys that start
    with the class' ``_STATE_DICT_MODEL_PREFIX`` value with the class'
    ``_NEW_STATE_DICT_MODEL_PREFIX`` value. Subclasses can set these
    values to update the state dict keys accordingly.

- ``convert_hf_to_neuron_state_dict`` - Converts a state dict from HF
  format to the format expected by Neuron. By default, this function
  returns the state dict without modifying it; subclasses can override
  this to provide custom conversion for each model.
- ``save_quantized_state_dict(cls, model_path, config)`` - Quantizes the
  model's state dict and saves the quantized checkpoint to the
  ``quantized_checkpoint_path`` from the given config's NeuronConfig.
- ``generate_quantized_state_dict(cls, model_path, config)`` - Generates
  the quantized state dict for this model. This function loads the
  HuggingFace model from the given model path in order to quantize the
  model. Then, this function passes the quantized model to
  ``prepare_quantized_state_dict`` to generate the state dict.
  Subclasses can override this function to customize quantization.
- ``prepare_quantized_state_dict(cls, hf_model_quant)`` - Prepares the
  quantized state dict for the model. By default, this function converts
  the state dict from qint8 to int8. Subclasses can override this
  function to customize quantization.
- ``load_hf_model(model_path)`` - Loads the equivalent HuggingFace model
  from the given model path. Subclasses must implement this function to
  use quantization or to generate expected outputs when evaluating
  accuracy with ``accuracy.py``.
- ``reset(self)`` - Resets the model state. By default, this function
  does nothing; subclasses can implement it to provide custom behavior.

NeuronBaseForCausalLM
~~~~~~~~~~~~~~~~~~~~~

NeuronBaseForCausalLM is the base application class that you use to generate
text with causal language models. This class extends NeuronApplicationBase.
You can extend this class to run text generation in custom models.

.. _attributes-5:

Attributes
^^^^^^^^^^

- ``kv_cache_populated`` - Whether the KV cache is populated.

.. _functions-4:

Functions
^^^^^^^^^

- ``NeuronBaseForCausalLM(self, *args, **kwargs)`` - Initializes the
  NeuronApplicationBase and configures the models used in this LM
  application, including context encoding, token gen, and others, based
  on the given NeuronConfig.
- ``forward(self, input_ids=None, seq_ids=None, attention_mask=None, position_ids=None, sampling_params=None, prev_hidden=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, medusa_args=None, return_dict=None, input_capture_hook=None)``
  - The forward function for causal LM. This function routes the forward
  pass to the correct sub-model (such as context encoding or token
  generation) based on the current model state. If an ``input_capture_hook``
  function is provided, the forward function calls the hook with the model
  inputs as arguments.
- ``reset(self)`` - Resets the model for a new batch of inference. After
  the model is reset, a subsequent run will invoke the context encoding
  model.
- ``reset_kv_cache(self)`` - Resets the KV cache by replacing its key
  values with zeroes.

NeuronBaseModel
~~~~~~~~~~~~~~~

NeuronBaseModel is the base class for all models. This class extends
``torch.nn.Module``. In instances of NeuronBaseModel, you define the
modules, such as attention, MLP, and decoder layers, that make up a model.
You can extend this class to define custom decoder models.

.. _attributes-6:

Attributes
^^^^^^^^^^

- ``sampler`` - The sampler to use for on-device sampling.
- ``kv_mgr`` - The KV cache manager to use to manage the KV cache.
- ``sequence_dimension`` - The dimension for sequence parallelism.

.. _functions-5:

Functions
^^^^^^^^^

- ``NeuronBaseModel(config, optimize_inference=True)`` - Initializes the
  Neuron model from the given config. If ``optimize_inference`` is true,
  then this initializes a KV cache manager and sampler (if on-device
  sampling).
- ``setup_attr_for_model(self, config)`` - Initializes the following
  attributes for the model. These attributes are used by modules within
  the model. Subclasses must implement this function to set these
  attributes from the config.

  - ``on_device_sampling``
  - ``tp_degree``
  - ``hidden_size``
  - ``num_attention_heads``
  - ``num_key_value_heads``
  - ``max_batch_size``
  - ``buckets``

- ``init_model(self, config)`` - Initializes the following modules for
  the model. Subclasses must implement this function.

  - ``embed_tokens``
  - ``layers``
  - ``norm``
  - ``lm_head``

- ``forward(self, input_ids, attention_mask, position_ids, seq_ids, accepted_indices=None, current_length=None, medusa_mask=None, scatter_index=None)``
  - The forward function for this model.

ModelWrapper
~~~~~~~~~~~~

Wraps a model to prepare it for compilation. Neuron applications, such
as NeuronBaseForCausalLM, use this class to prepare a model for
compilation. ModelWrapper defines the inputs to use when tracing the
model during compilation.

To define a custom model with additional model inputs, you can extend ModelWrapper
and override the ``input_generator`` function, which defines the inputs for tracing.

.. _functions-6:

Functions
^^^^^^^^^

- ``ModelWrapper(config, model_cls, tag, compiler_args)`` - Initializes
  a model wrapper from a given config and model class. This model class
  is used to compile the model with the given compiler args. The tag is
  used to identify the compiled model in the application.
- ``input_generator(self)`` - Returns a list of input tensors to use to trace
  the model for compilation. When you trace and compile a model, the trace captures
  only the code paths that are run with these inputs. To support different inputs and
  code paths based on configuration options, provide configuration-specific inputs
  in ``input_generator``.
