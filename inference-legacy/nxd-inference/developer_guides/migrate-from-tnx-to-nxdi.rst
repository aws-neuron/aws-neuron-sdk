.. _nxdi_migrate_from_tnx:


Migrating from Transformers NeuronX to  NeuronX Distributed(NxD) Inference
==========================================================================


.. contents:: Table of contents
   :local:
   :depth: 2


For customers who are currently using Transformers NeuronX, this migration guide explains the steps involved in
migrating from Transformers NeuronX to NxD Inference library.  


How is writing modeling code different in NxD Inference?
---------------------------------------------------------

In Transformers NeuronX, you write modeling code in HLO format using a Python HLO interface. In NeuronX Distributed Inference, you write modeling code in native PyTorch and Python, and the library converts it to HLO for you. 
This change makes it easier to develop models to run on Neuron, because you can start from existing Pytorch or Python modeling code.


How can I migrate from Transformers NeuronX to use NxD Inference with vLLM?
----------------------------------------------------------------------------

Transformers NeuronX library currently supports Llama and Mistral model architectures with vLLM integration. If you are using one of these models, like Llama 3.1, Llama 3, Llama 2, or Mistral-7b-V2, you can migrate to use NxD Inference library with vLLM using the following steps:


Update Environment Variable to force vLLM to use NxD Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As vLLM currently supports both Transformers NeuronX and NeuronX Distributed Inference libraries for the Llama and Mistral models, you need to update the following environment variable in the inference scripts to force vLLM to use NxD Inference.

.. code:: 

    # Force vLLM framework to use neuronx-distributed-inference
    os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"


Compiling and loading the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transformers NeuronX uses Neuron Persistent Cache to load a pre-compiled model so that there is no additional delay in compilation when loading the model on vLLM.  NxD Inference currently does not support Neuron Persistent Cache but provides the following way to load a pre-compiled model in NeuronX Distributed Inference.

For production use cases where customer wants to avoid compiling the model in NxD Inference for the first time, users can set the environment variable ``NEURON_COMPILED_ARTIFACTS`` which points to pre-compiled artifacts directory to avoid the compilation time. If the artifacts are not present within the specified directory, then compilation of the model would be triggered as a fallback mechanism and will store the artifacts by default in ``neuron-compiled-artifacts/{unique_hash}/``


Features currently not supported in NxD Inference through vLLM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NxD Inference doesn't yet support the following features that TNx supports in vLLM integration.

* Multi-Node Inference
* Persistent Cache
* concurrency > 1 support for speculation

Users can use exactly the same set of parameters to test out vLLM with NxD Inference library as they specify with Transformers NeuronX with the exception of ``override_neuron_config`` . Both Transformers NeuronX and NxD Inference allows overriding available NeuronConfig, but not all NeuronConfig parameters that are available with Transformers NeuronX are still valid/applicable in NxD Inference. Refer to the :ref:`neuron_config_migration_tnx_nxdi` to migrate your ``override_neuron_config`` params from Transformers NeuronX to NxD Inference.

Serialization support
----------------------

In both libraries, you serialize the compiled model, so you can use the model in subsequent runs without compiling it each time.

In Transformers NeuronX, the save function does not serialize sharded weights by default, and you can enable this functionality with the ``sharded_weights`` flag. In NeuronX Distributed Inference, the ``compile`` function serializes sharded weights by default, and you can disable this functionality with the ``save_sharded_checkpoint`` flag in ``NeuronConfig``.

Tranformers NeuronX
^^^^^^^^^^^^^^^^^^^

.. code::

    # Create and compile the Neuron model
    neuron_config = NeuronConfig()
    model_neuron = LlamaForSampling.from_pretrained(
        'openlm-research/open_llama_3b',
        batch_size=1,
        tp_degree=8,
        n_positions=128,
        neuron_config=neuron_config
    )

    # Compile the model.
    model_neuron.to_neuron()

    # Save the presharded weights and compiled artifacts to a directory.
    model_neuron.save('llama-artifacts', sharded_weights=True)

NeuronX Distributed Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code::
    
    model_path = "/home/ubuntu/models/open_llama_3b"
    compiled_model_path = "/home/ubuntu/compiled_models/open_llama_3b"

    neuron_config = NeuronConfig(
        batch_size=1,
        tp_degree=8,
        seq_len=128
    )

    config = LlamaInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path)
    )

    model = NeuronLlamaForCausalLM(model_path, config)

    # Compile the model, shard the weights, and save to the given path.
    model.compile(compiled_model_path)

Models supported in Transformers NeuronX and NxD Inference model hubs
----------------------------------------------------------------------

The following table depicts the list of models currently supported by TNx and their status in the NxD Inference library. For a more detailed list of models currently supported in NeuronX Distributed Inference, please refer to :ref:`NxD Inference model hub guide <nxdi-model-reference>`



+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| Model                      | Transformers NeuronX (TNx)               | NxD Inference (NxDI)                              |
+                            +--------------------+---------------------+------------------+--------------------------------+
|                            | supported in TNx   | vLLM Support (TNx)  | supported in NxDI| vLLM Support (NxD Inference)   |
+============================+====================+=====================+==================+================================+
| BLOOM                      | Yes                | No                  | No               | No                             |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| GPT2                       | Yes                | No                  | No               | No                             |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| GPT-J                      | Yes                | No                  | No               | No                             |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| GPT-Neox                   | Yes                | No                  | No               | No                             |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| Llama 2                    | Yes                | Yes                 | Yes              | Yes                            |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| Llama 3                    | Yes                | Yes                 | Yes              | Yes                            |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| Llama 3.1                  | Yes                | Yes                 | Yes              | Yes                            |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| Llama 3.2 (1B and 3B)      | Yes                | Yes                 | Yes              | Yes                            |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| Llama 3.2 (11B and 90B)    | No                 | No                  | Yes              | Yes                            |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| Mistral-V2                 | Yes                | Yes                 | Yes              | Yes                            |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| Mixtral                    | Yes                | No                  | Yes              | Yes                            |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+
| DBRX                       | No                 | No                  | Yes              | Yes                            |
+----------------------------+--------------------+---------------------+------------------+--------------------------------+




Onboarding custom or private models with NxD Inference
-------------------------------------------------------

If you need model support for one of the models not currently supported in NxD Inference or if you have a private model that you currently implemented support in Transformers Neuronx,
you need to implement the model using NxD Inference library.  You can use the :ref:`nxdi-onboarding-models` guide.

.. _neuron_config_migration_tnx_nxdi:

Neuron Config Migration
-----------------------

There are differences in Neuron Config parameters in Transformers NeuronX and :ref:`NxD Inference <nxd-inference-api-guide-neuron-config>` libraries.  
If you use TNx directly without vLLM, or if you use the ``override_neuron_config`` param in vLLM with TNx, then you must update config parameters according to the following table.


.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Transformers NeuronX parameter
     - NxD Inference parameter
     - Notes
   * - sparse_attn
     - N/A
     - 
   * - quant.quant_dtype
     - quantization_dtype
     - To use quantization, set ``quantized`` to True, and provide the ``quantized_checkpoints_path`` where the quantized model is stored (or will be stored).
   * - quant.dequant_dtype
     - torch_dtype
     - NxD Inference uses the inference dtype as the dequant dtype.
   * - quant.quantize_method
     - quantization_type
     - 
   * - quant.quantize_attn
     - N/A
     - 
   * - quant.no_quantize_list
     - N/A
     - 
   * - kv_cache_quant.quant_dtype
     - N/A
     - NxD Inference uses FP8 (torch.float8_e4m3fn) for KV cache quantization. To use KV cache quantization, set ``kv_cache_quant`` to True.
   * - kv_cache_quant.dequant_dtype
     - torch_dtype
     - NxD Inference uses the inference dtype as the dequant dtype.
   * - kv_cache_quant.quantize_method
     - N/A
     - NxD Inference uses direct cast.
   * - continuous_batching.max_num_seqs
     - max_batch_size
     - To use continuous batching, set ``is_continous_batching`` to True, and set ``tkg_batch_size`` to the max batch size.
   * - continuous_batching.max_model_len
     - seq_len
     - 
   * - continuous_batching.optimized_paged_attention
     - N/A
     - 
   * - continuous_batching.block_size
     - N/A
     - 
   * - continuous_batching.num_blocks
     - N/A
     - 
   * - attention_layout
     - N/A
     - NxD Inference uses BHSD layout.
   * - collectives_layout
     - N/A
     - NxD Inference uses BHSD layout.
   * - cache_layout
     - N/A
     - NxD Inference uses BHSD layout.
   * - padding_side
     - padding_side
     - NxD Inference defaults to padding on the right side.
   * - group_query_attention
     - N/A
     - 
   * - sequence_parallel_norm
     - sequence_parallel_enabled
     - 
   * - sequence_parallel_norm_threshold
     - N/A
     - 
   * - bf16_rms_norm
     - N/A
     - NxD Inference upcasts RMS norm inputs to fp32.
   * - on_device_embedding
     - N/A
     - 
   * - on_device_generation
     - on_device_sampling_config
     - 
   * - on_device_generation.max_length
     - seq_len
     - NxD Inference uses the model's sequence length.
   * - on_device_generation.do_sample
     - on_device_sampling_config.do_sample
     - 
   * - on_device_generation.top_k
     - on_device_sampling_config.top_k
     - NxD Inference supports top_k through dynamic sampling. Pass the top_k values to the model inputs.
   * - on_device_generation.top_p
     - N/A
     - NxD Inference supports top_p through dynamic sampling. Pass the top_p values to the model inputs.
   * - on_device_generation.temperature
     - N/A
     - NxD Inference supports temperature through dynamic sampling. Pass the temperature values to the model inputs.
   * - on_device_generation.top_p_min_tokens
     - N/A
     - NxD Inference defaults to a minimum of 1 token.
   * - on_device_generation.global_top_k
     - on_device_sampling_config.global_topk
     - 
   * - on_device_generation.eos_token_id
     - N/A
     - NxD Inference sampling treats EOS like any other token.
   * - on_device_generation.dynamic
     - on_device_sampling_config.dynamic
     - 
   * - on_device_generation.deterministic
     - on_device_sampling_config.deterministic
     - 
   * - on_device_generation.per_batch_line
     - N/A
     - 
   * - all_reduce_dtype
     - rpl_reduce_dtype
     - NxD Inference applies this dtype to only the all_reduce in attention's ``o_proj`` layer.
   * - cast_logits_dtype
     - N/A
     - 
   * - fuse_qkv
     - fused_qkv
     - 
   * - qkv_tiling
     - N/A
     - 
   * - weight_tiling
     - N/A
     - 
   * - mlp_in_weight_tiling_permute_order
     - N/A
     - 
   * - mlp_out_weight_tiling_permute_order
     - N/A
     - 
   * - mlp_out_weight_transpose
     - N/A
     - 
   * - log_softmax_scores
     - N/A
     - 
   * - shard_over_sequence
     - flash_decoding_enabled
     - 
   * - duplicate_q_weight_sos
     - N/A
     - 
   * - output_all_logits
     - N/A
     - 
   * - fused_rmsnorm_qkv
     - qkv_kernel_enabled
     - 
   * - fused_rmsnorm_mlp
     - mlp_kernel_enabled
     - 
   * - attn_output_transposed
     - N/A
     - 
   * - compilation_worker_count
     - N/A
     - 
