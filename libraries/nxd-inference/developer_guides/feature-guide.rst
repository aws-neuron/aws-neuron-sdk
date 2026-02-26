.. _nxdi-feature-guide:

NxD Inference Features Configuration Guide
==========================================

NxD Inference (``neuronx-distributed-inference``) is
an open-source PyTorch-based inference library that simplifies deep learning
model deployment on AWS Inferentia and Trainium instances. Neuronx Distributed
Inference includes a model hub and modules that users can reference to
implement their own models on Neuron.


.. contents:: Table of contents
   :local:
   :depth: 2

Checkpoint compatibility with HuggingFace Transformers
------------------------------------------------------

Models included in the NxD Inference model hub are checkpoint-compatible with
HuggingFace Transformers. Supporting other checkpoint formats in NxD Inference is possible through converting the
obtained checkpoint to the standard HuggingFace Transformers checkpoint format.

.. _nxdi-checkpoint-support:

Checkpoint support
------------------

NxD Inference supports older PyTorch binary checkpoints
and newer `safetensors <https://github.com/huggingface/safetensors>`__
checkpoints. For improved load speed and reduced host memory
consumption, we recommend to always use safetensors by default. Both
regular and sharded variants of checkpoints are supported.

NxD Inference supports weights stored in the model path in the following
formats:

=========== ======= ============================
Format      Sharded File name
=========== ======= ============================
Safetensors No      model.safetensors
Safetensors Yes     model.safetensors.index.json
Pickle      No      pytorch_model.bin
Pickle      Yes     pytorch_model.bin.index.json
=========== ======= ============================

If your weights are in another format, you must convert them to one of
these formats before you can compile and load the model to Neuron. See
the following references for more information about these formats:

- Safetensors:

  - https://github.com/huggingface/safetensors
  - https://huggingface.co/docs/safetensors/en/convert-weights

- Pickle:

  - https://docs.python.org/3/library/pickle.html

Compiling models
----------------
To run a model on Neuron with NxD Inference, you compile Python code into
a NEFF file (Neuron Executable File Format), which you can load to Neuron
devices using the Neuron Runtime.

When you call ``compile()``, NxD Inference does the following:

1. Trace the Python code to produce an HLO file.
2. Use the Neuron Compiler to compile the HLO file into a NEFF.

During the trace process, the model code is traced based on a given sample
tensor for each input. As a result, model code should avoid dynamic logic
that depends on the input values in a tensor, because NxD Inference compiles
only the code path that is traced for the sample input tensor.

::

    # Configure, initialize, and compile a model.
    model = NeuronLlamaForCausalLM(model_path, config)
    model.compile(compiled_model_path)


.. _nxdi-neuron-persistent-cache:

Neuron Persistent Cache
------------------------

The Neuron Persistent Cache is enabled by default for NxD Inference library.
Model artifacts which have been compiled once will be cached and reused on
successive runs when possible. Model artifacts will only be reused when
compiling with the same compiler version (neuronx-cc), model configurations,
and compiler flags. Neuron Persistent Cache also includes other features, such as using an S3 bucket as
the cache backend. For more detailed information, see the
:ref:`Persistent cache documentation <neuron-caching>`


Serialization support
---------------------

When you compile a model with NxD Inference, the library
serializes the model to a given folder. After you have a serialized
model, you can load it directly to a Neuron device without needing to
compile again.

The compile function does not serialize sharded weights by default, and you can
enable this functionality with the ``save_sharded_checkpoint`` flag in
NeuronConfig. For more information on weights sharding, see :ref:`nxdi-weights-sharding-guide`.

Logical NeuronCore Configuration (LNC) support
----------------------------------------------
On Trn2 instances, Neuron supports Logical NeuronCore (LNC) configuration,
which combines multiple physical NeuronCores into a single logical
NeuronCore. On Trn2 instances, the Neuron SDK is optimized for LNC=2, which means
each NeuronCore visible to the Neuron SDK is two physical NeuronCores.

NxD Inference automatically chooses the correct LNC configuration
based on the target platform. To override the default LNC configuration,
you can set the ``NEURON_LOGICAL_NC_CONFIG`` environment variable, or set the
``logical_nc_config`` flag in NeuronConfig.

::

   neuron_config = NeuronConfig(logical_nc_config=2)

For more information about logical NeuronCore support, see
:ref:`logical-neuroncore-config`.

.. _nxdi-feature-guide-tensor-parallelism:

Tensor-parallelism support
--------------------------

For transformer decoders used in large language models,
tensor-parallelism is necessary as it provides a way to shard the
models' large weight matrices onto multiple NeuronCores, and having
NeuronCores working on the same matrix multiply operation
collaboratively. neuronx-distributed-inference's tensor-parallelism
support makes heavy use of collective operations such as all-reduce,
which is supported natively by the Neuron runtime.

There are some principles for setting tensor-parallelism degree (number
of NeuronCores participating in sharded matrix multiply operations) for
Neuron-optimized transformer decoder models.

1. The number of attention heads needs to be divisible by the
   tensor-parallelism degree.
2. The total data size of model weights and key-value caches needs to be
   smaller than the tensor-parallelism degree multiplied by the amount
   of memory per Neuron core.

   1. On Trn2, each Neuron core has 24GB of memory (with LNC2).
   2. On Inf2/Trn1, each Neuron core has 16GB of memory.

3. The Neuron runtime supports the following tensor-parallelism degrees:

   1. Trn2: 1, 2, 4, 8, 16, 32, and 64 (with LNC2)
   2. Inf2: 1, 2, 4, 8, and 24
   3. Trn1: 1, 2, 8, 16, and 32

Examples
~~~~~~~~

1. ``meta-llama/Meta-Llama-3.1-8B`` has 32 attention heads, and when
   running at batch size 1 and bfloat16 precision, the model requires
   about 16GB memory. Therefore, a ``trn1.2xlarge`` with 32GB device
   memory is sufficient.
2. ``meta-llama/Meta-Llama-3.1-70B`` has 64 attention heads, and when
   running at batch size 1 and bfloat16 precision, the model requires
   about 148GB memory. Therefore, it can run on 16 NeuronCores on one
   ``trn1.32xlarge`` using 256GB device memory.

.. _nxdi-feature-guide-sequence-parallelism:

Sequence Parallelism
--------------------
Sequence parallelism splits tensors across the sequence dimension to
improve performance. You can enable sequence parallelism by setting
``sequence_parallel_enabled=True`` in NeuronConfig.

::

   neuron_config = NeuronConfig(sequence_parallel_enabled=True)

Compile-time Configurations
---------------------------

NxD Inference models support a variety of compile-time
configurations you can use to tune model performance. For more
information, see the :ref:`nxd-inference-api-guide`.

Hugging Face generate() API support
-----------------------------------

NxD Inference models support the HuggingFace `generate()
API <https://huggingface.co/docs/transformers/main/en/main_classes/text_generation>`__
via the ``HuggingFaceGenerationAdapter`` class. This adapter wraps a
Neuron model to provide the HuggingFace generation interface.

NxD Inference's supports the following HuggingFace
generation modes:

- Greedy decoding — ``num_beams=1`` and ``do_sample=False``.
- Multinomial sampling — ``num_beams=1`` and ``do_sample=True``.
- Assisted (speculative) decoding — ``assistant_model`` or
  ``prompt_lookup_num_tokens`` are specified.

NxD Inference doesn't currently support other
HuggingFace generation modes such beam-search sampling.

Note: When you call ``generate``, the number of prompts must match the
``batch_size`` for the model, which is an attribute of NeuronConfig.

::

   neuron_config = NeuronConfig(batch_size=2)

Example
~~~~~~~

The following example demonstrates how to wrap a model with
HuggingFaceGenerationAdapter to call ``generate()``.

::

   from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

   # Init Neuron model, HuggingFace tokenizer, HuggingFace and generation config.


   # Run generation with HuggingFaceGenerationAdapter.
   generation_model = HuggingFaceGenerationAdapter(model)
   inputs = tokenizer(prompts, padding=True, return_tensors="pt")
   outputs = generation_model.generate(
       inputs.input_ids,
       generation_config=generation_config,
       attention_mask=inputs.attention_mask,
       max_length=model.neuron_config.max_length,
       **kwargs,
   )

   output_tokens = tokenizer.batch_decode(
       outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
   )

   print("Generated outputs:")
   for i, output_token in enumerate(output_tokens):
       print(f"Output {i}: {output_token}")

On-device Sampling Support
--------------------------

On-device sampling performs sampling logic on the Neuron device (rather
than on the CPU) to achieve better performance. To enable on device
sampling, provide an OnDeviceSamplingConfig for the
``on_device_sampling_config`` attribute in NeuronConfig.

::

   on_device_sampling_config = OnDeviceSamplingConfig(global_topk=256)
   neuron_config = NeuronConfig(on_device_sampling_config=on_device_sampling_config)

Dynamic Sampling
~~~~~~~~~~~~~~~~

With dynamic sampling, you can pass different ``top_k``, ``top_p``, and
``temperature`` values to the ``forward`` call to configure sampling for
each input in a batch. To enable dynamic sampling, provide an
OnDeviceSamplingConfig with ``dynamic=True``.

::

   on_device_sampling_config = OnDeviceSamplingConfig(dynamic=True)
   neuron_config = NeuronConfig(on_device_sampling_config=on_device_sampling_config)

To use dynamic sampling, pass a ``sampling_params`` tensor to the
forward function of the model. The ``sampling_params`` tensor has shape
``[batch_size, 3]``, where the three values per batch are ``top_k``,
``top_p``, and ``temperature``.

The following example demonstrates how to create ``sampling_params`` for
a batch with two inputs. In the first input, ``top_k=50``,
``top_p=0.5``, and ``temperature=0.75``. In the second input,
``top_k=5``, ``top_p=1.0``, and ``temperature=1.0``.

::

   sampling_params = torch.tensor([[50, 0.5, 0.75], [5, 1.0, 1.0]])

Greedy Sampling
~~~~~~~~~~~~~~~

By default, on-device sampling uses greedy sampling, where the model
picks the highest scoring token.

Multinomial (Top-K) Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With multinomial (top-k) sampling, the model picks one of the top
*k*-highest scoring tokens. To use on-device multinomial sampling, you
must enable dynamic sampling. You can configure the default ``top_k``
attribute in the OnDeviceSamplingConfig, or you can specify the
``top_k`` value in each call to the model's ``forward`` function.

::

   on_device_sampling_config = OnDeviceSamplingConfig(top_k=5)

Top-P Support in On-Device Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use top-p in on-device sampling, enable dynamic sampling, and specify
``top_p`` values in the ``sampling_params``.

Temperature Support in On-Device Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To adjust temperature in on-device sampling, enable dynamic sampling,
and specify ``temperature`` values in the ``sampling_params``.

.. _qkv-weight-fusion:

QKV Weight Fusion
-----------------

QKV weight fusion concatenates a model's query, key and value weight
matrices to achieve better performance, because larger matrices allow
for more efficient data movement and compute. You can enable QKV weight
fusion by setting ``fused_qkv=True`` in the NeuronConfig.

::

   neuron_config = NeuronConfig(fused_qkv=True)

.. _nxdi-bucketing:

Bucketing
---------

LLM inference is a generation process that can produce variable length
sequences. This poses a problem since the Neuron compiler produces
executables which expect statically shaped inputs and outputs. To make
LLMs work with different shapes, NxD Inference supports
buckets and applies padding wherever it is required. When you run
inference, NxD Inference automatically chooses the
smallest bucket that fits the input for optimal performance. For more
information about bucketing, see :ref:`torch-neuronx-autobucketing-devguide`.

Automatic Bucketing
~~~~~~~~~~~~~~~~~~~

When automatic bucketing is enabled, NxD Inference
automatically chooses buckets for each model according to the following
logic:

- Context encoding: Powers of two between 128 and the max context
  length.

  - Note: Max context length is equivalent to sequence length by
    default.

- Token generation: Powers of two between 128 and the maximum sequence
  length.

To enable automatic bucketing, set ``enable_bucketing=True`` in
NeuronConfig.

::

   neuron_config = NeuronConfig(enable_bucketing=True)

Configuring Specific Buckets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure specific buckets to further optimize inference based
on the input and output length distribution that you expect to process
with your model. In NeuronConfig, set ``enable_bucketing=True``, and
provide a list of bucket sizes in ``context_encoding_buckets`` and/or
``token_generation_buckets``.

::

   neuron_config = NeuronConfig(
       enable_bucketing=True,
       context_encoding_buckets=[1024, 2048, 4096],
       token_generation_buckets=[8192]
   )

.. _nxdi-quantization:

Quantization
------------

NxD Inference supports quantization, where model weights
and data are converted to a smaller data type to reduce memory bandwidth
usage, which improves model performance.

Note: Quantization slightly reduces accuracy due to using data types
with lower precision and/or lower range.

.. _nxdi-weight-quantization:

Model Weight Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~

NxD Inference supports quantizing model weights to the
following data types:

- INT8 (``int8``) - 8 bit int.
- FP8 - 8 bit float.

  - ``f8e4m3`` - 8-bit float with greater precision and less range.

    - Important: To use ``f8e4m3`` for quantization, you must set the
      ``XLA_HANDLE_SPECIAL_SCALAR`` environment variable to ``1``.

  - ``f8e5m2`` - 8-bit float with greater range and less precision.

NxD Inference supports the following quantization methods, which you specify with `quantization_type` in NeuronConfig:

- `per_tensor_symmetric`
- `per_channel_symmetric`

.. _example-1:

Example
^^^^^^^

The following example demonstrates how to quantize a model to INT8. To quantize
a model to a different data type, change the ``quantization_dtype`` config
attribute in ``NeuronConfig``.

::

   from neuronx_distributed_inference.models.config import NeuronConfig
   from neuronx_distributed_inference.models.llama.modeling_llama import (
       LlamaInferenceConfig,
       NeuronLlamaForCausalLM
   )
   from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

   model_path = "/home/ubuntu/models/Llama-3.1-8B"
   quantized_model_path = "/home/ubuntu/models/Llama-3.1-8B-quantized"

   neuron_config = NeuronConfig(
       quantized=True,
       quantized_checkpoints_path=quantized_model_path,
       quantization_dtype="int8",
       quantization_type="per_tensor_symmetric"
   )

   config = LlamaInferenceConfig(
       neuron_config,
       load_config=load_pretrained_config(model_path)
   )

   # Quantize the model and save it to `quantized_checkpoints_path`.
   NeuronLlamaForCausalLM.save_quantized_state_dict(model_path, config)

   # Compile, load, and use the model.
   model = NeuronLlamaForCausalLM(model_path, config)

.. _nxdi-kv-cache-quantization:

KV Cache Quantization
~~~~~~~~~~~~~~~~~~~~~

NxD Inference supports KV cache quantization, where the
model's KV cache is quantized to a smaller data type. When enabled, the
model quantizes the KV cache to the ``torch.float8_e4m3fn`` data type.
Before using the KV cache, the model dequantizes the KV cache to the data
type specified by ``torch_dtype`` in NeuronConfig.

To enable KV cache quantization, set ``kv_cache_quant=True`` in
NeuronConfig.

::

   neuron_config = NeuronConfig(kv_cache_quant=True)

- Important: To use KV cache quantization, you must set the
  ``XLA_HANDLE_SPECIAL_SCALAR`` environment variable to ``1``.

.. _nxd-speculative-decoding:

Speculative Decoding
--------------------

Speculative decoding is a performance optimization technique where a
smaller *draft* LLM model predicts the next tokens, and the larger *target*
LLM model verifies those predictions. NxD Inference supports
the following speculative decoding implementations:

1. :ref:`Speculative decoding with a draft model <nxd-vanilla-speculative-decoding>`,
   where a separate draft model predicts the next *n* tokens for the target
   model. Each model is compiled independently.
2. :ref:`Medusa speculative decoding<nxd-medusa-speculative-decoding>`,
   where several small model heads predict next tokens, and the target
   model verifies all predictions at the same time.
3. :ref:`EAGLE speculative decoding<nxd-eagle-speculative-decoding>`,
   where the draft model uses additional context from the target model
   to improve generation efficiency. NxD Inference supports EAGLE v1 with
   a flat draft structure.

.. _nxd-vanilla-speculative-decoding:

Speculative Decoding with a Draft model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use speculative decoding with a draft model, you configure, compile, and load a
draft model in addition to the main target model. To enable 
speculative decoding with a draft model, set ``speculation_length`` and
``trace_tokengen_model=False`` in the target model's NeuronConfig. The
draft model's NeuronConfig should use the same configuration but with
these additional attributes reset to their defaults.

 Speculative decoding with a draft model currently supports only batch sizes of 1.

.. _example-2:

Example
^^^^^^^

The following example demonstrates using Llama-3.2 3B as a draft model
for Llama-3.1 70B. The speculation length is set to 5 tokens.

::

   import copy

   from transformers import AutoTokenizer, GenerationConfig

   from neuronx_distributed_inference.models.config import NeuronConfig
   from neuronx_distributed_inference.models.llama.modeling_llama import (
       LlamaInferenceConfig,
       NeuronLlamaForCausalLM
   )
   from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
   from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

   prompts = ["I believe the meaning of life is"]

   model_path = "/home/ubuntu/models/Llama-3.1-70B"
   draft_model_path = "/home/ubuntu/models/Llama-3.2-3B"
   compiled_model_path = "/home/ubuntu/neuron_models/Llama-3.1-70B"
   compiled_draft_model_path = "/home/ubuntu/neuron_models/Llama-3.2-3B"

   # Initialize target model.
   neuron_config = NeuronConfig(
       speculation_length=5,
       trace_tokengen_model=False
   )
   config = LlamaInferenceConfig(
       neuron_config,
       load_config=load_pretrained_config(model_path)
   )
   model = NeuronLlamaForCausalLM(model_path, config)

   # Initialize draft model.
   draft_neuron_config = copy.deepcopy(neuron_config)
   draft_neuron_config.speculation_length **=** 0
   draft_neuron_config.trace_tokengen_model **=** True
   draft_config = LlamaInferenceConfig(
       draft_neuron_config,
       load_config=load_pretrained_config(draft_model_path)
   )
   draft_model = NeuronLlamaForCausalLM(draft_model_path, draft_config)

   # Compile and save models.
   model.compile(compiled_model_path)
   draft_model.compile(compiled_draft_model_path)

   # Load models to the Neuron device.
   model.load(compiled_model_path)
   draft_model.load(compiled_draft_model_path)

   # Load tokenizer and generation config.
   tokenizer **=** AutoTokenizer.from_pretrained(model_path, padding_side**=**neuron_config.padding_side)
   generation_config = GenerationConfig.from_pretrained(model_path)

   # Run generation.
   _, output_tokens = get_generate_outputs(
       model,
       prompts,
       tokenizer,
       is_hf=False,
       draft_model=draft_model,
       generation_config=generation_config
   )

   print("Generated outputs:")
   for i, output_token in enumerate(output_tokens):
       print(f"Output {i}: {output_token}")

.. _nxd-medusa-speculative-decoding:

Medusa Speculative Decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use Medusa speculative decoding, you must use a model that is
specifically fine-tuned for Medusa speculation, such as
`text-generation-inference/Mistral-7B-Instruct-v0.2-medusa <https://huggingface.co/text-generation-inference/Mistral-7B-Instruct-v0.2-medusa>`__.
You must also provide a Medusa tree. For an example Medusa tree, see
``medusa_mc_sim_7b_63.json`` in the ``examples`` folder in NeuronX
Distributed Inference.

To enable Medusa, set ``is_medusa=True``, set the
``medusa_speculation_length``, set the ``num_medusa_heads``, and specify
the ``medusa_tree``.

::

   def load_json_file(json_path):
       with open(json_path, "r") as f:
           return json.load(f)

   medusa_tree = load_json_file("medusa_mc_sim_7b_63.json")

   neuron_config = NeuronConfig(
       is_medusa=True,
       medusa_speculation_length=64,
       num_medusa_heads=4,
       medusa_tree=medusa_tree
   )

To run generation with a Medusa model and the HuggingFace ``generate()``
API, set the ``assistant_model`` to the target model.

For more information about Medusa speculative decoding, see the official
implementation on GitHub: https://github.com/FasterDecoding/Medusa.

Medusa speculative decoding currently supports only batch sizes of 1.

.. _nxd-eagle-speculative-decoding:

EAGLE Speculative Decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~

NxD Inference supports EAGLE v1 speculative decoding with a flat draft structure.

EAGLE Checkpoint Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use EAGLE speculative decoding, you must use a draft
model that is specifically fine-tuned for EAGLE speculation. Additionally, to use EAGLE with
NxD Inference, the draft model must include the LM head weights from the target model.
These weights are shared between the draft and target model.

Because NxD Inference uses a flat draft structure, it predicts only one token per draft iteration.
Although NxD Inference doesn't support EAGLE with a tree structure, you can train
an EAGLE checkpoint in the same way. Note that depending on your use case and dataset, you
might see lower acceptance rate with the flat draft structure compared with using a tree structure.

NxD Inference supports EAGLE models with or without input normalization. By default,
NxD Inference expects that the EAGLE model doesn't use input normalization. To use
an EAGLE model with input normalization, set ``enable_eagle_draft_input_norm`` to ``True``
in NeuronConfig.

You can find links to pretrained EAGLE draft model checkpoints for various
popular models in the official EAGLE repository on GitHub: https://github.com/SafeAILab/EAGLE.
However, these pretrained EAGLE model checkpoints don't include the LM head
weights from the target model. To use these pretrained checkpoints with NxD Inference,
you must first copy the LM head weights from the target to the draft model.

The following code demonstrates how to perform this operation for a `Llama-3.1-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct>`__
target model and the corresponding `EAGLE draft <https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-70B>`__:

::

    import json
    import os

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    target_model_path = "Meta-Llama-3.1-70B-Instruct"
    draft_model_path = "Llama-3.1-70B-Instruct-EAGLE-Draft"

    DRAFT_MODEL_SAFETENSORS_NAME = "model.safetensors"
    LM_HEAD_WEIGHT_TENSOR_NAME = "lm_head.weight"
    TARGET_MODEL_SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"

    def find_lm_head_safetensors_location(model_dir):
        model_index_location_path = os.path.join(model_dir, TARGET_MODEL_SAFETENSORS_INDEX_NAME)

        with open(model_index_location_path, 'r') as f:
            model_index_locations = json.load(f)

        lm_head_safetensors_name = model_index_locations["weight_map"][LM_HEAD_WEIGHT_TENSOR_NAME]

        return lm_head_safetensors_name

    # Find the target model `lm_head.weight` location in safetensors
    target_lm_head_safetensors_name = find_lm_head_safetensors_location(target_model_path)
    target_lm_head_safetensors_path = os.path.join(target_model_path, target_lm_head_safetensors_name)

    # Open the target model.safetensor containing `lm_head.weight`
    with safe_open(target_lm_head_safetensors_path, framework="pt") as f:
        target_lm_head = f.get_tensor(LM_HEAD_WEIGHT_TENSOR_NAME)

    # Collect all tensors in the draft model
    draft_model_safetensors_path = os.path.join(draft_model_path, DRAFT_MODEL_SAFETENSORS_NAME)
    tensors = {}
    with safe_open(draft_model_safetensors_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Add the LM head weights and save out the new draft model.safetensors file
    tensors[LM_HEAD_WEIGHT_TENSOR_NAME] = target_lm_head.type(torch.float16)
    save_file(tensors, draft_model_safetensors_path)

.. _nxd-fused-speculative-decoding:

Fused Speculation
^^^^^^^^^^^^^^^^^

EAGLE speculation uses a feature called *fused speculation*, where the
draft model and target model are fused into a single compiled model to
improve performance. Fused speculation uses a different config called
FusedSpecNeuronConfig, which specifies the model class. draft config,
and draft model path to fuse with the target model.

.. _example-3:

Example
^^^^^^^

::

    import copy

    from neuronx_distributed_inference.models.config import (
        FusedSpecNeuronConfig,
        NeuronConfig,
        OnDeviceSamplingConfig
    )
    from neuronx_distributed_inference.models.llama.modeling_llama import (
        NeuronLlamaForCausalLM,
        NeuronLlamaModel
    )
    from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
    from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
    from transformers import AutoTokenizer, GenerationConfig

    prompt = "The future of AI is"

    model_path = "/home/ubuntu/models/Llama-3.1-70B-Instruct"
    draft_model_path = "/home/ubuntu/models/Llama-3.1-70B-Instruct-EAGLE-Draft"
    compiled_model_path = "/home/ubuntu/neuron_models/Llama-3.1-70B-Instruct-EAGLE"
    max_sequence_length = 1024

    # Initialize on-device sampling configuration.
    on_device_sampling_config = OnDeviceSamplingConfig(
        temperature=0.7,
        top_k=50,
        top_p=1.0,
    )

    # Initialize model configuration.
    neuron_config = NeuronConfig(
        # Neuron supports EAGLE batch sizes greater than 1.
        # We set batch size to 1 in this tutorial due to a
        # limitation in the transformers library for
        # generation with speculative decoding.
        # For more information, see: https://github.com/huggingface/transformers/issues/32165
        batch_size = 1,
        enable_eagle_speculation=True,
        enable_fused_speculation=True,
        max_context_length=max_sequence_length,
        max_length=max_sequence_length,
        on_device_sampling_config=on_device_sampling_config,
        seq_len=max_sequence_length,
        speculation_length=5,
        # For best performance, set to the maximum tensor
        # parallelism of your Neuron instance type.
        tp_degree=32,
        trace_tokengen_model=False
    )

    config = NeuronLlamaForCausalLM.get_config_cls()(
        neuron_config, load_config=load_pretrained_config(model_path)
    )

    # Initialize draft model configuration and set EAGLE-specific values.
    draft_neuron_config = copy.deepcopy(neuron_config)
    draft_neuron_config.trace_tokengen_model = True
    draft_neuron_config.enable_fused_speculation = False
    draft_neuron_config.is_eagle_draft = True
    draft_neuron_config.sequence_parallel_enabled = False

    draft_config = NeuronLlamaForCausalLM.get_config_cls()(
        draft_neuron_config, load_config=load_pretrained_config(draft_model_path))

    # Initialize fused speculation configuration.
    fused_spec_config = FusedSpecNeuronConfig(
        NeuronLlamaForCausalLM._model_cls,
        draft_config=draft_config,
        draft_model_path=draft_model_path,
    )
    config.fused_spec_config = fused_spec_config

    # Initialize model from configuration.
    model = NeuronLlamaForCausalLM(model_path, config)

    # Compile and save model.
    model.compile(compiled_model_path)

    # Load model to the Neuron device.
    model.load(compiled_model_path)

    # Load tokenizer and generation config.
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.max_length = 1024
    # pad_token_id is required for Hugging Face assisted sampling.
    generation_config.pad_token_id = tokenizer.eos_token_id

    # Run generation and print outputs.
    _, output_tokens = get_generate_outputs(
        model,
        [prompt],
        tokenizer,
        is_hf=False,
        # draft_model is not set here due to fused speculation.
        draft_model=None,
        generation_config=generation_config
    )

    print("Generated output:")
    for _, output in enumerate(output_tokens):
        print(output)

MoE model architecture support
------------------------------

NxD Inference supports mixture-of-experts (MoE) models.
The library includes ready-to-use modeling code for Mixtral and DBRX.
These models are built using reusable MoE modules from NeuronX
Distributed Core: ``RouterTopK``, ``ExpertMLPs``, and ``MoE``. You can
use these modules to onboard additional MoE models.

NxD Inference also provides a helper function,
``initialize_moe_module``, which you can use to initialize an MoE
model's MLP module from these MoE modules. For examples of how to use
this helper function, see the decoder layer module implementation in the
`Mixtral <https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/mixtral/modeling_mixtral.py>`__
and `DBRX <https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/dbrx/modeling_dbrx.py>`__
modeling code.

Grouped-query attention (GQA) support
-------------------------------------

NxD Inference provides a reusable attention module,
NeuronAttentionBase, which you can use when onboarding models. This
module is also used in NxD Inference modeling code like Llama and
Mixtral.

NxD Inference supports the following sharding strategies
for the KV cache used in the attention module:

- ``CONVERT_TO_MHA`` — Transforms a GQA attention mechanism into a
  traditional MHA mechanism by replicating the K/V heads to evenly match
  the corresponding Q heads. This consumes more memory than would
  otherwise be used with other sharding mechanisms but works in all
  cases.
- ``REPLICATE_TO_TP_DEGREE`` — Transforms a GQA attention mechanism such
  that there is exactlyone K/V head per tp_degree through replication
  e.g. 8 K/V heads with tp_degree=32 results in 32 K/V heads. This is
  more memory efficient but does not work for all configurations. Q
  heads are padded interleaved to retain correct alignment between Q and
  K/V heads.

The NeuronAttentionBase module uses ``REPLICATE_TO_TP_DEGREE`` by
default. If the TP degree isn't divisible by the number of KV heads,
NeuronAttentionBase uses ``CONVERT_TO_MHA``.

.. _nxdi_async_mode_feature_guide:

Asyncronous Runtime Support
---------------------------

NxD Inference offers certain model configurations to be run with Asyncronous Runtime Mode (Async mode).
Async mode allows NxD Inference to parallelize CPU logic with Neuron (NEFF) logic. As a result, any CPU overheads
within NxDI that exist between sequential model executions (ex. autoregressive loop in LLMs) are almost fully
eliminated. This reduces latency anywhere from 5% to 20% based on the model configuration.

This feature can be enabled with by setting ``async_mode`` to ``True`` in ``NeuronConfig``.

To use Async mode, a model configuration must meet the following prerequisites:
- On-device sampling is enabled.
- If speculation is enabled, fused speculation must also be enabled.

It is highly recommended to set ``async_mode`` to ``True`` for every other case, since it offers a latency reduction.
Furthermore, this feature is a purely runtime feature, so if you have a previously compiled model, and its configuration
doesn't fall under the unsupported case, ``async_mode`` will likely be able to improve performance.

.. note::
    If you are using vLLM, this feature works independently of vLLM's Async Engine. As a result, ``async_mode`` can be enabled
    whether vLLM is used or not.

.. _nxdi_prefix_caching:

Prefix Caching Support
----------------------

Prefix caching is a performance optimization technique where prompts in multiple requests sharing the same prefix can reuse the
previously computed KV cache. When context encoding a prompt that starts with a previously computed prefix, the encoding of the
prefix tokens will be skipped and the corresponding KV Cache will be fetched and used for encoding the rest of the tokens (suffix).
The performance benefit comes from the time saved by re-using the KV Cache instead of re-encoding the prefix tokens. NxD Inference
supports prefix caching during context encoding. To store KV cache and match to prefix efficiently, NxD Inference uses block KV Cache
layout for prefix caching. NxD Inference does not implement its own cache eviction, memory management, or prefix hashing for matches.
Instead, it requires external management of the block KV cache and expects active block tables and slot mappings to be provided with
each generation request. This feature integrates with vLLM by enabling automatic prefix caching, which manages the block tables,
handles automatic prefix matching across prompts, and performs cache evictions. More on automatic prefix caching support on vLLM
can be found `here <https://docs.vllm.ai/en/latest/design/v1/prefix_caching.html>`__.

To enable prefix caching with NxD Inference, set ``is_prefix_caching=True`` in NeuronConfig along with configurations for
block KV cache layout.

::

    neuron_config = NeuronConfig(
        is_prefix_caching=True,
        is_block_kv_layout=True,
        pa_num_blocks=1024,
        pa_block_size=32,
    )

``is_block_kv_layout=True`` and ``is_prefix_caching=True`` are set in NeuronConfig to enable block KV Cache layout and enable
prefix caching. The first two dimensions of the KV cache are set to the number of blocks and block size, respectively. These
configurations are specified using ``pa_num_blocks`` and ``pa_block_size`` in NeuronConfig. For optimal performance with Neuron,
it's recommended to set ``pa_block_size=32``. The minimum required ``pa_num_blocks`` to be initialized is
``(batch_size * max_model_len) / pa_block_size`` However, it is recommended to initialize more blocks than the required minimum
to accommodate caching of common prefixes. The higher the number of blocks, the greater the likelihood of cache hits, as fewer
cache evictions will occur. NxD Inference does not currently provide an automated solution to determine the maximum number of
KV Cache blocks that can be initialized in HBM without exceeding available memory space. Customers are advised to experiment with
increasing the number of blocks that balances the cache hit rate and memory taken. Any memory taken by increasing the cache will
impact the batch sizes and sequence lengths that can be supported, so customers are sugggested to pick the correct number of blocks
considering these trade offs and the specific inference workload they plan to run in production.

NxD Inference does not use paged attention for prefix caching. Instead, it follows a different process:
first gathering the block KV cache using the block table, then converting it to a flat KV cache layout, computing attention, and 
finally scattering the computed cache back to the block KV cache layout. This approach introduces overhead during
token generation requests due to layout conversions, which can negatively impact performance as the ``max_model_len`` increases.

.. _bucketing-with-prefix-caching:

Bucketing with Prefix Caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prefix caching handles both the prefix (cache hit) and suffix (no cache hit) portions of input prompts during context encoding.
A two-dimensional bucketing system has been introduced to support context encoding when prefix caching is enabled. This system
uses separate dimensions corresponding to the prefix and suffix (non cache-hit portion) of the input prompts. In contrast,
token generation still uses one-dimensional bucketing based on the maximum sequence length.

When bucketing is enabled, NxD Inference creates prefill (suffix) buckets (covering suffix portion) starting with powers of 2,
ranging from 512 up to the maximum context length. The prefix buckets mirror the prefill buckets, with one key difference: a special
prefix bucket of size 0 is added to handle requests with no cache hits. NxD Inference then creates a two-dimensional grid of all prefill
and prefix bucket combinations, which represents the effective set of buckets during context encoding. During request processing,
NxD Inference first identifies the smallest prefill bucket that can accommodate the largest suffix portion of the input prompts.
If prefill padding is needed, NxD Inference prioritizes moving tokens from the prefix's end to the prefill bucket before adding padding.
It then determines the smallest prefix bucket that can fit the largest prefix across prompts. These two dimensions together determine
the final (prefill, prefix) bucket combination used to serve the context encoding request.

You can configure specific buckets to optimize inference based on the expected distribution of prefix lengths, input lengths, and
output lengths for your model. In NeuronConfig, set ``enable_bucketing=True``, and provide a list of bucket sizes in
``context_encoding_buckets``, ``prefix_buckets`` and/or ``token_generation_buckets``. ``context_encoding_buckets`` corresponds to prefill
buckets when prefix caching is enabled.

::

    neuron_config = NeuronConfig(
        enable_bucketing=True,
        context_encoding_buckets=[512, 1024, 2048],
        prefix_buckets=[512, 1024]
        token_generation_buckets=[2048]
    )

Examples
^^^^^^^^

For ``context_encoding_buckets=[512, 1024, 2048]`` and ``prefix_buckets=[512, 1024]``

For requests with:

- Input prompt of size 1000 with no prefix, NxDI uses prefill bucket as 1024 and prefix bucket as 0.
- Input prompt of size 800 with 128 as the prefix size, and remaining 672 as the suffix size, NxDI first selects 1024
  as the prefill bucket. Remaining 352 prefill slots are filled up by moving entire prefix to the suffix part.
  So prefill bucket of 1024 and prefix bucket as 0 is used here.
- Input prompt of size 900 with 640 as the prefix size, and remaining 260 as the suffix size, NxDI first selects 512
  as the prefill bucket. Remaining 252 prefill slots are filled up by moving 252 tokens from the end of prefix to the suffix part.
  Effective prefix length now becomes 388, so prefill bucket of 512 and prefix bucket as 512 is used.
- Input prompt of size 1600 with 1280 as the prefix size and remaining 320 as the suffix size, NxDI selects 512 as the
  prefill bucket. Remaining 192 prefill slots are filled up by moving 192 tokens from the end of prefix to the suffix part.
  Effective prefix length now becomes 1088 which is larger than the largest prefix bucket of 1024. This leads to exception
  during the request processing.

The two-dimensional bucketing system exponentially increases the number of context encoding buckets. Therefore, users should exercise caution
when using auto-bucketing with large context lengths. It is recommended to limit the granularity of prefix buckets based on your
specific workload requirements.

For detailed examples of prefix caching with NxD Inference and vLLM, see :ref:`/libraries/nxd-inference/tutorials/trn2-llama3.3-70b-apc-tutorial.ipynb`.

Multi-LoRA Serving
------------------

NxD Inference supports serving with multiple LoRA adapters and users can specify different LoRA adapters for their requests at runtime. 
It also supports multi-LoRA serving with vLLM as the frontend.
NxD Inference currently supports loading of LoRA adapters for dense model families, including Llama-2, Llama-3.1, Llama-3.2, Llama-3.3, TinyLlama, OpenLLaMA, Qwen2, and Qwen3.
A current prerequisite is that the LoRA adapter checkpoints must be stored locally before the server is initialized and started.

Enable multi-LoRA serving
~~~~~~~~~~~~~~~~~~~~~~~~~

To enable multi-LoRA serving, provide a LoraServingConfig for ``lora_config`` attribute in NeuronConfig.

::

    lora_config = LoraServingConfig(
        max_loras=max_loras,
        max_cpu_loras=max_cpu_loras,
        batch_size=batch_size,
        dynamic_multi_lora=dynamic_multi_lora,
        base_model_quantized=quantized,
        lora_ckpt_json=lora_ckpt_json,
    )
    neuron_config = NeuronConfig(lora_config=lora_config)

Refer to :ref:`nxd-inference-api-guide` for more details of ``LoraServingConfig``.

NxD Inference primarily supports the format of LoRA adapters from `Huggingface PEFT <https://github.com/huggingface/peft>`__.
Each checkpoint path is a folder that contains a checkpoint file (.safetensors, .bin, or .pt) and a configuration json file (.json).
In addition, NxD inference also supports LoRA adapters trained from `NxD LoRA finetuning <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/lora_finetune_developer_guide.html>`__.
Each checkpoint path is a checkpoint file (.pt) that includes both LoRA adapter weights and the configuration. 

NxD Inference assumes all the LoRA adapters for multi-LoRA serving are available locally during compilation and their weights are loaded on neuron devices during serving.
When uploading a LoRA adapter checkpoint to NxDI for multi-LoRA serving, the user is required to name the adapter with a unique adapter ID, such as ``adapter_id_1``, which will be used by users to specify the LoRA adapter for serving at runtime and by NxDI for model compilation.

The maximum number of concurrent LoRA adapters in device memory and host memory for serving are specified by ``max_loras`` and ``max_cpu_loras``, respectively.
When ``dynamic_multi_lora=False``, all the LoRA adapters must be fully pre-loaded into device memory before the serving process begins.
Dynamic multi-LoRA serving is enabled by ``dynamic_multi_lora=True``, which loads more LoRA adapters to host memory and dynamically swaps them from CPU to HBM at runtime according to user requests.
NxD Inference can quantize the base model for multi-LoRA serving with ``base_model_quantized=True``. 
Refer to :ref:`nxd-inference-api-guide-neuron-config` for setting the quantization configurations.
The set of LoRA adapters are specified by ``lora_ckpt_json``, which is a JSON file describing the mapping between the adapters IDs and their local paths of the LoRA adapter checkpoint.
Refer to :ref:`nxd-inference-api-guide-neuron-config` for the JSON format.
For detailed examples of multi-LoRA serving in NxDI, see :ref:`/libraries/nxd-inference/tutorials/trn2-llama3.1-8b-multi-lora-tutorial.ipynb`.


Maximum number of LoRA adapters supported in device memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LoRA adapter size is much smaller than the base model, but its weights still consumes non-negligible on-device memory. 
The maximum number of LoRA adapters that can be concurrently supported in the device memory depends on the base model, the LoRA rank, the reserved HBM size for LoRA adapters, and how the LoRA adapters are sharded across TP groups.

Suppose a Trainium instance is used for multi-LoRA serving and the reserved HBM size on each neuron core for LoRA adapters is 2GB.
Each LoRA adapter has two parts, LoRA A and LoRA B, and only one of them can be partitioned with tensor parallelism and the other is just Linear layer.
We analyze the maximum number of LoRA adapters supported in the device memory under two cases: 1/ the linear layer is duplicated, and 2/ the linear layer is sharded.
These two cases can be specified by ``lora_shard_linear_layer`` in ``LoraServingConfig``.

When the linear layer is duplicated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The weight size of a LoRA adapter on each device is around half of the total LoRA adapter size in this case.
When the base model is Llama3.1 8B, the LoRA adapter checkpoint size with LoRA rank 16 in BF16 is around 170MB. 
Because ``2GB / (170MB / 2) = 23``, the maximum number of concurrent LoRA adapters is 23.
When the base model is Llama3.3 70B, the LoRA adapter checkpoint size with LoRA rank 16 in BF16 is around 830MB and we can set ``max_loras=4``.
We analyze the maximum number of LoRA adapters supported in NxD inference under two cases: the linear layer is duplicated and the linear layer is sharded.
These two cases can be specified by ``lora_shard_linear_layer`` in ``LoraServingConfig``.

.. list-table::
    :widths: auto
    :header-rows: 1 
    :stub-columns: 1    
    :align: left
      
    *   - Model
        - Reserved Memory size
        - LoRA rank
        - Maximum LoRAs
    
    *   - Llama3.1 8B
        - 2GB
        - 16
        - 23
    *   - Llama3.1 8B
        - 2GB
        - 32
        - 12
    *   - Llama3.3 70B
        - 2GB
        - 16
        - 4
    *   - Llama3.3 70B
        - 2GB
        - 32 
        - 2

When the linear layer is sharded
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The linear layer in a LoRA adapter is sharded across neuron cores in a TP group at the cost of Allgather communication overehead in this case.
The weight size of a LoRA adapter on each device is ``1/TP_DEGREE`` of the total LoRA adapter size.

.. list-table::
    :widths: auto
    :header-rows: 1 
    :stub-columns: 1    
    :align: left
      
    *   - Model
        - Reserved Memory size
        - LoRA rank
        - TP degree
        - Maximum LoRAs
    
    *   - Llama3.1 8B
        - 2GB
        - 16
        - 32
        - 376
    *   - Llama3.1 8B
        - 2GB
        - 32
        - 32
        - 188
    *   - Llama3.3 70B
        - 2GB
        - 16
        - 32
        - 77
    *   - Llama3.3 70B
        - 2GB
        - 32 
        - 32
        - 38

.. _nxdi_di_feature_guide:

Disaggregated Inference [BETA]
------------------------------

Disaggregated Inference is an LLM serving architecture separates the prefill and decode phases of inference onto different hardware resources.
Separating the compute intensive prefill phase from the memory bandwidth intensive decode phase can improve the LLM serving experience by

1. Removing prefill interruptions to decode from continuous batching to reduce inter token latency (ITL). These gains can be used to
achieve higher throughput by running with a higher decode batch size while staying under Service Level Objectives (SLO).

2. Adapt to changing traffic patterns while still remaining under application SLOs.

3. Enable independent scaling of resources and parallelism strategies for prefill (compute bound) and decode (memory bound).

See the :ref:`Disaggregated Inference Developer Guide<nxdi-disaggregated-inference>` and the :ref:`Disaggregated Inference Tutorial<nxdi-disaggregated-inference-tutorial>`
