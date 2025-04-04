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

Serialization support
---------------------

When you compile a model with NxD Inference, the library
serializes the model to a given folder. After you have a serialized
model, you can load it directly to a Neuron device without needing to
compile again.

The compile function serializes sharded weights by default, and you can
disable this functionality with the ``save_sharded_checkpoint`` flag in
NeuronConfig.

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

.. _nxdi-tensor-parallelism:

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

Multi-LoRA Serving
------------------

NxD Inference supports serving with multiple LoRA adapters and users can specify different LoRA adapters for their requests at runtime. 
It also supports multi-LoRA serving with vLLM as the frontend.
NxD Inference currently supports loading of LoRA adapters at server startup for dense models like Llama-3.3-70B. 
Dynamic loading of LoRA adapters at runtime is not currently supported and will be supported in a future Neuron release.

Enable multi-LoRA serving
~~~~~~~~~~~~~~~~~~~~~~~~~

To enable multi-LoRA serving, provide a LoraServingConfig for ``lora_config`` attribute in NeuronConfig.

::

    lora_config = LoraServingConfig(
        max_loras=max_loras,
        lora_ckpt_paths=lora_ckpt_paths,
    )
    neuron_config = NeuronConfig(lora_config=lora_config)

Refer to :ref:`nxd-inference-api-guide` for more details of ``LoraServingConfig``.

NxD Inference primarily supports the format of LoRA adapters from `Huggingface PEFT <https://github.com/huggingface/peft>`__.
Each checkpoint path is a folder that contains a checkpoint file (.safetensors, .bin, or .pt) and a configuration json file (.json).
In addition, NxD inference also supports LoRA adapters trained from `NxD LoRA finetuning <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/lora_finetune_developer_guide.html>`__.
Each checkpoint path is a checkpoint file (.pt) that includes both LoRA adapter weights and the configuration. 

NxD Inference assumes all the LoRA adapters for multi-LoRA serving are available locally during compilation and their weights are loaded on neuron devices during serving.
When uploading a LoRA adapter checkpoint to NxDI for multi-LoRA serving, the user is requried to name the adapter with a unique adapter ID, which will be used by users to specify the LoRA adapter for serving at runtime and by NxDI for model compilation.

The number of the multiple LoRA adapters for serving is specified by ``max_loras``.
The set of LoRA adapters in NxD Inference are specified by ``lora_ckpt_paths``, which is a dictionary with a key-value pair for each LoRA adapter. 
The key is the adapter ID named by the user and the value is the local path of the LoRA adapter checkpoint.
For detailed examples of multi-LoRA serving in NxDI, see :ref:`nxdi-trn2-llama3.1-8b-multi-lora-tutorial`.


Maximum number of LoRA adapters supported
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LoRA adapter size is much smaller than the base model, but its weights still consumes non-negligible on-device memory. 
The maximum number of LoRA adapters that can be concurrently supported in NxD Inference depends on the base model, the LoRA rank, the reserved memory size for LoRA adapters, and how the LoRA adapters are sharded across TP groups.

Suppose Trn1 instance is used for multi-LoRA serving and the reserved memory size on each neuron core for LoRA adapters is 2GB.
Each LoRA adapter has two parts, LoRA A and LoRA B, and only one of them can be partitioned with tensor parallelism and the other is just Linear layer.
We analyze the maximum number of LoRA adapters supported in NxD inference under two cases: the linear layer is duplicated and the linear layer is sharded.
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