.. _transformers_neuronx_developer_guide:

Transformers NeuronX (``transformers-neuronx``) Developer Guide
================================================================

Transformers NeuronX for Trn1 and Inf2 is a software package that enables
PyTorch users to perform large language model (LLM) :ref:`performant inference <neuron_llm_inference>` on
second-generation Neuron hardware (See: :ref:`NeuronCore-v2 <neuroncores-v2-arch>`).The :ref:`Neuron performance page <inf2-performance>` lists expected inference performance for commonly used Large Language Models.


Introduction
------------

The `Transformers NeuronX repository <https://github.com/aws-neuron/transformers-neuronx>`_
contains the source code of the AWS Neuron Transformers integration project. 
As it stands now, it mainly serves the purpose of
running transformer decoder inference (autoregressive sampling)
workflows on the Neuron platform.

Note: This project is **actively** in development. The Neuron team is
still heavily modifying the Neuron optimized module classes. The
functionality provided in this repository will not maintain long-term
API stability until version >= 1.0.0. For applications willing to reuse
code from this repository, we recommend treating the Neuron optimized
module implementations as samples, and pin the version of the main
library package ``torch-neuronx`` to avoid breaking interface changes as
new features are developed.



Checkpoint compatibility with HuggingFace Transformers
------------------------------------------------------

``transformers-neuronx`` is checkpoint-compatible with HuggingFace
Transformers. While the Neuron team reimplemented some HuggingFace
Transformers models from scratch for the purpose of maximizing the
execution efficiency of transformer decoders on Neuron, the
implementations are done with maximizing compatibility in mind, meaning
one can train transformer decoder models, say GPT2, using the standard
HuggingFace Transformers library, and then construct an
inference-optimized decoder model using transformers-neuronx's
``GPT2ForSampling`` class. If training was done with other libraries
such as MegatronLM, then it is still possible to convert the obtained
checkpoint to the standard HuggingFace Transformers checkpoint format,
and then move on to transformers-neuronx's optimized decoder
implementations.


Neuron optimized transformer decoders implemented in XLA High Level Operations (HLO)
------------------------------------------------------------------------------------

Due to the stateful nature of the autoregressive sampling computation,
an efficient implementation of autoregressive sampling using the Neuron
SDK requires rewriting the model forward function into a pure-function
computation running on fixed-shape tensors. Furthermore, we want the
pure-function computation be implemented in a compiled language so that
the Neuron compiler can perform extensive code analysis and
optimization. We chose XLA High Level Operations (HLO) as the compiled
language for implementing Neuron optimized transformer decoder classes.
The source code of these classes contains Python functions written in a
syntax called "PyHLO", name of a Neuron internal tool for
writing/compiling the HLO language in Python. As an example, a "language
model head" implemented in PyHLO may look like the following.

::

   class LmHeadHlo:

       ...

       def lm_head(self, scribe):
           dtype = self.dtype
           hidden_size = self.hidden_size
           n_active_tokens = self.n_active_tokens
           batch_size = self.batch_size
           vocab_size = self.vocab_size
           hidden = dtype[hidden_size, n_active_tokens, batch_size].Parameter(parameter_number=0)
           weight = dtype[hidden_size, vocab_size].Parameter(parameter_number=1)
           rhs_size = n_active_tokens * batch_size
           hidden = dtype[hidden_size, rhs_size].Reshape(hidden)
           dot_dims = dict(lhs_contracting_dimensions=[0], rhs_contracting_dimensions=[0])
           logits = dtype[vocab_size, rhs_size].Dot(weight, hidden, dot_dimension_numbers=dot_dims)
           return dtype[vocab_size, n_active_tokens, batch_size].Reshape(logits)

       ...

The ``transformers_neuronx.compiler.compile_py_func`` function can
convert the Python ``lm_head`` function into ``HloModuleProto``, a valid
input format for the ``neuronx-cc`` compiler.


Tensor-parallelism support
--------------------------

For transformer decoders used in large language models,
tensor-parallelism is necessary as it provides a way to shard the
models' large weight matrices onto multiple NeuronCores, and having
NeuronCores working on the same matrix multiply operation
collaboratively. transformers-neuronx's tensor-parallelism support makes
heavy use of collective operations such as all-reduce, which is
supported natively by the Neuron runtime.

There are some principles for setting tensor-parallelism degree (number
of NeuronCores participating in sharded matrix multiply operations) for
Neuron-optimized transformer decoder models.

1. The number of attention heads needs to be divisible by the
   tensor-parallelism degree.
2. The total data size of model weights and key-value caches needs to be
   smaller than 16 GB times the tensor-parallelism degree.
3. Currently, the Neuron runtime supports tensor-parallelism degrees 1,
   2, 8, and 32 on Trn1 and supports tensor-parallelism degrees 1, 2, 4,
   8, and 24 on Inf2.

Some examples:

1. ``facebook/opt-13b`` has 40 attention heads, and when running at
   batch size 1 and float16 precision the model requires ~29 GB memory,
   therefore a ``trn1.2xlarge`` with 32 GB device memory is sufficient.
2. ``facebook/opt-30b`` has 56 attention heads, and at batch size 1 and
   float16 precision the model requires ~66 GB memory, therefore it can
   run on 8 NeuronCores on one ``trn1.32xlarge`` using 128 GB device
   memory.
3. ``gpt2-xl`` has 25 attention heads and requires ~4 GB memory at
   bfloat16 precision. It runs without tensor-parallelism only.


Features
--------


Compile-time Configurations
---------------------------

Transformers Neuron models support a variety of compile-time configurations
that can be used to tune model performance. All models support the following
configurations:

- ``batch_size``: The batch size to compile a model for. Once the batch size has
  been set, this is the only size that is supported at inference time. Neuron
  uses ahead-of-time compilation to achieve high performance which requires
  that the compiled artifact shapes must be known at compilation time.
- ``n_positions``: The maximum number of positions (or sequence length) to allow
  during generation. This parameter directly controls the width of the KV
  cache. This parameter should be set to the maximum expected sequence length
  for the end application.
- ``tp_degree``: This parameter controls the number of tensor parallel shards to
  split the model into. Each shard will execute on a separate NeuronCore. To
  minimize latency, it is recommended to set the tensor parallelism to be
  equal to the number of NeuronCores that are available on an instance.
- ``amp``: This allows a models weights and compute to be cast to a different
  type. The options are; ``'bf16'``, ``'f16'``, or ``'f32'``. For
  models trained in ``float32``, the 16-bit mixed precision options
  (``'bf16'``, ``'f16'``) generally provide sufficient accuracy while
  significantly improving performance.
- ``context_length_estimate``: This parameter controls the maximum sequence
  length of the prompt/context handling compute graph. This parameter is
  not supported in ``GPTNeoXForSampling`` and ``GPTJForSampling``.

.. code-block:: python

    from transformers_neuronx import NeuronAutoModelForCausalLM

    model = NeuronAutoModelForCausalLM.from_pretrained(
        'gpt2',                      # Uses the GPT2 checkpoint from https://huggingface.co/gpt2
        batch_size=1,                # Allow inference with batch size 1 inputs
        n_positions=128,             # Allow a maximum size of 128 prompt & output tokens
        tp_degree=2,                 # Shard the model weights & compute across 2 NeuronCores
        amp='f16',                   # Downcast the weights & compute to float16
        context_length_estimate=64,  # Build an optimized context encoding network for a maximum prompt size of 64
    )
    model.to_neuron() # Load/compile the model



Checkpoint support and automatic model selection
------------------------------------------------

*New in release 2.18*

Transformers Neuron now supports a greater variety of checkpoints including
older pytorch binary checkpoints and newer `safetensors`_ checkpoints. For
improved load speed and reduced host memory consumption, it is recommended to
always use ``safetensors`` by default. Both regular and sharded variants of
checkpoints are supported. It is no longer recommended to use the
``save_pretrained_split`` function which was used in older Transformers Neuron
examples.

In addition to supporting standard checkpoint formats, Transformers Neuron
provides an AutoModel class ``NeuronAutoModelForCausalLM`` which can be
used to load the correct model without explicitly importing the
architecture-specific class.

.. _safetensors: https://github.com/huggingface/safetensors

.. code-block:: python

    from transformers_neuronx import NeuronAutoModelForCausalLM

    # Loads: https://huggingface.co/bigscience/bloom-560m
    bloom = NeuronAutoModelForCausalLM.from_pretrained('bigscience/bloom-560m')
    bloom.to_neuron()

    # Loads: https://huggingface.co/openlm-research/open_llama_3b_v2
    llama = NeuronAutoModelForCausalLM.from_pretrained('openlm-research/open_llama_3b_v2')
    llama.to_neuron()

    # This is equivalent to the following:
    from transformers_neuronx import BloomForSampling
    model = BloomForSampling.from_pretrained('bigscience/bloom-560m')
    model.to_neuron()

    from transformers_neuronx import LlamaForSampling
    llama = LlamaForSampling.from_pretrained('openlm-research/open_llama_3b_v2')
    llama.to_neuron()


.. note::

    Advanced features of huggingface hub access are not supported. This
    includes private repositories which require access tokens and branches.

    In order to support more advanced repository downloads, please download the
    model to a local directory and load it from there.



Hugging Face generate() API support
-----------------------------------

Transformers Neuron models support the Hugging Face `generate() <https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationMixin.generate>`__
API via the ``HuggingFaceGenerationModelAdapter`` adapter class. In the following example we
demonstrate how to run sampling with temperature using the ``GPT2`` model:

.. code-block:: python

    import torch
    from transformers import AutoTokenizer, AutoConfig
    from transformers_neuronx import GPT2ForSamplingWithContextBroadcasting, HuggingFaceGenerationModelAdapter

    # Create and compile the Neuron model
    model = GPT2ForSamplingWithContextBroadcasting.from_pretrained('gpt2')
    model.to_neuron()

    # Use the `HuggingFaceGenerationModelAdapter` to access the generate API
    config = AutoConfig.from_pretrained('gpt2')
    model = HuggingFaceGenerationModelAdapter(config, model)

    # Get a tokenizer and example input
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    text = "Hello, I'm a language model,"
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)

    # Run inference using temperature
    with torch.inference_mode():
        model.reset_generation()
        generated_sequence = model.generate(
            input_ids=encoded_input.input_ids,
            attention_mask=encoded_input.attention_mask,
            do_sample=True,
            max_length=256,
            temperature=0.7,
        )

    print([tokenizer.decode(tok) for tok in generated_sequence])


Note: As the Hugging Face generation API can expand the input's batch dimension
based on different generation configurations, we need to compile the neuron
model with different compile batch_size compared to the run time batch_size
(batch dimension of inputs to generation API).
- if ``do_sample=True``, ``compile_batch_size = runtime_batch_size x num_return_sequences x beam_size``
- otherwise, ``compile_batch_size = runtime_batch_size x num_return_sequences``



Neuron Persistent Cache
------------------------

The Neuron Persistent Cache is now enabled for Transformers Neuron by default.
Model artifacts which have been compiled once will be cached and reused on
successive runs when possible. Model artifacts will only be reused when
compiling with the same compiler version (neuronx-cc), model configurations,
and compiler flags. It also includes other features (i.e. using an S3 bucket as
the cache backend). For more detailed information, see the
:ref:`Persistent cache documentation <neuron-caching>`


.. _int8_weight_storage_support:


int8 weight storage support
------------------------

Transformers Neuron supports int8 weight storage for the ``GPT2`` model class.
int8 weight storage can be used to reduce memory bandwidth usage to improve
model performance. int8 weight storage support for additional model classes
will be added in an upcoming release. In the following example we demonstrate
how to apply int8 weight storage to the ``GPT2`` model via the
``QuantizationConfig`` and ``NeuronConfig`` configs:

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from transformers_neuronx import GPT2ForSamplingWithContextBroadcasting, NeuronConfig, QuantizationConfig

    # Set the weight storage config use int8 quantization and bf16 dequantization
    neuron_config = NeuronConfig(
        quant=QuantizationConfig(quant_dtype='s8', dequant_dtype='bf16'),
    )

    # Create and compile the Neuron model
    model = GPT2ForSamplingWithContextBroadcasting.from_pretrained(
        'gpt2',
        amp='bf16', # NOTE: When using quantization, amp type must match dequant type
        neuron_config=neuron_config
    )
    model.to_neuron()

    # Get a tokenizer and example input
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    text = "Hello, I'm a language model,"
    encoded_input = tokenizer(text, return_tensors='pt')

    # Run inference
    with torch.inference_mode():
        generated_sequence = model.sample(encoded_input.input_ids, sequence_length=256, start_ids=None)
    print([tokenizer.decode(tok) for tok in generated_sequence])



Parallel Input Prompt Context Encoding
--------------------------------------

Transformers Neuron supports parallel input prompt context encoding for the ``GPT2``
model class. Parallel context encoding can be used to significantly reduce
the latency of the input prompt context encoding before the autoregressive
decoder token generation loop. Parallel context encoding support for additional
model classes will be added in an upcoming release.

The ``GPT2ForSamplingWithContextBroadcasting`` class has a ``context_length_estimate``
variable that determines the number of input prompt tokens that will be processed in
parallel. For optimal results, this should be set to a power of 2 that is
closest to the most frequently seen input prompt length.
In the following example we demonstrate how to apply parallel context encoding
to the ``GPT2`` model via the ``GPT2ForSamplingWithContextBroadcasting`` class.
In this example, we set the ``context_length_estimate`` to be 128, which is
the closest power of 2 the length of the input prompt (97 tokens).

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from transformers_neuronx import GPT2ForSamplingWithContextBroadcasting

    # Create and compile the Neuron model
    model = GPT2ForSamplingWithContextBroadcasting.from_pretrained(
        'gpt2',
        context_length_estimate=256 # Create an optimized network which handles prompts up to 256 tokens
    )
    model.to_neuron()

    # Get a tokenizer and example input
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    text = "Hello, I'm a generative AI language model. Generative AI is a type of AI that can create new content and ideas, including conversations, stories, images, videos, and music. It is powered by large models that are pre-trained on vast amounts of data and commonly referred to as foundation models (FMs). With generative AI on AWS, you can reinvent your applications, create entirely new customer experiences, drive unprecedented levels of productivity, and transform your business. "
    encoded_input = tokenizer(text, return_tensors='pt')

    # Run inference
    with torch.inference_mode():
        generated_sequence = model.sample(encoded_input.input_ids, sequence_length=256)
    print([tokenizer.decode(tok) for tok in generated_sequence])



The ``GPT2ForSamplingWithContextBroadcasting`` class can also process
an input prompt that has a different batch size from the batch size of the
autoregressive decoder output. For example, an input prompt with batch size = 1 can
be used to produce an output of batch size = 5 to generate multiple suggestions
for the same input prompt. The input prompt batch size can be specified using
the ``prompt_batch_size`` argument and the autoregressive decoder output batch
size can be specified using the ``batch_size`` argument. In the following example
we demonstrate how to apply parallel context encoding to the ``GPT2`` model
to generate 5 outputs for a single input.

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from transformers_neuronx import GPT2ForSamplingWithContextBroadcasting

    # Create and compile the Neuron model
    model = GPT2ForSamplingWithContextBroadcasting.from_pretrained(
        'gpt2',
        prompt_batch_size=1, # This allows prompt and output batch to vary
        batch_size=5,
        context_length_estimate=256
    )
    model.to_neuron()

    # Get a tokenizer and example input
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    text = "Hello, I'm a generative AI language model. Generative AI is a type of AI that can create new content and ideas, including conversations, stories, images, videos, and music. It is powered by large models that are pre-trained on vast amounts of data and commonly referred to as foundation models (FMs). With generative AI on AWS, you can reinvent your applications, create entirely new customer experiences, drive unprecedented levels of productivity, and transform your business. "
    encoded_input = tokenizer(text, return_tensors='pt')

    # Run inference
    with torch.inference_mode():
        generated_sequence = model.sample(encoded_input.input_ids, sequence_length=256)

    for i, output in enumerate(generated_sequence):
        print('-' * 50)
        print(f'Batch {i} output:')
        print(tokenizer.decode(output))


Serialization support [Beta]
----------------------------

Transformers Neuron supports model serialization (model saving and loading) for
all models except the ``GPTJForSampling`` and ``GPTNeoXForSampling``` model
classes. In the following example we demonstrate how to save and load the
``GPT2`` model:

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from transformers_neuronx import GPT2ForSamplingWithContextBroadcasting, HuggingFaceGenerationModelAdapter

    # Create and compile the Neuron model
    model = GPT2ForSamplingWithContextBroadcasting.from_pretrained('gpt2')
    model.to_neuron()

    # Save the compiled Neuron model
    model.save('gpt2-compiled-artifacts')

    # Load the Neuron model
    model = GPT2ForSamplingWithContextBroadcasting.from_pretrained('gpt2')
    # Load the compiled Neuron artifacts
    model.load('gpt2-compiled-artifacts')
    # Since prior artifacts are loaded, this skips compilation
    model.to_neuron()

    # Get a tokenizer and example input
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    text = "Hello, I'm a language model,"
    encoded_input = tokenizer(text, return_tensors='pt')

    # Run inference
    with torch.inference_mode():
        generated_sequence = model.sample(encoded_input.input_ids, sequence_length=256, start_ids=None)
    print([tokenizer.decode(tok) for tok in generated_sequence])


Grouped-query attention (GQA) support [Beta]
----------------------------

Transformers Neuron supports grouped-query attention (GQA) models for
``Llama`` and ``Mistral`` model classes.
There are multiple sharding strategies for K/V cache, in order to satisfy different constraints.

- ``GQA.SHARD_OVER_HEADS`` distributes K/V caches along head dimension. This can be only used when K/V heads is multiple of tensor-parallelism degree. This is the default configuration.
- ``GQA.SHARD_OVER_BATCH`` distributes K/V caches along batch dimension. This can be only used when batch size is multiple of tensor-parallelism degree. This can be useful for large-batch inference.
- ``GQA.REPLICATED_HEADS`` replicates K/V heads. This can be used when neither batch size nor K/V heads can be divisible by tensor-parallelism degree. This can be useful for low-latency small-batch inference.
- ``GQA.ALL_GATHER_HEADS`` evenly splits the K/V heads across all NeuronCores. This is optimized for large-batch inference of GQA model without replication.

.. _mistral_gqa_code_sample:

In the following example we demonstrate how to configure these distributed inference strategies and
perform inference with the ``Mistral`` model:

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from transformers_neuronx import MistralForSampling, GQA, NeuronConfig

    # Set sharding strategy for GQA to be shard over heads
    neuron_config = NeuronConfig(
        group_query_attention=GQA.SHARD_OVER_HEADS
    )

    # Create and compile the Neuron model
    model_neuron = MistralForSampling.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', amp='bf16', neuron_config=neuron_config)
    model_neuron.to_neuron()

    # Get a tokenizer and exaple input
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    text = "[INST] What is your favourite condiment? [/INST]"
    encoded_input = tokenizer(text, return_tensors='pt')

    # Run inference
    with torch.inference_mode():
        generated_sequence = model_neuron.sample(encoded_input.input_ids, sequence_length=256, start_ids=None)
    print([tokenizer.decode(tok) for tok in generated_sequence])



Repeated Ngram Filtering
------------------------

Repeated Ngram Filtering reduces redundant ngram phrases within the generated text. It uses the same API as :ref:`HuggingFace API for NoRepeatedNGram <https://huggingface.co/docs/transformers/v4.38.2/en/internal/generation_utils#transformers.NoRepeatNGramLogitsProcessor>`. Set the parameter no_repeat_ngram_size to the size of ngram phrases to be filtered and pass it to the sampling function as in the example ``model.sample(inputs_ids, no_repeat_ngram_size=3)``




Top-K on-device sampling support [Beta]
--------------------------------------
Transformers Neuron supports Top-K Sampling on-device for all models except Mixtral models. 
In the following example, we demonstrate how to use on-device Top-K for the ``Llama`` model via 
the ``GenerationConfig`` and ``NeuronConfig`` configs.

.. code-block:: python

    import torch
    from transformers_neuronx import LlamaForSampling
    from transformers_neuronx.config import NeuronConfig, GenerationConfig
    from transformers import AutoTokenizer

    neuron_config = NeuronConfig(
        on_device_generation=GenerationConfig(max_length=128, top_k=10, do_sample=True)
    )

    # Create and compile the Neuron model
    model_neuron = LlamaForSampling.from_pretrained('openlm-research/open_llama_3b', batch_size=1, tp_degree=8, n_positions=128, neuron_config=neuron_config)
    model_neuron.to_neuron()

    # Get a tokenizer and exaple input
    tokenizer = AutoTokenizer.from_pretrained('openlm-research/open_llama_3b')
    text = "Hello, I'm a language model,"
    encoded_input = tokenizer(text, return_tensors='pt')

    # Run inference
    with torch.inference_mode():
        generated_sequence = model_neuron.sample(encoded_input.input_ids, sequence_length=128, top_k=10)
        print([tokenizer.decode(tok) for tok in generated_sequence])



Running inference with multiple models
--------------------------------------

Multiple transformers-neuronx models can be loaded at the same time as long
as the total number of consumed NeuronCores is less than or equal to the total
number of NeuronCores on the instance. For example, three tp-degree=8 models can be
loaded and run in parallel on an inf2.48xlarge which has 24 NeuronCores. The 
``NEURON_RT_NUM_CORES`` and ``NEURON_RT_VISIBLE_CORES`` environment variables
can be used to allocate the necessary number of NeuronCores to each process
to run multiple transformers-neuronx models in parallel. See the
:ref:`torch_neuronx_core_placement_guide` section for additional information
about how to use these environment variables.

It is important to notice that when multiple models are used on a single instance, 
the number of threads should be reduced to avoid race condition on host side. 
Assume the neuron instance (i.e. trn1) has 192 CPU cores. 
If one of the models keeps all CPU cores busy, there would be significant performance 
degradation in the rest of models. As a result, the number of threads for each model
should be limited to part of available cores. To do this, ``OMP_NUM_THREADS`` environment
variable can be set. For example, if there are 192 CPU cores available and four tp-degree=8 
models are used, one can export OMP_NUM_THREADS=48 to avoid race condition.


Streamer
----------------------------

LLMs generate tokens in auto-regressive loop. A model.sample call waits till 
the end of full sequence generation before returning the generated response.
It is possible to output an output token as soon as it is generated. To do this, 
a streamer object can be used. Streamer is an object which has 2 methods: put and end. 
There are several predefined streamer in transformers library such as TextIteratorStreamer.
The following example shows how to define a streamer and use it in transformers-neuronx:

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from transformers_neuronx import MistralForSampling, GQA

    import transformers
    from time import time

    # Create a custom streamer inherited from transformers.generation.streamers.BaseStreamer
    class CustomStreamer(transformers.generation.streamers.BaseStreamer):
        def __init__(self) -> None:
            self.reset()

        def reset(self):
            self.token_latencies = []
            self.iter = 0
            self.now = time()

        def put(self, tokens):
            now = time()
            token_latency = now - self.now
            print(f"Iteration {self.iter:4d}: Latency [s] {token_latency:6.3f} -- Token {tokens}")
            self.now = now
            self.iter += 1
            self.token_latencies.append(token_latency)


        def end(self):
            print("First 10 token latencies:", self.token_latencies[:10])


    # Create and compile the Neuron model
    model_neuron = MistralForSampling.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', amp='bf16')
    model_neuron.to_neuron()

    # Get a tokenizer and exaple input
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    text = "[INST] What is your favourite condiment? [/INST]"
    encoded_input = tokenizer(text, return_tensors='pt')

    streamer = CustomStreamer()
    # Run inference
    with torch.inference_mode():
        generated_sequence = model_neuron.sample(encoded_input.input_ids, sequence_length=256, start_ids=None, streamer=streamer)


Stopping Criteria
------------------
We can define custom stopping criteria to stop autoregressive loop. For example, if
we want to limit autoregressive loop after 0.5s, we can define and use stopping criteria 
class as follows:


.. code-block:: python

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    from transformers_neuronx import MistralForSampling, GQA, NeuronConfig
    from transformers_neuronx.stopping_criteria import StoppingCriteria, StoppingCriteriaList

    from time import time
    from typing import List, Optional, Callable


    class MaxTimeCriteria(StoppingCriteria):
        """
        This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
        time will start being counted when you initialize this function. You can override this by passing an
        `initial_time`.

        Args:
            max_time (`float`):
                The maximum allowed time in seconds for the generation.
            initial_time (`float`, *optional*, defaults to `time()`):
                The start of the generation allowed time.
        """

        def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
            self.max_time = max_time
            self.initial_timestamp = time() if initial_timestamp is None else initial_timestamp

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            dt = time() - self.initial_timestamp
            end_condition = dt > self.max_time
            if end_condition:
                print("Stopping!")
            return end_condition

    # Create a streamer. This can be a custom streamer too inherited from transformers.generation.streamers.BaseStreamer
    class CustomStreamer(transformers.generation.streamers.BaseStreamer):
        def __init__(self) -> None:
            self.reset()

        def reset(self):
            self.token_latencies = []
            self.iter = 0
            self.now = time()

        def put(self, tokens):
            now = time()
            token_latency = now - self.now
            print(f"Iteration {self.iter:4d}: Latency [s] {token_latency:6.3f} -- Token {tokens}")
            self.now = now
            self.iter += 1
            self.token_latencies.append(token_latency)


        def end(self):
            pass

    # Create and compile the Neuron model
    model_neuron = MistralForSampling.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2', amp='bf16')
    model_neuron.to_neuron()

    # Get a tokenizer and exaple input
    tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
    text = "[INST] What is your favourite condiment? [/INST]"
    encoded_input = tokenizer(text, return_tensors='pt')

    # Add stopping criteria to stop after 0.5 seconds
    stopping_criteria_list= StoppingCriteriaList([MaxTimeCriteria(0.5)])
    streamer = CustomStreamer()

    # Run inference
    with torch.inference_mode():
        model_neuron.sample(input_ids=encoded_input.input_ids, sequence_length=256, stopping_criteria_list=stopping_criteria_list, streamer=streamer)


Speculative sampling [Beta]
---------------------------

Transformers Neuron supports speculative sampling for the ``Llama`` and ``GPT2``
model classes. In speculative sampling, we use use a smaller draft model to speculate future tokens. 
These are then sent to the larger target model, which accepts or rejects these tokens. 
For more detailed information, see the original proposal by 
DeepMind titled :ref:`Accelerating Large Language Model Decoding with Speculative Sampling <https://arxiv.org/abs/2302.01318>`.
Speculative sampling is currently supported for batch size 1. 

In the following example, we demonstrate how to perform speculative sampling using the ``Llama`` model.

.. code-block:: python

    import torch
    from transformers import LlamaTokenizer
    from transformers_neuronx import LlamaForSampling
    from transformers_neuronx.speculation import SpeculativeGenerator
    
    # Load draft model
    draft_neuron_model = LlamaForSampling.from_pretrained('openlm-research/open_llama_3b', n_positions=256, batch_size=1, tp_degree=8, amp='f32')
    # Compile the model
    draft_neuron_model.to_neuron()

    # Load target model
    target_neuron_model = LlamaForSampling.from_pretrained('openlm-research/open_llama_13b', n_positions=256, batch_size=1, tp_degree=8, amp='f32')
    # Enable speculative decoder
    target_neuron_model.enable_speculative_decoder(4)
    # Compile the model
    target_neuron_model.to_neuron()

    # Initialize tokenizer and text prompt
    tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_13b")
    prompt = "Hello, I'm a generative AI language model."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Create SpeculativeGenerator
    spec_gen = SpeculativeGenerator(draft_neuron_model, target_neuron_model, 4)

    # Call speculative sampling on given input
    response = spec_gen.sample(
        input_ids=input_ids,
        sequence_length=30,
    )

    # Decode the response
    generated_text = tokenizer.decode(response[0])
    print(f"\nDecoded tokens: {generated_text}")


