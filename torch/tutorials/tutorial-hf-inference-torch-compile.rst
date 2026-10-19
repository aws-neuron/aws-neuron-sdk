.. meta::
   :description: Compile a Hugging Face Transformers model with torch.compile and the neuron backend on AWS Trainium, and generate text with it. Covers the static cache, chunked prefill, decoding, batches, chat templates, and tensor parallelism.
   :keywords: TorchNeuron, torch-neuronx, AWS Neuron, Trainium, tutorial, inference, Hugging Face, transformers, generate, text generation, torch.compile, CompileConfig, static cache, tensor parallelism, tp_plan, Qwen3, Qwen2.5
   :date-modified: 2026-10-15

.. _torchneuron-tutorial-hf-inference-torch-compile:

===================================================================
Tutorial: Run a Hugging Face model for inference with torch.compile
===================================================================

This tutorial shows you how to compile a Hugging Face Transformers model with
``torch.compile`` and the ``neuron`` backend on AWS Trainium. You then generate
text with the compiled model. The tutorial uses the steps of the Hugging Face
`Text generation <https://huggingface.co/docs/transformers/en/llm_tutorial>`_
tutorial. When you complete it, you can compile ``generate()`` on a NeuronCore
and control its output. You can also shard a model across NeuronCores and
compile it.

Overview
--------

``generate()`` compiles the forward pass of a model when the model uses a
static cache and a ``CompileConfig``. With chunked prefill, ``generate()`` also
runs the prompt through the compiled forward pass. With the ``neuron`` backend,
``torch.compile`` then makes one Neuron program for the prefill and one for the
decoding step. In this tutorial, you:

- Load Qwen3-8B on a NeuronCore and turn on compilation for ``generate()``.
- Generate text. The first call compiles the model.
- Change the generation configuration, and save it.
- Control the output length, the decoding strategy, and a batch of prompts.
- Use a chat template.
- Shard a model across two NeuronCores, and compile it.

Before you start
----------------

This tutorial assumes that you have experience in the following areas:

- Hugging Face Transformers: models, tokenizers, and ``generate()``.
- PyTorch on Neuron. See :doc:`/torch/get-started/quickstart-torch-neuron`.
- ``torch.compile`` on Neuron. See :doc:`/torch/compile/index`.

Model details
-------------

This tutorial uses these models from the Hugging Face Hub:

- ``Qwen/Qwen3-8B`` for Steps 1 to 8.
- ``Qwen/Qwen2.5-14B-Instruct`` for the tensor parallelism example in Step 9.

The models load in their checkpoint data type, ``bfloat16``. The Qwen3-8B
weights are approximately 16.4 GB. One logical NeuronCore of a Trn2 instance
has 24 GB of memory, so Qwen3-8B runs on one NeuronCore. The
Qwen2.5-14B-Instruct weights are approximately 29.5 GB, so you must shard this
model across two or more NeuronCores.


Prerequisites
-------------

- **A Trn2 instance with torch-neuronx installed and verified:** See
  :doc:`/torch/get-started/quickstart-torch-neuron`.
- **Hugging Face Transformers:** Tested with ``transformers`` 5.19.0.
- **Access to the Hugging Face Hub:** The models download on first use. A
  Hugging Face token is not necessary for these models.

----

Prepare your environment
------------------------

Install or update Transformers in the Python environment that has
torch-neuronx:

.. code-block:: bash

   pip install transformers==5.19.0

You do not have to import ``torch_neuronx`` to use the ``neuron`` device or the
``neuron`` backend. PyTorch loads torch-neuronx automatically when Python
imports ``torch``. See :doc:`TorchNeuron overview </torch/about-pytorch-neuron>`.

Step 1: Load the model on a NeuronCore
--------------------------------------

In this step, you load Qwen3-8B and its tokenizer.

Pass ``device_map="neuron"`` to ``from_pretrained()``:

.. code-block:: python

   from transformers import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", device_map="neuron")
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", padding_side="left")
   print(model.device)

.. code-block:: text

   neuron:0

.. important::

   Use ``device_map="neuron"``, not ``device_map="auto"``. Transformers
   support for ``neuron`` with ``"auto"`` is not available yet.

Step 2: Turn on compilation for generate()
------------------------------------------

In this step, you set the generation configuration of the model. Then each
``generate()`` call compiles the model with ``torch.compile``.

Set these values in ``model.generation_config``:

.. code-block:: python

   from transformers import CompileConfig

   model.generation_config.cache_implementation = "static"
   model.generation_config.compile_config = CompileConfig(
       backend="neuron", dynamic=False, mode="default"
   )
   model.generation_config.max_cache_len = 128
   model.generation_config.prefill_chunk_size = 64
   model.generation_config.do_sample = False
   model.generation_config.temperature = None
   model.generation_config.top_p = None
   model.generation_config.top_k = None

- ``cache_implementation="static"``: ``generate()`` compiles the model only
  with a static cache. A static cache has a fixed length, so the shapes of the
  decoding step do not change.
- ``CompileConfig(backend="neuron", dynamic=False, mode="default")``:
  ``backend="neuron"`` selects the torch-neuronx backend for ``torch.compile``.
  The default backend, Inductor, does not support Neuron. ``dynamic=False`` is
  necessary because torch-neuronx does not compile graphs with dynamic shapes.
  torch-neuronx supports only the default mode. Set ``mode="default"`` to
  prevent a warning for each compiled graph.
- ``max_cache_len=128``: the length of the static cache. Set it to the longest
  padded prompt plus ``max_new_tokens`` that you expect. Then calls with a
  different ``max_new_tokens`` use the same compiled programs.
- ``prefill_chunk_size=64``: ``generate()`` runs the prompt through the compiled
  forward pass in chunks of 64 tokens. Pad each prompt to 64 tokens (see
  Step 3). Then each chunk has the same shape, and the prefill uses one
  compiled program for all prompts of up to 64 tokens.
- ``do_sample=False`` and no ``temperature``, ``top_p``, or ``top_k``: the
  Qwen3 checkpoint sets these values for sampling. Greedy decoding gives the
  same output at each call. Top-p sampling sorts the full vocabulary at each
  step, outside the compiled program, and is slow on Neuron.

Step 3: Generate text
---------------------

In this step, you generate text. The first call compiles the model.

Pad the prompt on the left to 64 tokens, the ``prefill_chunk_size`` from
Step 2, and move the inputs to ``model.device``. Then pass them to
``generate()`` and decode the output tokens:

.. code-block:: python

   model_inputs = tokenizer(
       ["A list of colors: red, blue"], padding="max_length", max_length=64, return_tensors="pt"
   ).to(model.device)
   generated_ids = model.generate(**model_inputs, max_new_tokens=20)
   print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

.. code-block:: text

   A list of colors: red, blue, green, yellow, purple, orange, black, white, pink, teal, brown, gray

The first call compiles the prefill and the decoding step, so it takes longer
than the next calls. Run the same code again: the second call uses the
compiled programs. A prompt with a different length also uses them, because
the tokenizer pads it to 64 tokens.

torch-neuronx keeps the compiled programs in a cache on the disk. A later
call with the same shapes uses the cache, even in a new Python process.

Step 4: Change the generation configuration
-------------------------------------------

In this step, you read the generation configuration, change it for one call,
and save it.

The ``generation_config`` attribute shows the values that are different from
the defaults, including the compilation settings from Step 2:

.. code-block:: python

   print(model.generation_config)

.. code-block:: text

   GenerationConfig {
     "bos_token_id": 151643,
     "cache_implementation": "static",
     "compile_config": {
       "backend": "neuron",
       "dynamic": false,
       "fullgraph": false,
       "mode": "default",
       "options": null
     },
     "do_sample": false,
     "eos_token_id": [
       151645,
       151643
     ],
     "max_cache_len": 128,
     "pad_token_id": 151643,
     "prefill_chunk_size": 64
   }

To change a setting for one call, pass it to ``generate()``. This example
uses top-k sampling:

.. code-block:: python

   generated_ids = model.generate(**model_inputs, max_new_tokens=20, do_sample=True, top_k=50)
   print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

The call uses the same compiled programs as Step 3. The output changes at each
call, because the call uses sampling.

To keep a set of settings, save a ``GenerationConfig`` to a directory and load
it again. ``save_pretrained()`` does not save ``compile_config``, so set it
again after you load the configuration:

.. code-block:: python

   from transformers import GenerationConfig

   generation_config = GenerationConfig(
       max_new_tokens=50,
       do_sample=True,
       top_k=50,
       eos_token_id=model.config.eos_token_id,
       pad_token_id=tokenizer.pad_token_id,
       cache_implementation="static",
       max_cache_len=128,
       prefill_chunk_size=64,
   )
   generation_config.save_pretrained("./my_generation_config")

   generation_config = GenerationConfig.from_pretrained("./my_generation_config")
   generation_config.compile_config = CompileConfig(
       backend="neuron", dynamic=False, mode="default"
   )
   generated_ids = model.generate(**model_inputs, generation_config=generation_config)
   print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

Step 5: Control the output length
---------------------------------

In this step, you set the number of new tokens.

If you do not set ``max_new_tokens``, ``generate()`` adds a maximum of 20
tokens to the prompt:

.. code-block:: python

   model_inputs = tokenizer(
       ["A sequence of numbers: 1, 2"], padding="max_length", max_length=64, return_tensors="pt"
   ).to(model.device)

   generated_ids = model.generate(**model_inputs)
   print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

   generated_ids = model.generate(**model_inputs, max_new_tokens=50)
   print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

.. code-block:: text

   A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8,
   A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,

Both calls use the programs that Step 3 compiled, because the padded prompt
and the new tokens fit in ``max_cache_len``.

Step 6: Select a decoding strategy
----------------------------------

In this step, you compare greedy search and multinomial sampling.

Greedy search selects the most probable token at each step. Set
``do_sample=True`` for multinomial sampling:

.. code-block:: python

   model_inputs = tokenizer(
       ["I am a cat."], padding="max_length", max_length=64, return_tensors="pt"
   ).to(model.device)

   generated_ids = model.generate(**model_inputs, max_new_tokens=20)
   print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

   generated_ids = model.generate(**model_inputs, max_new_tokens=20, do_sample=True)
   print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])

The greedy call prints:

.. code-block:: text

   I am a cat. I have a name, but I don't know what it is. I have a collar, but

The sampled output changes at each call. ``generate()`` selects the next
token outside the compiled programs, so both strategies use the same programs.

Step 7: Generate for a batch of prompts
---------------------------------------

In this step, you generate text for two prompts of different lengths in one
batch.

Pad the prompts on the left. The model continues from the last token of each
row. Padding on the right puts pad tokens between the prompt and the new text.

.. code-block:: python

   model_inputs = tokenizer(
       ["1, 2, 3", "A, B, C, D, E"], padding="max_length", max_length=64, return_tensors="pt"
   ).to(model.device)
   generated_ids = model.generate(**model_inputs, max_new_tokens=20)
   print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

.. code-block:: text

   ['1, 2, 3, 4, 5, 6, 7, 8, 9, ', 'A, B, C, D, E, F, G, H, I, J, K, L, M, N, O']

A batch of 2 runs the prefill and the decoding step for 2 sequences. Thus the
first call with this batch size compiles new programs.

Step 8: Use a chat template
---------------------------

In this step, you format a conversation with the chat template of the model.

Pass the messages to ``apply_chat_template()`` to get the prompt text. Set
``enable_thinking=False`` to turn off the thinking mode of Qwen3. Then
tokenize and pad the prompt as in Step 3:

.. code-block:: python

   messages = [
       {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
       {"role": "user", "content": "How many cats does it take to change a light bulb?"},
   ]
   prompt = tokenizer.apply_chat_template(
       messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
   )
   model_inputs = tokenizer(
       [prompt], padding="max_length", max_length=64, return_tensors="pt"
   ).to(model.device)
   input_length = model_inputs["input_ids"].shape[1]
   generated_ids = model.generate(**model_inputs, max_new_tokens=50)
   print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])

.. code-block:: text

   Arrr, ye be askin’ a question fit for a scallywag! Aye, how many cats be needin’ to change a light bulb? Well, me hearties, it taketh one cat, but only if it

Step 9: Shard a model across NeuronCores and compile it
-------------------------------------------------------

In this step, you shard a model that is too large for one NeuronCore with
tensor parallelism, and compile it.

On Trn2, one logical NeuronCore has two physical cores that share 24 GB of
high-bandwidth memory (HBM). If the weights and the cache of a model are larger
than this memory, you must shard the model across NeuronCores. With
``device_map="neuron"``, Qwen2.5-14B-Instruct stops with an out-of-memory
error.

Transformers can shard a model with the tensor-parallel plan in its
configuration. Start one process for each NeuronCore with ``torchrun``. In each
process, start the process group before any other Neuron call. Pass
``distributed_config={"tp_plan": "auto", "tp_size": world_size}`` to
``from_pretrained()``. Then turn on compilation as in Step 2. Save this code as
``generate_tp.py``:

.. code-block:: python

   import torch
   import torch.distributed as dist
   from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig

   dist.init_process_group(backend="neuron")
   world_size = dist.get_world_size()

   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen2.5-14B-Instruct",
       dtype=torch.bfloat16,
       distributed_config={"tp_plan": "auto", "tp_size": world_size},
   )
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct", padding_side="left")

   model.generation_config.cache_implementation = "static"
   model.generation_config.compile_config = CompileConfig(
       backend="neuron", dynamic=False, mode="default"
   )
   model.generation_config.max_cache_len = 128
   model.generation_config.prefill_chunk_size = 64
   model.generation_config.do_sample = False
   model.generation_config.temperature = None
   model.generation_config.top_p = None
   model.generation_config.top_k = None

   model_inputs = tokenizer(
       ["A list of colors: red, blue"], padding="max_length", max_length=64, return_tensors="pt"
   ).to("neuron")
   generated_ids = model.generate(**model_inputs, max_new_tokens=20)
   if dist.get_rank() == 0:
       print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
   dist.destroy_process_group()

Stop the Python session from Steps 1 to 8 first. That session uses the
NeuronCores, so the ``torchrun`` processes cannot get them. Then run the
script on two NeuronCores:

.. code-block:: bash

   torchrun --nproc-per-node 2 generate_tp.py

.. code-block:: text

   A list of colors: red, blue, green, yellow, orange, purple, pink, brown, black, white.

Each process holds half of the sharded weights and compiles its part of the
model. The tensor-parallel degree (``--nproc-per-node``) must divide the number
of key-value heads of the model (``num_key_value_heads`` in its
``config.json``). Qwen2.5-14B-Instruct has 8 key-value heads, so you can use
2, 4, or 8 NeuronCores.

.. note::

   If the configuration of a model has no tensor-parallel plan,
   ``"tp_plan": "auto"`` in ``distributed_config`` does not shard the model.
   Then every process loads all of the weights.

To make sure that Transformers shards the weights, check that the parameters
of the model are ``DTensor`` objects.


----

Confirmation
------------

With greedy decoding, the compiled ``generate()`` gives the same text as
``generate()`` without compilation. To check your setup, run Step 3 two times.
The first call compiles the model. Both calls print:

.. code-block:: text

   A list of colors: red, blue, green, yellow, purple, orange, black, white, pink, teal, brown, gray

Congratulations! You compiled a Hugging Face model with
``torch.compile`` on AWS Trainium. If you encountered any issues, see the
**Common issues** section below.

Common issues
-------------

- **The model runs on the CPU:** ``device_map="auto"`` selects the CPU on a
  Neuron instance. Use ``device_map="neuron"``. Check ``model.device``.
- ``ValueError: When passing device_map as a string, ... but found neuron``:
  ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` is set, so PyTorch did not load
  torch-neuronx. Unset the variable, or add ``import torch_neuronx`` before
  you load the model.
- ``InductorError ... no scheduling registered for device neuron``: The
  static cache is set, but ``compile_config`` is not. Transformers then
  compiles with the default Inductor backend, which does not support Neuron.
  Set ``compile_config=CompileConfig(backend="neuron", dynamic=False, mode="default")``.
  See `Step 2: Turn on compilation for generate()`_.
- ``RuntimeError: Compilation error occurred on Neuron`` with
  ``[NCC_EMOD023] The StableHLO module could not be converted to HLO``: The
  model compiled again with dynamic shapes, which torch-neuronx does not
  support. This occurs after a call with a new shape when ``dynamic`` is not
  ``False``. Use ``CompileConfig(backend="neuron", dynamic=False, mode="default")``
  and set ``max_cache_len``.
- **A call compiles again:** Some changes make new shapes for the prefill or
  the decoding step: a new batch size, a prompt that you do not pad to
  ``prefill_chunk_size``, or a longer static cache. Then ``generate()``
  compiles a new program. After one call uses a longer static cache, all later
  calls on the model also use the longer cache. They do not use the programs
  from before that call. To prevent this, pad each prompt to
  ``prefill_chunk_size``. Keep the batch size the same. Set ``max_cache_len``
  for the longest call that you expect.
- **A prompt is longer than 64 tokens:** Set ``prefill_chunk_size`` and the
  ``max_length`` of the tokenizer to the length of your longest prompt.
  Increase ``max_cache_len`` by the same number of tokens. Then the prefill
  stays one chunk with one compiled program.
- **Beam search does not use the compiled program:** In Transformers 5.19.0,
  only greedy search and sampling use the compiled program. A call with
  ``num_beams`` greater than 1 runs without it.
- **Sampling is very slow:** The generation configuration has a ``top_p``
  value. Top-p sampling sorts the full vocabulary at each step, outside the
  compiled program. Set ``top_p=None``, and use ``top_k`` instead.
- ``RuntimeError: NRT Execution error occurred on Neuron for operation=neuron::alloc::lazy``
  with a ``Neuron OOM`` warning: The model is too large for the memory of one
  NeuronCore. Shard it across NeuronCores. See
  `Step 9: Shard a model across NeuronCores and compile it`_.

Related information
-------------------

- :doc:`/torch/compile/index`
- :doc:`/torch/compile/how-to-optimize-with-torch-compile`
- `Hugging Face Text generation tutorial <https://huggingface.co/docs/transformers/en/llm_tutorial>`_
