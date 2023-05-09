.. _transformers_neuronx_readme:

Transformers Neuron (``transformers-neuronx``) Developer Guide
==============================================================

Transformers Neuron for |Trn1|/|Inf2| is a software package that enables
PyTorch users to perform large language model (LLM) inference on
second-generation Neuron hardware (See: :ref:`NeuronCore-v2 <neuroncores-v2-arch>`).


Introduction
------------

The `Transformers Neuron repository <https://github.com/aws-neuron/transformers-neuronx>`_
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

Installation
------------

--------------------
Stable Release
--------------------

To install the most rigorously tested stable release, use the PyPI pip wheel:

::

    pip install transformers-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com

--------------------
Development Version
--------------------

To install the development version with the latest features and improvements, use ``git`` to install from the
`Transformers Neuron repository <https://github.com/aws-neuron/transformers-neuronx>`_:

::

   pip install git+https://github.com/aws-neuron/transformers-neuronx.git

.. raw:: html

   <details>
   <summary>Installation Alternatives</summary>
   <br>

Without ``git``, save the `Transformers Neuron repository <https://github.com/aws-neuron/transformers-neuronx>`_ package contents locally and use:

::

   pip install transformers-neuronx/ # This directory contains `setup.py`

Similarly, a standalone wheel can be created using the ``wheel`` package
with the local repository contents:

::

   pip install wheel
   cd transformers-neuronx/  # This directory contains `setup.py`
   python setup.py bdist_wheel
   pip install dist/transformers_neuronx*.whl

This generates an installable ``.whl`` package under the ``dist/``
folder.

.. raw:: html

   </details>

.. warning::
    The development version may contain breaking changes. Please use it with caution.
    Additionally, the APIs and functionality in the development version are
    subject to change without warning.


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
tensor-parallelism is neccessary as it provides a way to shard the
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

------------------------
Hugging Face generate() API support
------------------------

Transformers Neuron models support the Hugging Face `generate() <https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation#transformers.GenerationMixin.generate>`__
API via the ``HuggingFaceGenerationModelAdapter`` adapter class. In the following example we
demonstrate how to run sampling with temperature using the ``GPT2`` model:

.. code-block:: python

    from transformers_neuronx.gpt2.model import GPT2ForSampling
    from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
    from transformers_neuronx.module import save_pretrained_split
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load and save the CPU model
    model_cpu = AutoModelForCausalLM.from_pretrained('gpt2')
    save_pretrained_split(model_cpu, 'gpt2-split')

    # Create and compile the Neuron model
    model_neuron = GPT2ForSampling.from_pretrained('gpt2-split', batch_size=1, tp_degree=2, n_positions=256, amp='f32', unroll=None)
    model_neuron.to_neuron()

    # Use the `HuggingFaceGenerationModelAdapter` to access the generate API
    model = HuggingFaceGenerationModelAdapter(model_cpu.config, model_neuron)

    # Get a tokenizer and exaple input
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    text = "Hello, I'm a language model,"
    encoded_input = tokenizer(text, return_tensors='pt', padding=True)

    # Run inference using temperature
    model.reset_generation()
    sample_output = model.generate(
        input_ids=encoded_input.input_ids,
        attention_mask=encoded_input.attention_mask,
        do_sample=True, 
        max_length=256, 
        temperature=0.7,
    )
    print([tokenizer.decode(tok) for tok in sample_output])


------------------------
Serialization support
------------------------

Transformers Neuron supports model serialization (model saving and loading) for
the ``GPT2`` model class. Serialization support for additional model classes
will be added in an uncoming relesae. In the following example we demonstrate
how to save and load the ``GPT2`` model:

.. code-block:: python

    from transformers_neuronx.gpt2.model import GPT2ForSampling
    from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
    from transformers_neuronx.module import save_pretrained_split
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load and save the CPU model
    model_cpu = AutoModelForCausalLM.from_pretrained('gpt2')
    save_pretrained_split(model_cpu, 'gpt2-split')

    # Create and compile the Neuron model
    model_neuron = GPT2ForSampling.from_pretrained('gpt2-split', batch_size=1, tp_degree=2, n_positions=256, amp='f32', unroll=None)
    model_neuron.to_neuron()

    # Save the compiled Neuron model
    model_neuron._save_compiled_artifacts('gpt2-neuron')

    # Load the Neuron model
    model_neuron = GPT2ForSampling.from_pretrained('gpt2-split', batch_size=1, tp_degree=2, n_positions=256, amp='f32', unroll=None)
    model_neuron._load_compiled_artifacts('gpt2-neuron') # Load the compiled Neuron artifacts
    model_neuron.to_neuron() # Load the model weights but skip compilation


Examples
--------

The `AWS Neuron Samples GitHub
Repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx>`__
contains examples of running autoregressive sampling using HuggingFace
transformers checkpoints on Inf2 & Trn1.

Currently supported models
--------------------------

-  `GPT2 <https://huggingface.co/docs/transformers/model_doc/gpt2>`__
-  `GPT-J <https://huggingface.co/docs/transformers/model_doc/gptj>`__
-  `OPT <https://huggingface.co/docs/transformers/model_doc/opt>`__

Release notes
-----------------

Visit the :ref:`transformers-neuronx-rn` section.


Upcoming features
-----------------

Performance metrics
~~~~~~~~~~~~~~~~~~~

The ``transformers-neuronx`` samples currently provide limited
performance data. We are looking into adding additional metrics, such as
``tokens / second`` and latency measurements.

Troubleshooting
---------------

Please refer to our `Contact
Us <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/contact.html>`__
page for additional information and support resources. If you intend to
file a ticket and you can share your model artifacts, please re-run your
failing script with ``NEURONX_DUMP_TO=./some_dir``. This will dump
compiler artifacts and logs to ``./some_dir``. You can then include this
directory in your correspondance with us. The artifacts and logs are
useful for debugging the specific failure.
