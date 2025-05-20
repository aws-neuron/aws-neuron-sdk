.. _nxd-examples-migration-guide:

Migrating from NxD Core inference examples to NxD Inference
===========================================================

We have migrated the NeuronX Distributed Core `examples/inference <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference>`__
folder to a separate package, NeuronX Distributed (NxD) Inference
(``neuronx-distributed-inference``), so you can import and use it as a
proper library. This new library, NxD Inference, includes production ready
models that you can deploy out of the box with model inference backends,
such as vLLM. This library also provides modules that you can use to
implement your own models to run with the Neuron SDK.

If you use the inference examples from NxD Core, follow this guide to migrate
to NxD Inference. For more information about NxD Inference and to see examples
of how to use it, see :ref:`nxdi-feature-guide`, :ref:`NxD Inference Tutorials <nxdi-tutorials-index>`,
and the `generation_demo.py script <https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/examples/generation_demo.py>`__.

.. warning::
   Previous inference examples (including Llama 2, Llama 3, Mixtral, and DBRX) in
   the NxD Core GitHub repository were removed in Neuron Release 2.23.
   The models and example code are implemented in the
   NxD Inference library, so you can easily integrate them with your inference
   scripts. If you use these examples in NxD Core, we recommend
   that you update your inference scripts to use the NxD Inference model hub
   instead. If your use case requires you to directly integrate with the NxD
   Core library (and not NxD Inference) then you can continue to use the NxD
   Core library directly. For an example of how to integrate with NxD Core directly,
   see the newer `Llama3.2 1B sample <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/llama>`__
   added in Neuron Release 2.23. For more information, see :ref:`announce-eos-nxd-examples`.

.. contents:: Table of contents
   :local:
   :depth: 2

Changes
-------

1. New config interface
~~~~~~~~~~~~~~~~~~~~~~~

NxD Inference includes a new model config interface, ``InferenceConfig``,
where NeuronConfig is an attribute within the model config, and the
model config no longer extends HuggingFace's PretrainedConfig. NxDI
includes an adapter for loading an HuggingFace's config into this model
config. The configurations are serialized into a file named
``neuron_config.json``.

**This change means that the config structure is inverted compared to
the NxD examples folder.**

- To access the model config (similar to HuggingFace's
  PreTrainedConfig), use ``config`` (or ``model.config``,
  ``self.config``).
- To access the NeuronConfig, use ``config.neuron_config`` (or
  ``model.neuron_config``, ``self.neuron_config``).

To onboard a custom model, you define config classes that extend InferenceConfig
and NeuronConfig. The following example from DBRX shows how to define a
DBRX-specific NeuronConfig (NeuronDbrxConfig) and InferenceConfig
(DbrxInferenceConfig). DbrxInferenceConfig that defines required config
attributes and specifies that NeuronDbrxConfig is the NeuronConfig
class. The required attributes are typically set by loading a
PretrainedConfig (in this case, HuggingFace's DbrxConfig) into the
InferenceConfig. Alternatively, a user can manually provide these
attributes to avoid depending on an HuggingFace config class.

::

   class NeuronDbrxConfig(MoENeuronConfig):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.fused_qkv = True


   class DbrxInferenceConfig(InferenceConfig):
       def get_required_attributes(self) -> List[str]:
           return [
               "d_model",
               "n_heads",
               "max_seq_len",
               "emb_pdrop",
               "resid_pdrop",
               "pad_token_id",
               "vocab_size",
               "attn_config",
               "ffn_config",
           ]

       @classmethod
       def get_neuron_config_cls(cls):
           return NeuronDbrxConfig

.. note:: 

   NeuronDbrxConfig extends MoENeuronConfig, which is a subclass of NeuronConfig
   that includes attributes that are specific to mixture-of-experts (MoE) models.


To load the config from an HuggingFace checkpoint or a compiled
checkpoint, pass ``load_pretrained_config(path)`` as the ``load_config``
hook when you create the InferenceConfig.

::

   from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

   neuron_config = DbrxNeuronConfig()  # Provide args
   config = DbrxInferenceConfig(
       neuron_config,
       load_config=load_pretrained_config(model_path),
   )

To serialize the config, call ``save(path)``.

::

   config.save(compiled_model_path)

To deserialize the config, call ``load(path)``.

::

   config = DbrxInferenceConfig.load(compiled_model_path)

NeuronConfig also supports nested configs now. For example, see the
OnDeviceSamplingConfig class and its integration into NeuronConfig.

2. New base application interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NeuronApplicationBase takes general purpose features from
NeuronBaseForCausalLM, such as compile and load, and makes them
available in a new abstract base class. You can extend this base class
to define other types of application heads, such as for image
classification.

3. New generation inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Neuron model classes no longer extend HuggingFace's PretrainedModel,
so they no longer include a HuggingFace ``generate()`` function.
Additionally, GenerationConfig arguments are no longer passed through
the model config. To run HuggingFace generation in NxD Inference, wrap
the Neuron model in a HuggingFaceGenerationAdapter, and pass a
GenerationConfig when you call ``generate()``.

::

   from transformers import GenerationConfig

   from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

   # Init config, model, and tokenizer.

   generation_config = GenerationConfig.from_pretrained(model_path)
   generation_config_kwargs = {
       "do_sample": True,
       "top_k": 1,
       "pad_token_id": generation_config.eos_token_id,
       "max_length": neuron_config.max_length,
   }
   generation_config.update(**generation_config_kwargs)

   inputs = tokenizer(prompts, padding=True, return_tensors="pt")
   generation_model = HuggingFaceGenerationAdapter(model)
   outputs = generation_model.generate(
       inputs.input_ids,
       generation_config=generation_config,
       attention_mask=inputs.attention_mask,
   )

4. New quantization interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This new base class also includes an interface for quantization, which
was previously part of the ``run_llama_quantized.py`` example in the old
NxD examples folder. The following example saves a quantized checkpoint
for a Llama model. In this example, the ``config`` includes a
``neuron_config`` with quantization enabled.

::

   NeuronLlamaForCausalLM.save_quantized_state_dict(model_path, config)

5. Inference demo script (replaces runners)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In place of ``runner.py`` and various ``run_x.py`` examples, NxD-I
provides an ``inference_demo`` console script. When you run the script,
you provide a model path and configuration parameters to use for
inference. This script includes benchmarking and accuracy checking
features that you can use verify that your models and modules work
correctly.

The following example demonstrates how to run Llama-3-8b with token
matching and benchmarking enabled.

::

   inference_demo \ 
     --model-type llama \
     --task-type causal-lm \
     run \ 
       --model-path /home/ubuntu/model_hf/Llama-3.1-8b/ \ 
       --compiled-model-path /home/ubuntu/traced_model/Llama-3.1-8b/ \ 
       --torch-dtype bfloat16 \ 
       --tp-degree 32 \ 
       --batch-size 2 \ 
       --max-context-length 32 \ 
       --seq-len 64 \ 
       --on-device-sampling \ 
       --enable-bucketing \ 
       --top-k 1 \ 
       --do-sample \ 
       --pad-token-id 2 \ 
       --prompt "I believe the meaning of life is" \ 
       --prompt "The color of the sky is" \ 
       --check-accuracy-mode token-matching \ 
       --benchmark

For additional examples, see the ``neuronx-distributed-inference``
GitHub repository:
https://github.com/aws-neuron/neuronx-distributed-inference.