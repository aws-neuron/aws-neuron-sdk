.. _nxdi-model-reference:

NxD Inference - Production Ready Models
=======================================

Neuronx Distributed Inference provides production ready models that you can
directly use for seamless deployment. You can view the source code for all
supported models in the `NxD Inference GitHub repository <https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models>`__. 

.. note:: 
   
   If you are looking to deploy a custom model integration, you can follow the
   :ref:`model onboarding guide <nxdi-onboarding-models>`. You can refer to the source
   code for supported models in the `NxD Inference GitHub repository <https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models>`__
   and make custom changes required for your use case.

.. contents:: Table of contents
   :local:
   :depth: 2

Using Models to Run Inference
-----------------------------

You can run models through vLLM or integrate directly with NxD
Inference.

Using vLLM
~~~~~~~~~~

If you are using vLLM for production deployment, we recommend that you
use the vLLM API to integrate with NxD Inference. The vLLM API automatically
chooses the correct model and config classes based on the model's config file.
For more information, refer to the :ref:`nxdi-vllm-user-guide`.

Integrating Directly with NxD Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use NxD Inference directly, you construct model and configuration
classes. For more information about which model and configuration classes to use for each
model, see :ref:`nxdi-supported-model-architectures`. To see an example of how to
run inference directly with NxD Inference, see the `generation_demo.py
script <https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/examples/generation_demo.py>`__.

.. _nxdi-supported-model-architectures:

Supported Model Architectures
-----------------------------

NxD Inference currently provides support for the following model
architectures.

Llama (Text)
~~~~~~~~~~~~

NxD Inference supports Llama text models. The Llama model architecture
supports all Llama text models, including Llama 2, Llama 3, Llama 3.1,
Llama 3.2, and Llama 3.3. You can also use the Llama model architecture
to run any model based on Llama, such as Mistral.

Neuron Classes
^^^^^^^^^^^^^^

- Neuron config class: NeuronConfig
- Inference config class: LlamaInferenceConfig
- Causal LM model class: NeuronLlamaForCausalLM

Compatible Checkpoint Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct (requires
  Trn2)
- https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

Llama (Multimodal)
~~~~~~~~~~~~~~~~~~

NxD Inference supports Llama 3.2 multimodal models. You can use HuggingFace
checkpoints or the original Meta checkpoints. To use the Meta checkpoint,
you must first convert the checkpoint to Neuron format. For more information
about how to run Llama3.2 multimodal inference, and for details about 
how to convert the original Meta checkpoints to run on NxD Inference, see :ref:`/libraries/nxd-inference/tutorials/llama3.2-multimodal-tutorial.ipynb`.

.. _neuron-classes-0:

Neuron Classes
^^^^^^^^^^^^^^

- Neuron config class: MultimodalVisionNeuronConfig
- Inference config class: MllamaInferenceConfig
- Causal LM model class: NeuronMllamaForCausalLM

.. _compatible-checkpoint-examples-0:

Compatible Checkpoint Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct

Llama4
~~~~~~~~~~~~~~~~~~

NxD Inference supports Llama 4 models, including both Scout and Maverick checkpoints.
You can use Hugging Face checkpoints. Both checkpoints leverage early fusion for native multimodality,
enabling them to process text and image inputs. For more information
about how to run Llama 4 inference, see :ref:`/libraries/nxd-inference/tutorials/llama4-tutorial.ipynb`.

.. _neuron-classes-1:

Neuron Classes
^^^^^^^^^^^^^^

- Neuron config class: Llama4NeuronConfig
- Inference config class: Llama4InferenceConfig
- Causal LM model class: NeuronLlama4ForCausalLM

.. _compatible-checkpoint-examples-1:

Compatible Checkpoint Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
- https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct

Mixtral
~~~~~~~

NxD Inference supports models based on the Mixtral model architecture,
which uses mixture-of-experts (MoE) architecture.

.. _neuron-classes-2:

Neuron Classes
^^^^^^^^^^^^^^

- Neuron config class: MoENeuronConfig
- Inference config class: MixtralInferenceConfig
- Causal LM model class: NeuronMixtralForCausalLM

.. _compatible-checkpoint-examples-2:

Compatible Checkpoint Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1

DBRX
~~~~

NxD Inference supports models based on the DBRX model architecture,
which uses mixture-of-experts (MoE) architecture.

.. _neuron-classes-3:

Neuron Classes
^^^^^^^^^^^^^^

- Neuron config class: DbrxNeuronConfig
- Inference config class: DbrxInferenceConfig
- Causal LM model class: NeuronDbrxForCausalLM

.. _compatible-checkpoint-examples-3:

Compatible Checkpoint Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- https://huggingface.co/databricks/dbrx-instruct

Qwen2.5
~~~~~~~~

NxD Inference supports models based on the Qwen2.5 model architecture.

.. _neuron-classes-4:

Neuron Classes
^^^^^^^^^^^^^^

- Neuron config class: Qwen2NeuronConfig
- Inference config class: Qwen2InferenceConfig
- Causal LM model class: NeuronQwen2ForCausalLM

.. _compatible-checkpoint-examples-4:

Compatible Checkpoint Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- https://huggingface.co/Qwen/Qwen2.5-72B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-14B-Instruct (Not tested, but expected to work out of the box)
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct (Not tested, but expected to work out of the box)
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct (Not tested, but expected to work out of the box)
- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Qwen3
~~~~~~

NxD Inference supports models based on the Qwen3 model architecture.

.. _neuron-classes-5:

Neuron Classes
^^^^^^^^^^^^^^

- Neuron config class: Qwen3NeuronConfig
- Inference config class: Qwen3InferenceConfig
- Causal LM model class: NeuronQwen3ForCausalLM

.. _compatible-checkpoint-examples-5:

Compatible Checkpoint Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- https://huggingface.co/Qwen/Qwen3-0.6B
- https://huggingface.co/Qwen/Qwen3-1.7B
- https://huggingface.co/Qwen/Qwen3-4B
- https://huggingface.co/Qwen/Qwen3-8B
- https://huggingface.co/Qwen/Qwen3-14B
- https://huggingface.co/Qwen/Qwen3-32B


FLUX.1 [BETA]
~~~~~~~~~~~~~~~~~~

NxD Inference supports FLUX.1-dev model checkpoint for text to image generation.
You can use Hugging Face checkpoints. For more information
about how to run FLUX.1-dev inference, see :ref:`/libraries/nxd-inference/tutorials/flux-inference-tutorial.ipynb`.

.. _neuron-classes-6:

Neuron Classes
^^^^^^^^^^^^^^

- Flux Application class: NeuronFluxApplication
- Flux Pipeline class: NeuronFluxPipeline
- Flux Backbone Neuron config class: FluxBackboneInferenceConfig

.. _compatible-checkpoint-examples-6:

Compatible Checkpoint Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- https://huggingface.co/black-forest-labs/FLUX.1-dev
