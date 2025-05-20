.. _neuronx_distributed_inference_developer_guide:

Developer guide for Neuronx-Distributed Inference
=================================================

Neuronx-Distributed (NxD Core) provides fundamental building blocks that enable you to run
advanced inference workloads on AWS Inferentia and Trainium instances. These building
blocks include parallel linear layers that enable distributed inference, a model builder
that compiles PyTorch modules into Neuron models, and more.

Neuron also offers Neuronx-Distributed (NxD) Inference,
which is a library that provides optimized model and module implementations that build on top
of NxD Core. We recommend that you use NxD Inference to run inference workloads and onboard
custom models. For more information about NxD Inference, see :ref:`nxdi-overview`.

For examples of how to build directly on NxD Core, see the following:

* `Llama 3.2 1B inference sample <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/llama>`_
* T5 3B inference tutorial :ref:`[html] </src/examples/pytorch/neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`