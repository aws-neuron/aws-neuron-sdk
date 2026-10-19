.. meta::
    :description: Home page for the NxD Inference for Training (NxDI) library included with the Neuron SDK.
    :date-modified: 12/02/2025

.. _neuronx-distributed-inference-index:


NxD Core for Inference
=======================

.. note:: Maintenance status

   The inference-context NxD Core documentation in this section is in **maintenance
   status** — it continues to receive critical fixes and remains the supported path for
   **Trn1/Inf2** inference and for custom **NxD Inference API** serving (for example
   Flux/diffusion). This content is **not** end-of-support. For new inference work, the
   go-forward path is the :ref:`vLLM Neuron plugin <vllm-neuron-home>`; the task-oriented
   :ref:`Inference on Neuron <inference-home>` page shows where each path fits.

NeuronX Distributed Core (NxD Core) is a package for supporting different distributed
inference mechanisms for Neuron devices. It provides XLA-friendly
implementations of some of the more popular distributed
inference techniques. As the size of the model scales, fitting
these models on a single device becomes impossible and hence we have to
make use of model sharding techniques to partition the model across
multiple devices. 

As part of this library, we enable support for Tensor
Parallelism sharding technique with other distributed library supported to be 
added in future.

.. _neuronx_distributed_inference_developer_guide:

About NeuronX-Distributed (NxD) Inference
------------------------------------------

NeuronX Distributed (NxD Core) provides fundamental building blocks that enable you to run advanced inference workloads on AWS Inferentia and Trainium instances. These building blocks include parallel linear layers that enable distributed inference, a model builder that compiles PyTorch modules into Neuron models, and more.

As part of NxD Core, Neuron offers NxD Inference, which is a library that provides optimized model and module implementations that build on top of NxD Core. For more information about NxD Inference, see :ref:`nxdi-overview`.

For examples of how to build directly on NxD Core, see the following:

* `Llama 3.2 1B inference sample <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/llama>`_
* T5 3B inference tutorial :ref:`[html] </src/examples/pytorch/neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`

.. toctree::
    :maxdepth: 1
    :hidden:

    Setup </archive/libraries/neuronx-distributed/setup/index>
    App Notes </archive/libraries/neuronx-distributed/app_notes>
    API Reference Guide </archive/libraries/neuronx-distributed/api-reference-guide>
    Developer Guide </inference-legacy/nxd-core/developer-guide-inference>
    LoRA Guide </archive/libraries/neuronx-distributed/lora_finetune_developer_guide>

    Tutorials  </archive/libraries/neuronx-distributed/tutorials/index>
    Misc  </archive/libraries/neuronx-distributed/neuronx-distributed-misc>

NxD Core for Inference Documentation
-------------------------------------

.. dropdown::  Setup  
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    
    .. include:: /archive/libraries/neuronx-distributed/setup/index.txt

.. dropdown::  App Notes  
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
   
    .. include:: /archive/libraries/neuronx-distributed/app_notes.txt

.. dropdown::  API Reference Guide  
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    
    .. include:: /archive/libraries/neuronx-distributed/api-reference-guide.txt

.. dropdown::  Developer Guide  
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    
    .. include:: /inference-legacy/nxd-core/developer-guide-inference.txt

.. dropdown::  Tutorials  
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    
    .. include:: /archive/libraries/neuronx-distributed/tutorials/neuronx_distributed_tutorials.txt


.. dropdown::  Misc  
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
    
    .. include:: /archive/libraries/neuronx-distributed/neuronx-distributed-misc.txt
