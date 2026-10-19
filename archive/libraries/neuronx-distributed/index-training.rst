.. meta::
    :description: Home page for the NxD Core for Training (NxDT) library included with the Neuron SDK.
    :date-modified: 12/02/2025

.. _neuronx-distributed-training-index:
.. _neuronx-distributed-index:


NxD Core for Training
=======================

.. note:: Maintenance status

   The NeuronX Distributed Training (NxD Training) library documented on this page is in
   **maintenance status** as of Neuron SDK 2.32 — it continues to receive critical bug fixes
   and security patches, and this documentation remains available. It is **not**
   end-of-support. For new PyTorch training work on Trainium, use :ref:`TorchNeuron, the native
   PyTorch backend <torch-neuron-home>`; the task-oriented :ref:`Training on Neuron
   <training-home>` page shows where it fits.

NeuronX Distributed Core (NxD Core) is a package for supporting different distributed training mechanisms for Neuron devices. It provides XLA-friendly implementations of some of the more popular distributed
training techniques. As the size of the model scales, fitting these models on a single device becomes impossible and hence we have to make use of model sharding techniques to partition the model across multiple devices. 


About NeuronX-Distributed (NxD) for Training
---------------------------------------------

NeuronX Distributed (NxD Core) provides fundamental building blocks that enable you to run advanced inference workloads on AWS Inferentia and Trainium instances. These building blocks include parallel linear layers that enable distributed inference, a model builder that compiles PyTorch modules into Neuron models, and more.

The NeuronX Distributed Training (NxD Training) library is a collection of open-source tools and libraries designed to empower customers to train PyTorch models on AWS Trainium instances. It combines both ease-of-use and access to features built on top of
NxD Core library. Except for a few Trainium-specific features, NxD Training is compatible with training platforms like NVIDIA's NeMo.

.. toctree::
    :maxdepth: 1
    :hidden:

    Setup </archive/libraries/neuronx-distributed/setup/index>
    App Notes </archive/libraries/neuronx-distributed/app_notes>
    API Reference Guide </archive/libraries/neuronx-distributed/api-reference-guide>
    Developer Guide  </archive/libraries/neuronx-distributed/developer-guide-training>
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
    
    .. include:: /archive/libraries/neuronx-distributed/developer-guide-training.txt

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
