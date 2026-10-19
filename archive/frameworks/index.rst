.. meta::
   :description: ML Framework support on AWS Neuron SDK - PyTorch and JAX integration for high-performance machine learning on AWS Inferentia and Trainium.
   :date-modified: 2026-03-12
   :keywords: AWS Neuron, machine learning

.. _frameworks-neuron-sdk:

ML framework support on AWS Neuron SDK
=======================================

AWS Neuron provides integration with popular machine learning frameworks, enabling you to accelerate your existing models on AWS Inferentia and Trainium with minimal code changes. Choose from our comprehensive framework support to optimize your inference and training workloads.

Frameworks
-----------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: PyTorch on AWS Neuron
        :link: torch/index
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Complete PyTorch integration for both inference and training on all Neuron hardware.

        * **TorchNeuron Native** - Native PyTorch backend with eager execution and ``torch.compile``
        * **PyTorch NeuronX (torch-neuronx)** - ``Inf2``, ``Trn1``, ``Trn2`` (inference & training)
        * See: :doc:`/torch/index`

    .. grid-item-card:: JAX on AWS Neuron
        :link: /jax/index
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        **Beta release**

        Experimental JAX support with Neuron Kernel Interface (NKI) integration.

        * **JAX NeuronX** - Neuron hardware support
        * Research and development focus
        * **Status**: Beta - active

.. note::

   Looking for TensorFlow, MXNet, or torch-neuron (Inf1) documentation? These frameworks
   have been archived. See :doc:`/archive/index` for legacy framework documentation.

Hardware compatibility matrix
-----------------------------

.. list-table::
   :header-rows: 1
   :class: compatibility-matrix

   * - Framework
     - Inf2
     - Trn1/Trn1n
     - Trn2
     - Inference
     - Training
   * - **torch-neuronx**
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **JAX NeuronX**
     - ✅
     - ✅
     - N/A
     - ✅
     - N/A
