.. meta::
   :description: ML Framework support on AWS Neuron SDK - PyTorch, TensorFlow, MXNet, and JAX integration for high-performance machine learning on AWS Inferentia and Trainium.
   :date-modified: 2025-10-03

.. _frameworks-neuron-sdk:

ML framework support on AWS Neuron SDK
=======================================

AWS Neuron provides integration with popular machine learning frameworks, enabling you to accelerate your existing models on AWS Inferentia and Trainium with minimal code changes. Choose from our comprehensive framework support to optimize your inference and training workloads.

.. grid:: 2 2 2 2
    :gutter: 3
    :class-container: framework-grid

    .. grid-item-card:: PyTorch on AWS Neuron
        :link: torch/index
        :link-type: doc
        :class-header: bg-primary text-white
        :class-body: framework-card-body

        **Supports PyTorch 2.8**
        
        Complete PyTorch integration for both inference and training on all Neuron hardware.
        
        * **PyTorch NeuronX** - ``Inf2``, ``Trn1``, ``Trn2`` (inference & training)
        * **PyTorch Neuron** - ``Inf1`` (inference only)
        * Native PyTorch API compatibility
        * Distributed training support
        * Advanced profiling and debugging tools
  
    .. grid-item-card:: JAX on AWS Neuron
        :link: jax/index
        :link-type: doc
        :class-header: bg-info text-white
        :class-body: framework-card-body

        **Beta release**
        
        Experimental JAX support with Neuron Kernel Interface (NKI) integration.
        
        * **JAX NeuronX** - Neuron hardware support
        * NKI JAX interface
        * Research and development focus
        * Custom kernel development
        * **Status**: Beta - active development

Hardware compatibility matrix
-----------------------------

.. list-table::
   :header-rows: 1
   :class: compatibility-matrix

   * - Framework
     - Inf1
     - Inf2
     - Trn1/Trn1n
     - Trn2
     - Inference
     - Training
   * - **PyTorch NeuronX**
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **PyTorch Neuron**
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
   * - **TensorFlow NeuronX**
     - ❌
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * - **TensorFlow Neuron**
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
   * - **MXNet Neuron**
     - ✅
     - ❌
     - ❌
     - ❌
     - ✅
     - ❌
   * - **JAX NeuronX**
     - ❌
     - ✅
     - ✅
     - ✅
     - ✅
     - 🚧
