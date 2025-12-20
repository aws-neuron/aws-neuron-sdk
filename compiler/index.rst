.. _neuron_cc:

Neuron Graph Compiler
======================

The Neuron Graph Compiler is a sophisticated compilation system that transforms Machine Learning models from various frameworks (TensorFlow, MXNet, PyTorch, XLA HLO) into highly optimized code for AWS Neuron accelerators. It performs deep analysis of model structure, applies hardware-specific optimizations, and generates executable code tailored for maximum performance on Neuron hardware.

The Neuron compiler is available in two versions to support different AWS ML accelerator architectures:
 
* **neuronx-cc**: The newer XLA-based compiler supporting NeuronCores v2 architecture (Trn1, Inf2, Trn1n, Trn2). This compiler leverages the XLA (Accelerated Linear Algebra) framework to provide advanced optimizations for modern ML workloads.
* **neuron-cc**: The TVM-based compiler supporting NeuronCores v1 architecture (Inf1). This compiler uses the TVM (Tensor Virtual Machine) framework as its foundation.

Key capabilities of the Neuron Graph Compiler include:

* **Performance optimization**: Intelligently converts FP32 operations to more efficient formats (BF16/FP16/TF32/FP8) with configurable precision-performance tradeoffs. By default, the compiler automatically casts FP32 matrix multiplication operations to BF16 for optimal performance while maintaining accuracy.

* **Model-specific optimizations**: Provides specialized optimizations for different model architectures:
  * **Generic**: Applies general optimizations suitable for all model types
  * **Transformer**: Implements specific optimizations for transformer-based architectures like BERT, GPT, and other attention-based models
  * **U-Net**: Applies specialized memory optimizations for U-Net architectures to prevent performance-impacting data transfers

* **Distributed training support**: Enables efficient large language model (LLM) training through distribution strategies that shard parameters, gradients, and optimizer states across data-parallel workers.

* **Advanced memory management**: Optimizes memory usage for large models through techniques like model sharding across multiple NeuronCores, with configurable logical NeuronCore settings to control sharding degree.

* **Optimization levels**: Provides multiple optimization levels (1-3) to balance compilation time against runtime performance, allowing users to choose the appropriate tradeoff for their workflow.

* **Mixed precision support**: Offers fine-grained control over precision and performance through auto-casting options, supporting multiple numeric formats (FP32, TF32, FP16, BF16, FP8) with different strengths in dynamic range and numeric precision.

The compilation process is typically transparent to users, as the compiler is invoked automatically within ML frameworks through Neuron Framework plugins. Models are analyzed, optimized, and compiled into a NEFF file (Neuron Executable File Format), which is then loaded by the :doc:`Neuron runtime </neuron-runtime/index>` for execution on Neuron devices.

.. toctree::
    :maxdepth: 1
    :hidden:

    /compiler/neuronx-cc
    /compiler/neuron-cc
    Error codes </compiler/error-codes/index>

.. tab-set::

   .. tab-item:: Neuron Graph Compiler for Trn1 & Inf2

         .. dropdown::  API Reference Guide
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in

               * :ref:`Neuron Compiler CLI Reference Guide <neuron-compiler-cli-reference-guide>`

         .. dropdown:: Graph Compiler Developer Guide
                  :class-title: sphinx-design-class-title-med
                  :class-body: sphinx-design-class-body-small
                  :animate: fade-in

                  * :ref:`neuronx-cc-training-mixed-precision`
  
         .. dropdown:: Graph Compiler Error Code Reference
                  :class-title: sphinx-design-class-title-med
                  :class-body: sphinx-design-class-body-small
                  :animate: fade-in

                  * :ref:`ncc-errors-home`

         .. dropdown::  Misc
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in
               :open:

               * :ref:`FAQ <neuronx_compiler_faq>`
               * :ref:`What's New <neuronx-cc-rn>`

   .. tab-item:: Neuron Graph Compiler for Inf1


         .. dropdown:: Graph Compiler API Reference Guide
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in

               * :ref:`neuron-compiler-cli-reference`


         .. dropdown:: Graph Compiler Developer Guide
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in


               * :ref:`neuron-cc-training-mixed-precision`



         .. dropdown::  Misc
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in
               :open:

               * :ref:`FAQ <neuron_compiler_faq>`
               * :ref:`What's New <neuron-cc-rn>`
               * :ref:`neuron-supported-operators`
