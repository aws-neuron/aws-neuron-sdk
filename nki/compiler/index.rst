.. _nki_compiler_home:

.. meta::
    :description: Documentation home for the AWS Neuron SDK NKI Compiler.
    :date-modified: 12-02-2025

NKI Compiler Documentation
==========================

The NKI (Neuron Kernel Interface) Compiler is a key component of the AWS Neuron SDK that enables direct programming of NeuronDevices on AWS Trn/Inf instances. It compiles NKI kernel functions marked with ``nki.jit`` into optimized code that can run efficiently on NeuronCores. This section provides an overview of the NKI Compiler's architecture, its integration with the Neuron SDK, and best practices for developing high-performance kernels.


.. grid:: 1
    :margin: 3
    :gutter: 3

    .. grid-item-card:: About the NKI Compiler
        :link: /nki/compiler/about/index
        :link-type: doc
        
        NKI has its own compiler! Learn more about it here.

    .. grid-item-card:: How the NKI Compiler Works with the Neuron Graph Compiler
        :link: /nki/compiler/about/how-nki-works-with-compiler
        :link-type: doc
        
        Learn how the NKI compiler integrates with the :doc:`Neuron SDK graph compiler </compiler/index>` to optimize and execute kernel functions on NeuronDevices.

    .. grid-item-card:: Migrate NKI Beta 1 Kernels to the New NKI Compiler
        :link: migrate-nki-kernels-beta2
        :link-type: ref
        
        Learn how to migrate AWS Neuron NKI Beta 1 kernels to the new NKI compiler in Beta 2.

.. toctree::
   :maxdepth: 1
   :hidden:

   About the NKI Compiler <about/index>
   Graph Compiler Integration <about/how-nki-works-with-compiler>