.. meta::
   :description: Overview of the NKI Compiler, its integration with the Neuron SDK, and how it enables efficient kernel development for AWS Neuron hardware.
   :keywords: NKI Compiler, Neuron Kernel Interface, AWS Neuron SDK, kernel compilation, Trainium, Inferentia, machine learning acceleration

.. _nki_compiler_about:

======================
About the NKI Compiler
======================

This topic covers the NKI Compiler and how it applies to developing with the AWS Neuron SDK. The NKI compiler is responsible for compiling NKI kernels. The NKI compiler interacts with the Neuron graph compiler to produce a complete model.

Overview
----------

The NKI language allows kernel writers to have direct, fine grained control over Neuron devices. Through low level APIs that reflect the Neuron instruction set architecture (ISA), NKI empowers developers to take direct control over critical performance optimizations during kernel development. This approach requires a dedicated NKI compiler, separate from :doc:`the existing Neuron graph compiler </compiler/index>`, which compiles kernel code while preserving the developer's optimization choices. To seamlessly integrate NKI into model architectures defined in machine learning frameworks like JAX and PyTorch, the NKI compiler also works in conjunction with the Neuron Graph compiler.

The diagram below shows the detailed compilation flow inside the Neuron compilers and how they work together to build the overall binary that is executable on Neuron hardware. The NKI Compiler first parses the kernel code into an AST representation for semantic analysis. It then performs a small number of middle end and back end transformations on the AST, optimizing resource allocations and instruction scheduling, producing optimized NKI IR that gets integrated back into the overall model.

.. image:: /nki/img/compiler/nki-compiler-1.jpg

.. important::
    While the NKI language looks and feels like Python, it is not actually Python code. When the Python interpreter encounters a top level function decorated with ``@nki.jit``, it invokes the NKI compiler to handle compilation of that function.

.. code-block:: python
    
    # this is a Python function that calls 'kernel', which is a NKI kernel
    def a_function(x,y,z):
        kernel(x, y, z)

    # this is a NKI kernel that will be compiled by the NKI compiler and 
    # integrated back into the overall model by the Neuron Graph compiler
    @nki.jit
    def kernel(x,y,z):
        # this is kernel code


Using Python features within NKI kernels that are not supported will result in useful errors from the NKI compiler indicating that the feature is not a valid NKI feature. Neuron has intentionally constrained the NKI language to be as minimal as possible while serving the needs of building high performance kernels for today's popular models and will continue to grow and evolve the language over time. 

NKI Compiler Open Source
-------------------------

Neuron is planning on releasing the source code for the NKI compiler to increase awareness and transparency, to enable easier development of tools, and to invite participation and collaboration as we evolve the NKI language. Developers will be able to download the compiler sources, modify them, build the compiler, and use their locally built compiler in their overall model compilation flow. 

To do this, developers will be able to download our sources from our public git repository: https://github.com/aws-neuron/nki

The repo contains all the sources for the entire NKI compiler, as well as build instructions on how to produce a standalone nki.whl. Once built, developers can install their locally built wheel: ``pip install nki.whl``. This will replace the default NKI compiler that is installed with the Neuron SDK package. The local wheel will then be registered to handle subsequent ``@nki.jit`` decorators and will be picked up and integrated with the rest of the Neuron Graph compiler flow.

Note that upon installing a locally built wheel, developers must reinstall the Neuron SDK in order to revert their changes to the official version of the NKI compiler. Also, the officially built compiler will have an officially tagged version whereas locally built versions will not. Any bug and error reports will contain the version of the compiler used.


Understanding the NKI Compiler
--------------------------------

The following topics provide more details on the NKI Compiler and working with it in your kernel development.

.. grid:: 1
    :margin: 3

    .. grid-item-card:: How the NKI Compiler Works with the Neuron Graph Compiler
        :link: nki-compiler-integration
        :link-type: ref
        
        Learn how the NKI compiler integrates with the Neuron SDK graph compiler to optimize and execute kernel functions on NeuronDevices.

    .. grid-item-card:: Migrate NKI Beta 1 Kernels to the New NKI Compiler
        :link: migrate-nki-kernels-beta2
        :link-type: ref
        
        Learn how to migrate AWS Neuron NKI Beta 1 kernels to the new NKI compiler in Beta 2.
