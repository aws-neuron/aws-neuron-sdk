.. _nki_deep-dives_home:

.. meta::
    :description: Documentation home for the AWS Neuron SDK NKI Deep Dives and other advanced materials.
    :keywords: NKI, AWS Neuron, Deep Dives, Advanced Programming
    :date-modified: 12/01/2025

NKI Deep Dives
==============

This section provides in-depth technical documentation and guides for advanced users of the Neuron Kernel Interface (NKI). These deep dives offer detailed explanations of NKI concepts, programming patterns, and best practices to help you maximize the performance and capabilities of your NKI code on AWS Neuron devices.

Advanced NKI Programming
-------------------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: NKI Language Guide (Beta 2)
      :link: nki-language-guide
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Comprehensive guide to the NKI language, including syntax, tensor operations, control flow, and class support for Trainium devices.

   .. grid-item-card:: NKI Programming Model (Legacy)
      :link: programming_model
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Core concepts of NKI programming including memory hierarchy, tiling, and execution model.

..
   .. grid-item-card:: MXFP8/4 Matrix Multiplication Guide
      :link: mxfp8-matmul 
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Guide to performing matrix multiplication using MXFP8 data types in NKI kernels, including data layout, quantization, and tiling strategies.


.. toctree::
    :maxdepth: 1
    :hidden:

    nki-language-guide
    programming_model

..
    MXFP8/4 Matrix Multiplication <mxfp8-matmul>
