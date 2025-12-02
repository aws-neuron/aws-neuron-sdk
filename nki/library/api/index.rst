.. meta::
    :description: API reference for the pre-built NKI Library kernels included with the AWS Neuron SDK.
    :date-modified: 11/28/2025

.. _nkl_api_ref_home:

NKI Library Kernel API Reference
=================================

The NKI Library  provides pre-built reference kernels you can use directly in your model development with the AWS Neuron SDK and NKI. These kernel APIs provide the default classes, functions, and parameters you can use to integrate the NKI Library kernels into your models.

**Source code for these kernel APIs can be found at**: https://github.com/aws-neuron/nki-library

Normalization and Quantization Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`RMSNorm-Quant Kernel API Reference </nki/library/api/rmsnorm-quant>`
     - API reference for the RMSNorm-Quant kernel included in the NKI Library. The kernel performs optional RMS normalization followed by quantization to ``fp8``.

QKV Projection Kernels
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`QKV Kernel API Reference </nki/library/api/qkv>`
     - API reference for the QKV kernel included in the NKI Library. The kernel performs Query-Key-Value projection with optional normalization fusion.

Attention Kernels
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Attention CTE Kernel API Reference </nki/library/api/attention-cte>`
     - API reference for the Attention CTE kernel included in the NKI Library. The kernel implements attention specifically optimized for Context Encoding use cases.
   * - :doc:`Attention TKG Kernel API Reference </nki/library/api/attention-tkg>`
     - API reference for the Attention TKG kernel included in the NKI Library. The kernel implements attention specifically optimized for Token Generation (Decoding) use cases with small active sequence lengths.

Multi-Layer Perceptron (MLP) Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`MLP Kernel API Reference </nki/library/api/mlp>`
     - API reference for the MLP kernel included in the NKI Library. The kernel implements a Multi-Layer Perceptron with optional normalization fusion and various optimizations.

Output Projection Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`Output Projection CTE Kernel API Reference </nki/library/api/output-projection-cte>`
     - API reference for the Output Projection CTE kernel included in the NKI Library. The kernel computes the output projection operation optimized for Context Encoding use cases.
   * - :doc:`Output Projection TKG Kernel API Reference </nki/library/api/output-projection-tkg>`
     - API reference for the Output Projection TKG kernel included in the NKI Library. The kernel computes the output projection operation optimized for Token Generation use cases.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Normalization and Quantization Kernels

    RMSNorm-Quant <rmsnorm-quant>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: QKV Projection Kernels

    QKV <qkv>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Attention Kernels

    Attention CTE <attention-cte>
    Attention TKG <attention-tkg>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: MLP Kernels

    MLP <mlp>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Output Projection Kernels

    Output Projection CTE <output-projection-cte>
    Output Projection TKG <output-projection-tkg>
