.. meta::
    :description: NKI Library specifications for the pre-built kernels included with the AWS Neuron SDK.
    :date-modified: 12/02/2025

.. _nkl_design_spec_home:

NKI Library Design Specifications
==================================

The NKI Library provides pre-built kernels you can review and modify in your own kernel development with the AWS Neuron SDK and NKI. In this section, learn how the NKI Library kernels are designed and optimized so you can apply the same techniques to your own custom NKI kernels.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Kernel
     - Description
     - Source Code
   * - :doc:`RMSNorm-Quant kernel specification </nki/library/specs/design-rmsnorm-quant>`
     - Performs optional RMS normalization followed by quantization to ``fp8``.
     - `Source code <https://github.com/aws-neuron/nki-samples/tree/main/src/nki_samples/reference/rmsnorm_quant>`_

.. toctree::
    :maxdepth: 1
    :hidden:

    RMSNorm-Quant <design-rmsnorm-quant>