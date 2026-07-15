.. _error-code-evrf051:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF051.

NCC_EVRF051
===========

**Error message**: Data type F8E4M3FN is not supported on TRN1/TRN2.

The F8E4M3FN (8-bit floating point with 4-bit exponent and 3-bit mantissa) data
type is natively supported on Trainium3 (Trn3) and later hardware generations.
The compiler raises this error for an F8E4M3FN operand targeting Trn1 or Trn2
unless the experimental F8E4M3 conversion flag is enabled.

For ``QuantizeMX``, the compiler also raises this error when
``backend_config`` selects ``"float8_e4m3fn"`` and the conversion flag is
enabled, because MXFP operations do not support conversion to F8E4M3.

To fix this error for ``QuantizeMX``, target Trn3 or later and do not enable
``--experimental-unsafe-fp8e4m3fn-as-fp8e4m3``. For other operations targeting
Trn1 or Trn2, the flag can be used to convert F8E4M3FN operands to F8E4M3.

* More information on supported data types: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html
