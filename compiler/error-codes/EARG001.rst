.. _error-code-earg001:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EARG001.

NCC_EARG001
===========

**Error message**: This error occurs when you attempt to use a Logical Neuron Core (LNC) configuration that is not supported by the target Neuron architecture.

For example, a trn1 instance running the following code will run into this error:

.. code-block:: python

   traced_model = torch_neuronx.trace(
      model,
      input,
      compiler_args=['--lnc', '2']  # ERROR: lnc=2 not supported on trn1
   )

On trn1, only lnc=1 is supported.

Physical Neuron Core:

- Actual hardware compute unit on the chip

- Has dedicated compute resources, memory, etc.

Logical Neuron Core:

- Software abstraction grouping multiple physical cores

- Controlled via the NEURON_LOGICAL_NC_CONFIG environment variable or the --lnc flag (when using neuronx-cc directly)

For more information: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/device-memory.html#logical-neuron-cores
