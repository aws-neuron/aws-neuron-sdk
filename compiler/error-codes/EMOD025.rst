.. _error-code-emod025:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EMOD025.

NCC_EMOD025
===========

**Error message**: Dynamic shape is not supported: instruction '<name>' has shape '<shape>'. The Neuron compiler requires all tensor dimensions to be statically sized.

The Neuron compiler requires fully static shapes. This error occurs when the input model contains a tensor whose shape has one or more dynamically-sized dimensions. This includes both unbounded and bounded dynamic dimensions.

Erroneous code example:

.. code-block:: python

    import torch

    class Model(torch.nn.Module):
        def forward(self, x):
            return x + x

    model = Model()

    # A dynamic batch dimension produces a dynamically-sized shape, which the
    # Neuron compiler cannot compile.
    compiled = torch.compile(model, backend="neuronx", dynamic=True)

To fix this error, recompile the model with fully static input shapes. When compiling via PyTorch ``torch.compile``, set ``dynamic=False`` to disable dynamic shape specialization:

.. code-block:: python

    # Disable dynamic shape specialization so all tensor dimensions are static.
    compiled = torch.compile(model, backend="neuronx", dynamic=False)

If your model inherently varies an input dimension (for example a variable sequence length), pad or bucket the inputs to a fixed size before compilation so that every tensor dimension is statically known.
