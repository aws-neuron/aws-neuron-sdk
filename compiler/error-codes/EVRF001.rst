.. _error-code-evrf001:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF001.

NCC_EVRF001
===========

**Error message**: An unsupported operator was used.

Try using alternative operators from the full list of supported operators via `neuronx-cc list-operators --framework XLA` to workaround the limitation.

Before:

.. code-block:: python

    class Model(torch.nn.Module):
        def forward(self, A, b):
            return torch.triangular_solve(b, A)


Possible workaround:

.. code-block:: python

    class Model(torch.nn.Module):
        def forward(self, A, b):
            # Although slower than triangular_solve, this is mathematically equivalent
            A_inv = torch.inverse(A)
            return A_inv @ b
