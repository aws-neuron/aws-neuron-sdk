.. _error-code-evrf005:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF005.

NCC_EVRF005
===========

**Error message**: The compiler found usage of F8E4M3FNUZ, F8E4M3B11FNUZ, or F8E5M2FNUZ data type which is not supported.

Erroneous code example:

.. code-block:: python

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 10)
        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x
    input_tensor = torch.randn(1, 10).to(torch.float8_e4m3fnuz)

To fix this error:

.. code-block:: python
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.linear2 = nn.Linear(20, 10)
        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x
    input_tensor = torch.randn(1, 10).to(torch.float8_e4m3fnuz)
    # Convert to a supported type
    input_tensor = input_tensor.to(torch.float16)

* More information on supported data types: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html
