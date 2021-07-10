Troubleshooting Guide for Torch-Neuron
======================================

General Torch-Neuron issues
---------------------------

If you see an error about "Unknown builtin op: neuron::forward_1" like below, please ensure that import line "import torch_neuron" (to register the Neuron custom operation) is in the inference script before using torch.jit.load.

::

   Unknown builtin op: neuron::forward_1.
   Could not find any similar ops to neuron::forward_1. This op may not exist or may not be currently supported in TorchScript.


TorchVision related issues
--------------------------

If you encounter an error like below, it is because latest torchvision
version >= 0.7 is not compatible with Torch-Neuron 1.5.1. Please
downgrade torchvision to version 0.6.1:

::

   E   AttributeError: module 'torch.jit' has no attribute '_script_if_tracing'                                                                                      


torch.jit.trace issues
----------------------
The :ref:`/neuron-guide/neuron-frameworks/pytorch-neuron/api-compilation-python-api.rst`
uses the :code:`torch.jit.trace` function in PyTorch to generate :code:`ScriptFunction`
models for execution on Inferentia. Due to this, your exisiting PyTorch model must be
torch-jit-traceable.

Please visit https://pytorch.org/docs/stable/generated/torch.jit.trace.html
to review the properties that a model must have to be torch-jit-traceable.
The PyTorch-Neuron trace API accepts :code:`torch.jit.trace` :code:`**kwargs`,
such as :code:`strict=False`.


Compiling models with outputs that are not torch-traceable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Models that have non torch-traceable outputs can be "wrapped" to enable
compilation for Inferentia. This process typically involves writing a wrapper
that converts the model's output into a form that is torch-traceable, compiling
the wrapped model for Inferentia using :code:`torch.neuron.trace()`, and
then writing a second wrapper that converts the model's output back into the original
form. The following is an example of wrapping a model with non-torch-traceable outputs
to compile it for Inferentia:

.. code-block:: python

   import torch
   import torch_neuron
   import torch.nn as nn


   class NonTraceableModel(nn.Module):
      def __init__(self):
         super(NonTraceableModel, self).__init__()
         self.conv = nn.Conv2d(1, 1, 3)

      def forward(self, x):
         a = self.conv(x) + 1
         b = self.conv(x) + 2
         c = self.conv(x) + 3
         # An output that is a list of tuples and tensors is not torch-traceable
         return [(a, b), c]


   class TraceableModel(nn.Module):
      def __init__(self, non_traceable_model):
         super(TraceableModel, self).__init__()
         self.non_traceable_model = non_traceable_model

      def forward(self, x):
         out = self.non_traceable_model(x)
         # An output that is a tuple of tuples and tensors is torch-traceable
         return tuple(out)


   class NeuronModel(nn.Module):
      def __init__(self, model):
         super(NeuronModel, self).__init__()
         self.traceable_model = TraceableModel(model)

      def forward(self, x):
         out = self.traceable_model(x)
         # Return the output in the original format
         return list(out)


   non_traceable_model = NonTraceableModel()

   model = NeuronModel(non_traceable_model)
   model.eval()

   inputs = torch.rand(1, 1, 3, 3)

   # Compile the traceable wrapped model
   model.traceable_model = torch.neuron.trace(model.traceable_model, inputs)
