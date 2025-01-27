.. _pytorch-neuron-inference-troubleshooting:

Troubleshooting Guide for PyTorch Neuron (``torch-neuron``)
===========================================================

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


2GB protobuf limit related issues
---------------------------------

If you encounter an error like below, it is because the model size is larger than 2GB.
To compile such large models, use the :ref:`separate_weights=True <torch_neuron_trace_api>` flag. Note,
ensure that you have the latest version of compiler installed to support this flag.
You can upgrade neuron-cc using 
:code:`python3 -m pip install neuron-cc[tensorflow] -U --force --index-url=https://pip.repos.neuron.amazonaws.com`

::

   E google.protobuf.message.DecodeError: Error parsing message with type 'tensorflow.GraphDef'




torch.jit.trace issues
----------------------
The :ref:`/neuron-guide/neuron-frameworks/pytorch-neuron/api-compilation-python-api.rst`
uses the PyTorch :func:`torch.jit.trace` function to generate
:class:`~torch.jit.ScriptModule` models for execution on Inferentia. Due to that,
to execute your PyTorch model on Inferentia it must be torch-jit-traceable,
otherwise you need to make sure your model is torch-jit-traceable. You can try
modifying your underlying PyTorch model code to make it traceable. If it's not
possible to change your model code, you can :ref:`write a wrapper around your
model <wrapping-non-traceable-models>` that makes it torch-jit-traceable to
compile it for Inferentia.

Please visit :func:`torch.jit.trace` to review the properties that a model must
have to be torch-jit-traceable. The PyTorch-Neuron trace API
:func:`torch_neuron.trace` accepts :code:`**kwargs` for :func:`torch.jit.trace`.
For example, you can use the :code:`strict=False` flag to
:ref:`compile models with dictionary outputs <compiling-models-with-kwargs>`.


.. _wrapping-non-traceable-models:

Compiling models with outputs that are not torch-jit-traceable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To enable compilation of models with non torch-jit-traceable outputs, you can
use a technique that involves writing a wrapper that converts the model's
output into a form that is torch-jit-traceable. You can then compile the
wrapped model for Inferentia using :func:`torch_neuron.trace`.


The following example uses a wrapper to compile a model with non
torch-jit-traceable outputs. This model cannot be compiled for Inferentia in
its current form because it outputs a list of tuples and tensors, which is not
torch-jit-traceable.

.. code-block:: python

    import torch
    import torch_neuron
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(1, 1, 3)

        def forward(self, x):
            a = self.conv(x) + 1
            b = self.conv(x) + 2
            c = self.conv(x) + 3
            # An output that is a list of tuples and tensors is not torch-traceable
            return [(a, b), c]

    model = Model()
    model.eval()

    inputs = torch.rand(1, 1, 3, 3)

    # Try to compile the model
    model_neuron = torch.neuron.trace(model, inputs) # ERROR: This cannot be traced, we must change the output format


To compile this model for Inferentia, we can write a wrapper around the model
to convert its outputs into a tuple of tensors, which is torch-jit-traceable.

.. code-block:: python

    class NeuronCompatibilityWrapper(nn.Module):
        def __init__(self):
            super(NeuronCompatibilityWrapper, self).__init__()
            self.model = Model()

        def forward(self, x):
            out = self.model(x)
            # An output that is a tuple of tuples and tensors is torch-jit-traceable
            return tuple(out)

Now, we can successfully compile the model for Inferentia using the
:code:`NeuronCompatibilityWrapper` wrapper as follows:

.. code-block:: python

    model = NeuronCompatibilityWrapper()
    model.eval()

    # Compile the traceable wrapped model
    model_neuron = torch.neuron.trace(model, inputs)

If the model's outputs must be in the original form, a second wrapper can be
used to transform the outputs after compilation for Inferentia. The following
example uses the :code:`OutputFormatWrapper` wrapper to convert the compiled
model's output back into the original form of a list of tuples and tensors.

.. code-block:: python

    class OutputFormatWrapper(nn.Module):
        def __init__(self):
            super(OutputFormatWrapper, self).__init__()
            self.traceable_model = NeuronCompatibilityWrapper()

        def forward(self, x):
            out = self.traceable_model(x)
            # Return the output in the original format of Model()
            return list(out)

    model = OutputFormatWrapper()
    model.eval()

    # Compile the traceable wrapped model
    model.traceable_model = torch.neuron.trace(model.traceable_model, inputs)


Compiling a submodule in a model that is not torch-jit-traceable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example shows how to compile a submodule that is part of a non
torch-jit-traceable model. In this example, the top-level model :code:`Outer`
uses a dynamic flag, which is not torch-jit-traceable. However, the
submodule :code:`Inner` is torch-jit-traceable and can be compiled for
Inferentia.

.. code-block:: python

    import torch
    import torch_neuron
    import torch.nn as nn

    class Inner(nn.Module) :
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3)

        def forward(self, x):
            return self.conv(x) + 1


    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

        def forward(self, x, add_offset: bool = False):
            base = self.inner(x)
            if add_offset:
                return base + 1
            return base

    model = Outer()
    inputs = torch.rand(1, 1, 3, 3)

    # Compile the traceable wrapped submodule
    model.inner = torch.neuron.trace(model.inner, inputs)

    # TorchScript the model for serialization
    script = torch.jit.script(model)
    torch.jit.save(script, 'model.pt')

    loaded = torch.jit.load('model.pt')

Alternatively, for usage scenarios in which the model configuration is static
during inference, the dynamic flags can be hardcoded in a wrapper to make
the model torch-jit-traceable and enable compiling the entire model for Inferentia.
In this example, we assume the :code:`add_offset` flag is always
:code:`True` during inference, so we can hardcode this conditional path in the
:code:`Static` wrapper to remove the dynmaic behavior and compile the entire
model for Inferentia.

.. code-block:: python

    class Static(nn.Module):
        def __init__(self):
            super().__init__()
            self.outer = Outer()

        def forward(self, x):
            # hardcode `add_offset=True`
            output = self.outer(x, add_offset=True)
            return output

    model = Static()

    # We can now compile the entire model because `add_offset=True` is hardcoded in the Static wrapper
    model_neuron = torch.neuron.trace(model, inputs)
