.. meta::
   :description: Quickstart for running an NKI kernel from torch-neuronx using nki_op. Minimal path to a running NKI kernel in a PyTorch model on AWS Trainium, in eager mode and with torch.compile.
   :keywords: TorchNeuron, torch-neuronx, NKI, nki_op, quickstart, AWS Neuron, Trainium, PyTorch
   :date-modified: 2026-10-13

.. _torchneuron-quickstart-nki-kernel:

====================================================
Quickstart: Run an NKI kernel from torch-neuronx
====================================================

This quickstart gets you from zero to a running NKI kernel called from a
PyTorch model on AWS Trainium using ``torch-neuronx``. You write a small NKI
kernel, register it as a PyTorch custom op with ``@nki_op``, and call it from
a model in eager mode and with ``torch.compile``. When you have completed it,
you will have a working NKI kernel that gives the same result in both modes.

**This quickstart is for**: TorchNeuron users who want to call a custom NKI
kernel from a PyTorch model on Neuron as fast as possible.

**Time to complete**: ~15 minutes


Prerequisites
-------------

- A Trn2 or Trn3 instance
- Neuron SDK 2.33 or later
- PyTorch 2.14 and ``torch-neuronx`` 2.14, on Python 3.12, 3.13, or 3.14
- ``torch-neuronx`` installed and verified. See
  :doc:`quickstart-torch-neuron`.
- Basic familiarity with PyTorch custom ops is helpful but not required.

The ``nki`` package is required. ``torch-neuronx`` imports it when it loads,
so this command checks both packages:

.. code-block:: bash

   python -c "import torch_neuronx, nki; print('torch-neuronx and NKI are ready')"

If the command prints ``torch-neuronx and NKI are ready``, you can start.
If it fails, see **Common issues** below.

Step 1: Write a minimal NKI kernel
-----------------------------------

In this step, you will write an NKI kernel that adds two tensors element by
element. This gives you a known-good starting point before you connect it to
PyTorch.

Create a file named ``nki_quickstart.py`` with this content:

.. code-block:: python

   import nki
   import nki.isa as nisa
   import nki.language as nl


   @nki.jit
   def add_kernel(a, b):
       # Copy the inputs from device memory (HBM) to on-chip memory (SBUF).
       a_sbuf = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
       nisa.dma_copy(dst=a_sbuf, src=a)
       b_sbuf = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
       nisa.dma_copy(dst=b_sbuf, src=b)

       # Add the two tiles on the NeuronCore.
       c_sbuf = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
       nisa.tensor_tensor(dst=c_sbuf, data1=a_sbuf, data2=b_sbuf, op=nl.add)

       # Allocate the output in HBM, copy the result to it, and return it.
       out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
       nisa.dma_copy(dst=out, src=c_sbuf)
       return out

The kernel allocates its own output tensor and returns it. It does not change
its inputs. This makes the PyTorch registration in Step 2 as simple as
possible.

This kernel processes each input as one tile. The first dimension of an SBUF
tile is the partition dimension, and its maximum size is 128
(``nl.tile_size.pmax``). In this quickstart, you use inputs of shape
``(128, 128)``. To process larger tensors, the kernel must loop over tiles.
See :doc:`/nki/index`.

Step 2: Wrap the kernel with ``@nki_op``
-----------------------------------------

In this step, you will register the NKI kernel as a PyTorch custom op so that
PyTorch can call it like any other op.

Add this code to the end of ``nki_quickstart.py``:

.. code-block:: python

   import torch
   from torch_neuronx import nki_op


   @nki_op("quickstart::add", mutates_args=())
   def nki_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
       return add_kernel(a, b)

The arguments of ``@nki_op`` are:

- ``"quickstart::add"``: The op name, in the form ``"{namespace}::{name}"``.
  The op is also available as ``torch.ops.quickstart.add``.
- ``mutates_args=()``: The names of the arguments that the kernel writes to.
  This kernel writes to no input, so the collection is empty.

PyTorch builds the op schema from the type annotations. Annotate every
argument and the return value.

.. note::

   If a kernel writes its result into an input tensor, add the name of that
   argument to ``mutates_args``. For example, use ``mutates_args={"out"}``.
   For in-place kernels and autograd, see
   :doc:`/torch/guides/how-to-work-with-nki`.

Step 3: Call the kernel from a small model
-------------------------------------------

In this step, you will call the registered op in the forward pass of a small
``nn.Module`` and run it on Neuron in eager mode.

Add this code to the end of ``nki_quickstart.py``:

.. code-block:: python

   class SmallModel(torch.nn.Module):
       def forward(self, a, b):
           c = a * 2
           d = b + 3
           e = nki_add(c, d)  # Runs the NKI kernel.
           return e * c


   def reference(a, b):
       c = a * 2
       d = b + 3
       return (c + d) * c


   a = torch.rand(128, 128)
   b = torch.rand(128, 128)
   expected = reference(a, b)

   model = SmallModel()
   eager_out = model(a.to("neuron"), b.to("neuron")).cpu()
   print("Eager output matches the reference:", torch.allclose(eager_out, expected, rtol=1e-5, atol=1e-5))

The model mixes standard PyTorch ops with the NKI op. The ``reference``
function computes the same result on the CPU with standard PyTorch ops only.

Run the script:

.. code-block:: bash

   python nki_quickstart.py

You should see this output:

.. code-block:: text

   Eager output matches the reference: True

Step 4: Compose with ``torch.compile``
--------------------------------------

In the final step, you will compile the same model with
``torch.compile(backend="neuron")`` and confirm that the NKI op gives the same
result.

Add this code to the end of ``nki_quickstart.py``:

.. code-block:: python

   compiled_model = torch.compile(model, backend="neuron", fullgraph=True, dynamic=False)
   compiled_out = compiled_model(a.to("neuron"), b.to("neuron")).cpu()
   print("Compiled output matches the reference:", torch.allclose(compiled_out, expected, rtol=1e-5, atol=1e-5))

``fullgraph=True`` makes ``torch.compile`` raise an error if the graph breaks.
``dynamic=False`` is the only supported value on Neuron.
A successful run confirms that the NKI op is captured inside the compiled
graph, together with the PyTorch ops around it.

Run the script again:

.. code-block:: bash

   python nki_quickstart.py

The first run of the compiled model is slower because the Neuron compiler
compiles the graph. Later runs with the same shapes use the compiled graph
from the cache.

Confirmation
------------

If the kernel is connected correctly, the script prints:

.. code-block:: text

   Eager output matches the reference: True
   Compiled output matches the reference: True

Both lines must show ``True``. This confirms that the NKI kernel runs as a
PyTorch op in eager mode and inside a ``torch.compile`` graph.

Congratulations! You have now called a custom NKI kernel from a PyTorch model
on AWS Trainium. If you encountered any issues, see the **Common issues**
section below.

Common issues
-------------

- **"No module named 'nki'"**: The ``nki`` package is not installed in the
  active Python environment. Install it from the Neuron pip repository:

  .. code-block:: bash

     pip install nki --extra-index-url https://pip.repos.neuron.amazonaws.com

- **Kernel compilation error that refers to the partition dimension**: The
  first dimension of the input is larger than 128. Use inputs with a first
  dimension of 128 or less, or add tiling to the kernel. See
  :doc:`/nki/index`.
- **Error when PyTorch infers the op schema**: An argument or the return
  value of the ``@nki_op`` function has no type annotation. Add a type
  annotation to each argument and to the return value.
- **Error that the op is already registered**: The same op name was
  registered two times in one Python process. This can occur when you run the
  same cell again in a notebook. Restart the Python process, or use a
  different op name.

Clean up
--------

This quickstart creates no persistent resources. The ``quickstart::add`` op
exists only while the Python process runs. Delete ``nki_quickstart.py`` if
you do not need it.

Next steps
----------

Now that you've completed this quickstart, see the day-to-day NKI
integration patterns and the NKI language docs.

- :doc:`/torch/guides/how-to-work-with-nki` — Day-to-day NKI integration
  patterns for torch-neuronx, including in-place kernels and autograd.
- :doc:`/nki/index` — NKI language and API documentation.

Further reading
---------------

- :doc:`/torch/get-started/quickstart-torch-neuron` — Install and verify
  torch-neuronx first.
- :doc:`/torch/get-started/quickstart-build-first-model` — Eager,
  ``torch.compile``, and mixed-mode quickstart without NKI.
