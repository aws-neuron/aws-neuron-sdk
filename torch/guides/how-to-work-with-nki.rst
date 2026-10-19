.. meta::
   :description: How to add NKI kernels to PyTorch code with torch-neuronx on AWS Trainium. Call a kernel on Neuron tensors, register it as a PyTorch custom operator with nki_op, declare in-place updates, add a backward pass, and run it inside torch.compile.
   :keywords: TorchNeuron, torch-neuronx, NKI, nki.jit, nki_op, custom op, mutates_args, register_autograd, torch.compile, AWS Neuron, Trainium
   :date-modified: 2026-10-15

.. _torchneuron-how-to-work-with-nki:

====================================================
How to work with NKI kernels from torch-neuronx
====================================================

Task overview
-------------

This topic shows how to add Neuron Kernel Interface (NKI) kernels to PyTorch code with
``torch-neuronx``. You call a kernel on Neuron tensors, register it as a PyTorch custom operator,
and use the operator in eager mode, inside ``torch.compile``, and in training. This page covers
how kernels integrate with PyTorch. To learn how to write kernels, see the
:ref:`NKI documentation <neuron-nki>`.

Before you write a kernel
^^^^^^^^^^^^^^^^^^^^^^^^^

Try eager mode and ``torch.compile`` first. Before writing a kernel, check whether
`aws-neuron/nki-library <https://github.com/aws-neuron/nki-library>`__ already has one; it
provides kernels for common ML operations on Trainium. Write your own kernel in two cases:

- A profile from :ref:`Neuron Explorer <neuron-explorer-home>` shows that one operator, or a few
  operators in sequence, limits step time, and ``torch.compile`` does not speed it up enough.
- Standard PyTorch operators cannot express the operation efficiently.

.. _torchneuron-nki-how-kernels-run:

How NKI kernels run in torch-neuronx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You write a kernel as a Python function decorated with ``@nki.jit``. When you call it with tensors
on a Neuron device, ``torch-neuronx`` handles the call:

- **Eager mode:** The first call compiles the kernel for the input shapes, data types, and
  non-tensor argument values of that call. Later calls with the same inputs reuse the compiled
  kernel.
- **Inside** ``torch.compile(backend="neuron")``: The kernel call becomes one node in the captured
  graph and compiles together with the surrounding PyTorch operations, so it does not cause a
  graph break.
- **Caching across processes:** ``torch-neuronx`` turns on the NKI kernel cache, so a new process
  reuses kernels that an earlier process compiled.

There are two layers:

- **The kernel** (``@nki.jit``) is the computation. Call it directly, in eager mode or inside
  ``torch.compile``. A kernel that writes into an input updates it in either mode. Autograd
  does not see a direct call: its output has no gradient.
- **The operator** (``@nki_op``) wraps the kernel call in a PyTorch custom operator. The operator
  has a name in ``torch.ops`` and a schema that declares its inputs, outputs, and in-place
  updates, and you can register a backward pass on it.

Call the kernel directly for forward-only code (step 1). Wrap it in ``nki_op`` to train through
it or to call it as a PyTorch operator (steps 2 to 4).

Prerequisites
-------------

- **torch-neuronx installed and verified:** A Trainium instance with a working ``torch-neuronx``
  installation. See :ref:`Quickstart: Install and verify torch-neuronx
  <torchneuron-quickstart-install-verify>`.
- **NKI installed:** The ``nki`` Python package in the same Python environment. See
  :ref:`Set up your NKI environment <how-to-set-up-nki-env>`.
- **Basic NKI knowledge:** You can read a simple NKI kernel. See the
  :ref:`NKI language guide <nki-language-guide>`.


Instructions
------------

The code on this page forms one script. Run the blocks in order.

**1:** Write a kernel and call it from PyTorch

A kernel that you call from PyTorch follows these conventions:

- Each tensor argument arrives in device memory (HBM).
- The kernel allocates each output in HBM with ``buffer=nl.shared_hbm`` and returns it. Each
  returned tensor becomes a PyTorch tensor.
- Non-tensor arguments, such as a Python ``float``, are constants in the compiled kernel. Each
  new value compiles the kernel again.

.. code-block:: python

   import nki
   import nki.isa as nisa
   import nki.language as nl
   import torch


   @nki.jit
   def add_kernel(a, b):
       # Copy both inputs from HBM into on-chip memory (SBUF).
       a_sbuf = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
       nisa.dma_copy(dst=a_sbuf, src=a)
       b_sbuf = nl.ndarray(b.shape, dtype=b.dtype, buffer=nl.sbuf)
       nisa.dma_copy(dst=b_sbuf, src=b)

       # Add the two tiles element by element.
       c_sbuf = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
       nisa.tensor_tensor(dst=c_sbuf, data1=a_sbuf, data2=b_sbuf, op=nl.add)

       # Allocate the output in HBM, copy the result to it, and return it.
       out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
       nisa.dma_copy(dst=out, src=c_sbuf)
       return out


   a = torch.rand(128, 128, device="neuron")
   b = torch.rand(128, 128, device="neuron")

   c = add_kernel(a, b)
   torch.testing.assert_close(c, a + b)

.. note::

   - For the tiling requirements of NKI kernels, see :ref:`NKI tiling <nki-about-tiling>`.
   - Kernels use the logical NeuronCore (LNC) configuration set by ``NEURON_LOGICAL_NC_CONFIG``.
     Don't set the LNC in the call (``kernel[lnc]``); a value that doesn't match fails to compile.
     See :doc:`Logical NeuronCore configuration </about-neuron/arch/neuron-features/logical-neuroncore-config>`.

**2:** Register the kernel as a PyTorch operator with ``nki_op``

Write a small Python function that calls the kernel, and decorate it with ``@nki_op``. This
registers the function as a PyTorch operator.

- **Name:** Pass a name in the form ``namespace::name``, such as ``my_ops::nki_add``. Pick any
  namespace for your project.
- **Type annotations:** Annotate every parameter and the return value. ``nki_op`` uses them to
  describe the operator's inputs and outputs to PyTorch.
- **In-place updates:** Use ``mutates_args={}`` for a kernel that doesn't write into its inputs.
  Step 3 covers kernels that do.

.. code-block:: python

   from torch_neuronx import nki_op


   @nki_op("my_ops::nki_add", mutates_args={})
   def nki_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
       return add_kernel(a, b)


   c = nki_add(a, b)
   torch.testing.assert_close(c, a + b)

   # The operator is also available in the torch.ops namespace.
   c = torch.ops.my_ops.nki_add(a, b)

**3:** Declare the inputs that the kernel updates in place

When a kernel writes into one of its input tensors, list that argument in ``mutates_args``.
This declares the update in the operator schema, as
`torch.library.custom_op <https://docs.pytorch.org/docs/stable/library.html#torch.library.custom_op>`__
requires. The following kernel writes its result into the input ``y``:

.. code-block:: python

   @nki.jit
   def add_into_kernel(x1, x2, y):
       x1_sbuf = nl.ndarray(x1.shape, dtype=x1.dtype, buffer=nl.sbuf)
       nisa.dma_copy(dst=x1_sbuf, src=x1)
       x2_sbuf = nl.ndarray(x2.shape, dtype=x2.dtype, buffer=nl.sbuf)
       nisa.dma_copy(dst=x2_sbuf, src=x2)

       y_sbuf = nl.ndarray(y.shape, dtype=y.dtype, buffer=nl.sbuf)
       nisa.tensor_tensor(dst=y_sbuf, data1=x1_sbuf, data2=x2_sbuf, op=nl.add)

       # Write the result into the caller's tensor y.
       nisa.dma_copy(dst=y, src=y_sbuf)
       return y


   @nki_op("my_ops::nki_add_into", mutates_args={"y"})
   def nki_add_into(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor) -> None:
       add_into_kernel(x1, x2, y)


   y = torch.zeros(128, 128, device="neuron")
   nki_add_into(a, b, y)
   torch.testing.assert_close(y, a + b)

**4:** Add a backward pass for training

Register a backward function on the operator with ``register_autograd``. The backward function
receives a context object and the gradient of the output, and it returns one gradient for each
input. It can call other operators, including another ``nki_op`` kernel.

.. code-block:: python

   @nki_add.register_autograd
   def nki_add_backward(ctx, grad_output):
       # For c = a + b, the gradient with respect to both a and b is grad_output.
       return grad_output, grad_output


   a_train = torch.rand(128, 128, device="neuron", requires_grad=True)
   b_train = torch.rand(128, 128, device="neuron", requires_grad=True)

   nki_add(a_train, b_train).sum().backward()
   torch.testing.assert_close(a_train.grad, torch.ones_like(a_train))
   torch.testing.assert_close(b_train.grad, torch.ones_like(b_train))

When the backward pass needs tensors from the forward pass, save them with a ``setup_context``
function. See
`torch.library.register_autograd <https://docs.pytorch.org/docs/stable/library.html#torch.library.register_autograd>`__.

**5:** Use the operator inside ``torch.compile``

Call the operator from a function or module that you compile with ``backend="neuron"``. A direct
kernel call works here too. Set ``fullgraph=True`` while you add a kernel. ``torch.compile`` then
raises an error if any part of the function can't be captured, instead of silently splitting the
function into several graphs. If compilation succeeds, the kernel is part of the compiled graph.
Gradients through the operator also work inside ``torch.compile``.

.. code-block:: python

   @torch.compile(backend="neuron", fullgraph=True)
   def scale_and_add(a, b):
       return nki_add(a * 2, b)


   torch.testing.assert_close(scale_and_add(a, b), a * 2 + b)

   # Gradients through the compiled function.
   a_train.grad = None
   b_train.grad = None
   scale_and_add(a_train, b_train).sum().backward()
   torch.testing.assert_close(a_train.grad, torch.full_like(a_train, 2.0))
   torch.testing.assert_close(b_train.grad, torch.ones_like(b_train))

Confirm your work
-----------------

To confirm that the integration is correct, run the operator inside an ``nn.Module`` in eager mode
and with ``torch.compile``, and compare both results with a PyTorch reference. Add this code to
the end of the script:

.. code-block:: python

   import torch.nn as nn


   class AddModule(nn.Module):
       def forward(self, a, b):
           return nki_add(a, b)


   model = AddModule().to("neuron")
   eager_out = model(a, b)
   compiled_out = torch.compile(model, backend="neuron", fullgraph=True)(a, b)

   torch.testing.assert_close(eager_out, a + b)
   torch.testing.assert_close(compiled_out, a + b)
   print("NKI operator output matches in eager mode and in torch.compile")

If both checks pass, the script prints:

.. code-block:: text

   NKI operator output matches in eager mode and in torch.compile

Common issues
-------------

These issues can occur when you add NKI kernels to PyTorch code.

.. rubric:: ``SBUF and PSUM tensors must have at least 2 dimensions``

- **Possible solution:** The kernel received a one-dimensional tensor. Pass two-dimensional
  tensors, for example ``torch.rand(128, 128, device="neuron")`` instead of
  ``torch.rand(128, device="neuron")``.

.. rubric:: ``ModuleNotFoundError: No module named 'torch_neuronx.pyhlo'``

- **Possible solution:** A kernel was called with tensors on the ``meta`` device. A direct
  ``@nki.jit`` call works only for tensors on a Neuron device. First, check whether the ``meta``
  tensors are intended. Most code passes only Neuron tensors, and then the fix is to move the
  inputs to the ``neuron`` device. If your code must also run on ``meta`` tensors, use one of these:

  - Wrap the kernel once with ``torch_neuronx.wrap_nki`` and call the result, for example
    ``add = wrap_nki(add_kernel)`` and then ``add(a, b)``. This builds the same wrapper that a
    ``@nki.jit`` call builds on Neuron tensors.
  - Pass a ``fake_fn`` to ``@nki_op``: a function that takes the operator's arguments and returns
    empty tensors shaped like its outputs. PyTorch uses it to get output shapes without running
    the kernel, for example for ``meta`` inputs and during ``torch.compile`` tracing. For
    ``nki_add``, whose output is shaped like ``a``, use ``fake_fn=lambda a, b: torch.empty_like(a)``.

.. rubric:: ``element 0 of tensors does not require grad and does not have a grad_fn``

- **Possible solution:** ``backward()`` reached the output of a direct kernel call, which is not
  connected to autograd. Wrap the kernel call in an ``@nki_op`` operator (step 2) and register a
  backward function (step 4).

.. rubric:: ``Trying to backward through my_ops.nki_add.default but no autograd formula was registered``

- **Possible solution:** ``backward()`` reached an ``@nki_op`` operator that has no backward pass.
  Register a backward function with ``register_autograd`` (step 4).

.. rubric:: ``@nki_op`` fails with a schema error

- **Possible solution:** ``nki_op`` builds the operator schema from type annotations. Annotate
  every parameter and the return value, for example ``a: torch.Tensor`` and ``-> torch.Tensor``.

.. rubric:: Each call is slow, or the kernel compiles again on every call

- **Possible solution:** The kernel compiles again for each new input shape, data type, or
  non-tensor argument value. Keep input shapes and non-tensor argument values fixed across calls.

.. rubric:: Results are stale after a change that is not in the kernel source or its arguments

- **Possible solution:** The NKI kernel compile cache persists across processes. It assumes the
  kernel depends only on its source and its declared arguments, so pass any value that changes
  what the kernel does, such as an environment variable, as a kernel argument. Code in an
  editable install or a vendored copy is not tracked. With ``NKI_TRACE_CACHE_VERBOSE=1``, a
  reused kernel logs ``HIT``. To turn off the cache, set ``NKI_ENABLE_TRACE_CACHE=0`` before you
  start Python. See :ref:`TorchNeuron environment variables <torchneuron-env-vars>`.

Related information
-------------------

- :ref:`Quickstart: Run an NKI kernel from torch-neuronx <torchneuron-quickstart-nki-kernel>`
- :ref:`About TorchNeuron <about-pytorch-neuron>`
- :ref:`torch.compile on Neuron <torch-compile-index>`
- :ref:`Graph breaks <torchneuron-graph-breaks>`
- :ref:`NKI documentation <neuron-nki>`
- :ref:`NKI API reference <nki_api_reference>`
- `aws-neuron/nki-library <https://github.com/aws-neuron/nki-library>`__: kernels for common ML
  operations
- `torch.library <https://docs.pytorch.org/docs/stable/library.html>`__: the PyTorch custom
  operator API that ``nki_op`` uses
