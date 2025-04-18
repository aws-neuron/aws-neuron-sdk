.. _nki_framework_custom_op:

NKI Kernel as a Framework Custom Operator
===========================================

This document demonstrates how to insert a NKI kernel as a custom
operator into a PyTorch or JAX model using simple code examples.

Using NKI kernels
-------------------------------

To register a NKI kernel registration, you need to call a decorated
NKI function.

Let's examine a guiding example below where we
randomly initialize two inputs, add them together, and then
multiply the result by the two input tensors element-wise.
This effectively calculates: ``a * b * (a + b)``.

We define a common NKI kernel for addition. For more information on the kernel, see
:doc:`SPMD Tensor Addition <tutorials/spmd_tensor_addition>`.

.. nki_example:: examples/tensor_addition/spmd_tensor_addition_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_27

^^^^^^^
PyTorch
^^^^^^^

We can perform ``(a + b) * a * b`` using native PyTorch code.
::

   import torch
   from torch_xla.core import xla_model as xm

   device = xm.xla_device()

   a = torch.randn(256, 1024, dtype=torch.float32).to(device)
   b = torch.randn(256, 1024, dtype=torch.float32).to(device)
   c = a + b
   out = a * b * c

   print(out)

Now let's replace the tensor addition (``c = a + b``) with a NKI
kernel.
To do this we replace the ``+`` operator with a call to the NKI kernel
caller (``nki_tensor_add``), and everything else works as before.

.. nki_example:: examples/tensor_addition/spmd_tensor_addition_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_28

::

   device = xm.xla_device()
   a = torch.randn(256, 1024, dtype=torch.float32).to(device)
   b = torch.randn(256, 1024, dtype=torch.float32).to(device)
   c = nki_tensor_add(a, b) # calling a NKI kernel, instead of the built-in torch op
   out = a * b * c
   print(out)

To understand what happens under the hood when we compile the above
code, we can print HLO IR graph generated by XLA by setting the
``NEURON_FRAMEWORK_DEBUG`` environment variable. For example, you may add the
following lines to your code:

::

   import os
   os.environ['NEURON_FRAMEWORK_DEBUG'] = "1"

A ``.pbtxt`` file is then written in your run directory that has the
corresponding human-readable HLO IR.

Let's examine the XLA output of this example.
In line #5 we can identify that the tensor addition is now
mapped to an HLO ``custom-call`` instruction, with
``AwsNeuronCustomNativeKernel`` as ``custom_call_target``. The output of
that ``custom-call`` is then consumed by the next instruction in line
#6 as usual.

.. code-block::
   :linenos:

   ENTRY %SyncTensorsGraph.22 (p0.2: f32[256,1024], p1.2: f32[256,1024]) -> (f32[256,1024]) {
    %p1.2 = f32[256,1024]{1,0} parameter(1), frontend_attributes={neff_input_name="input1"}
    %p0.2 = f32[256,1024]{1,0} parameter(0), frontend_attributes={neff_input_name="input0"}
    %multiply = f32[256,1024]{1,0} multiply(f32[256,1024]{1,0} %p1.2, f32[256,1024]{1,0} %p0.2)
    %custom-call.2 = f32[256,1024]{1,0} custom-call(f32[256,1024]{1,0} %p1.2, f32[256,1024]{1,0} %p0.2), custom_call_target="AwsNeuronCustomNativeKernel", api_version=API_VERSION_UNSPECIFIED, backend_config="...")
    %multiply.1 = f32[256,1024]{1,0} multiply(f32[256,1024]{1,0} %multiply, f32[256,1024]{1,0} %custom-call.2)
    ROOT %tuple = (f32[256,1024]{1,0}) tuple(f32[256,1024]{1,0} %multiply.1), frontend_attributes={neff_output_names="output0"}
   }

The Neuron compiler replaces the above custom-call with
the corresponding NKI kernel implementation while optimizing the rest of the
compute graph as usual. At the end of the compilation process, a single
compiled binary NEFF file
is generated representing the entire graph
including the NKI kernel. For more information about NEFF files, see `Neuron Compiler <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/index.html>`__.

.. _nki_framework_custom_op_jax:

^^^
JAX
^^^

We can perform ``(a + b) * a * b`` using native JAX code.

::

   import jax
   import jax.numpy as jnp

   @jax.jit
   def jax_customop_tutorial(a, b):
      c = a + b
      out = a * b * c
      return out

   seed = jax.random.PRNGKey(0)
   seed_a, seed_b = jax.random.split(seed)
   a = jax.random.normal(seed_a, (256, 1024), dtype=jnp.float32)
   b = jax.random.normal(seed_b, (256, 1024), dtype=jnp.float32)

   print(jax_customop_tutorial(a, b))

Similar to the PyTorch example above, let's replace the tensor addition ``(c = a + b)`` with
the addition NKI kernel. To do this we replace the ``+`` operator with a call to the NKI kernel
caller (``nki_tensor_add``), and everything else works as before.

.. nki_example:: examples/tensor_addition/spmd_tensor_addition_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_28

::

   import jax
   import jax.numpy as jnp

   @jax.jit
   def jax_customop_tutorial(a, b):
      c = nki_tensor_add(a, b) # calling a NKI kernel, instead of the built-in jax op
      out = a * b * c
      return out

   seed = jax.random.PRNGKey(0)
   seed_a, seed_b = jax.random.split(seed)
   a = jax.random.normal(seed_a, (256, 1024), dtype=jnp.float32)
   b = jax.random.normal(seed_b, (256, 1024), dtype=jnp.float32)
   print(jax_customop_tutorial(a, b))


To understand what happens under the hood when we compile the above code,
we can print the HLO IR graph by adding the following snippet to your code:

::

   print(jax.jit(jax_customop_tutorial)
      .lower(a, b)
      .compile()
      .runtime_executable()
      .hlo_modules()[0].to_string()
   )

Let's examine the XLA output of this example.
In line #7 we can identify that the tensor addition is now
mapped to an HLO ``custom-call`` instruction, similar to PyTorch. The output of
that ``custom-call`` is then consumed by the next instruction in line
#8 as usual.

.. code-block::
   :linenos:

   HloModule jit_add, entry_computation_layout={(f32[256,1024]{1,0}, f32[256,1024]{1,0})->(f32[256,1024]{1,0})}, allow_spmd_sharding_propagation_to_output={true}

   ENTRY %main.11 (Arg_0.1: f32[256,1024], Arg_1.2: f32[256,1024]) -> (f32[256,1024]) {
    %Arg_0.1 = f32[256,1024]{1,0} parameter(0), sharding={replicated}
    %Arg_1.2 = f32[256,1024]{1,0} parameter(1), sharding={replicated}
    %multiply.0 = f32[256,1024]{1,0} multiply(f32[256,1024]{1,0} %Arg_0.1, f32[256,1024]{1,0} %Arg_1.2), metadata={op_name="jit(add)/jit(main)/jit(jax_customop_tutorial)/mul" source_file="/tmp/ipykernel_3935360/2333914945.py" source_line=61}
    %custom-call.0 = f32[256,1024]{1,0} custom-call(f32[256,1024]{1,0} %Arg_0.1, f32[256,1024]{1,0} %Arg_1.2), custom_call_target="AwsNeuronCustomNativeKernel", api_version=API_VERSION_STATUS_RETURNING, metadata={op_name="jit(add)/jit(main)/jit(jax_customop_tutorial)/nki_call[func=<function nki_tensor_add_kernel_ at 0x7f6be28f6f80> grid=(2, 2) out_shape=(ShapeDtypeStruct(shape=(256, 1024), dtype=float32),)]" source_file="/home/ubuntu/nki/src/jax_neuronx/core.py" source_line=34}, backend_config="..."
    %multiply.1 = f32[256,1024]{1,0} multiply(f32[256,1024]{1,0} %multiply.0, f32[256,1024]{1,0} %custom-call.0), metadata={op_name="jit(add)/jit(main)/jit(jax_customop_tutorial)/mul" source_file="/tmp/ipykernel_3935360/2333914945.py" source_line=61}
    ROOT %tuple.10 = (f32[256,1024]{1,0}) tuple(f32[256,1024]{1,0} %multiply.1)
   }

The Neuron compiler replaces the above custom-call with
the corresponding NKI kernel implementation while optimizing the rest of the
compute graph as usual. At the end of the compilation process, a single
compiled binary NEFF file
is generated representing the entire graph
including the NKI kernel. For more information about NEFF files, see `Neuron Compiler <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/index.html>`__.


Using NKI in training graphs
----------------------------

If you are using NKI to implement a new operator in a training graph,
you might need to make the new operator interplay with the
``autograd`` engine in the framework. To do this, in PyTorch, you can
subclass the framework’s base operator class and implement both the ``forward()``
and ``backward()`` methods. The ``autograd`` engine then uses the ``backward()``
method when performing auto-differentiation. See
`Extending torch.autograd <https://pytorch.org/docs/stable/notes/extending.html>`__ in the
PyTorch Docs for instructions on doing this in PyTorch. To do this in JAX,
you can create a ``custom_vjp`` rule (vjp stands for Vector-Jacobian product), which binds the
``forward()`` and ``backward()`` calls. See
`Autodiff Cookbook <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`__ in
the JAX Docs for instructions on doing this.

Let's reuse the ``nki_tensor_add`` kernels from before and demonstrate how to train a
simple compute graph ``(a+b)*a*b`` in both PyTorch and JAX.

.. _nki_framework_custom_op_pytorch:

^^^^^^^
PyTorch
^^^^^^^

We define a ``NkiAddFunc``
class, which leverages the ``nki_tensor_add`` kernel in its ``forward()``
function. The gradients of both input tensors in ``y = a + b`` are
ones, so the ``backward()`` function
propagates the ``dy`` gradients from the previous backward function.

::

   import torch
   import torch_xla.core.xla_model as xm
   device = xm.xla_device()

   class NkiAddFunc(torch.autograd.Function):
     @staticmethod
     def forward(ctx, a, b):
       return nki_tensor_add(a, b)

     @staticmethod
     def backward(ctx, dy, *args):
       # gradients for a and b
       return dy, dy

   # now, let's define the compute graph
   a = torch.randn(256, 1024, dtype=torch.float32).to(device).detach().requires_grad_()
   b = torch.randn(256, 1024, dtype=torch.float32).to(device).detach().requires_grad_()
   c = NkiAddFunc.apply(a, b)
   out = a * b * c

   # here we define a (dummy) loss-function, in prep for backward propagation
   loss = out.sum()

   # lastly, let's invoke the auto-grad engine
   loss.backward()

   xm.mark_step()

^^^
JAX
^^^

We define a ``custom_vjp`` function ``nki_add_func`` by using
the ``@jax.custom_vjp`` decorator which directly calls
the ``nki_tensor_add`` kernel. We then define and register
the ``forward()`` and ``backward()`` implementations of the
``nki_add_func`` function via ``defvjp()``. Just like the PyTorch
example before, the ``backward()`` implementation simply passes
the gradients through. Finally, to start training, we execute the
forward pass by calling ``nki_add_func(a, b) * x * y``.
To get the gradients, we call ``jax.grad`` directly with a loss function.

::

   @jax.custom_vjp
   def nki_add_func(a, b):
      return nki_tensor_add(a, b)

   def f_forward(a, b):
      # operator output and residual (same as input here)
      return nki_add_func(a, b), (a, b)

   def f_backward(res, grad):
      # gradients for a and b
      return grad, grad

   nki_add_func.defvjp(f_forward, f_backward) # line 11

   @jax.jit
   def jax_customop_tutorial_and_grad(a, b):
      out = nki_add_func(a, b) * x * y

      # use the same dummy loss function (output sum) as PyTorch example above
      grad = jax.grad(lambda x, y: (nki_add_func(x, y) * x * y).sum(), argnums=(0, 1))(a, b)
      return out, *grad

   c, grad_a, grad_b = jax_customop_tutorial_and_grad(a, b)
