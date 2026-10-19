.. meta::
   :description: Run your first PyTorch model on AWS Trainium with torch-neuronx. Confirm TorchNeuron works in eager mode, with torch.compile using the neuron backend, and in mixed mode.
   :keywords: TorchNeuron, torch-neuronx, quickstart, build model, first model, eager mode, torch.compile, mixed mode, PyTorch, Trainium, AWS Neuron
   :date-modified: 2026-10-15

.. _torchneuron-quickstart-build-first-model:

Quickstart: Build your first model with torch-neuronx
=======================================================

This topic guides you through running your first PyTorch model on AWS Trainium with
``torch-neuronx``. It shows the three ways TorchNeuron executes a model — eager mode,
``torch.compile`` with the ``neuron`` backend, and mixed mode — so you can confirm which one fits
your workload. When you have completed it, you will have run the same model all three ways on
Trainium hardware. This page runs forward passes only; for a complete training loop, continue to
:doc:`/torch/tutorials/tutorial-simple-training`.

**This quickstart is for**: ML engineers confirming that TorchNeuron works for their use case

**Time to complete**: ~10 minutes

Prerequisites
-------------

* An EC2 instance with Trainium hardware (Trn2 or Trn3)
* A `Neuron Base DLAMI <https://aws.amazon.com/machine-learning/amis/>`_ (recommended) or an
  Ubuntu AMI with the Neuron packages installed
* Neuron SDK 2.33 or later with PyTorch 2.13 or 2.14
* A working ``torch-neuronx`` installation — see
  :doc:`/torch/get-started/quickstart-torch-neuron`
* Familiarity with defining a model in PyTorch

Activate the environment you set up in the install quickstart (for example
``source ~/torch-neuronx-venv/bin/activate``) before running the scripts.

Confirm your environment before you start:

.. code-block:: bash

   neuron-ls

If at least one Neuron device is listed, you are ready. For example, on a ``trn2.3xlarge``:

.. code-block:: text

   instance-type: trn2.3xlarge
   logical-neuroncore-config: 2
   +--------+--------+----------+--------+--------------+----------+------+
   | NEURON | NEURON |  NEURON  | NEURON |     PCI      |   CPU    | NUMA |
   | DEVICE | CORES  | CORE IDS | MEMORY |     BDF      | AFFINITY | NODE |
   +--------+--------+----------+--------+--------------+----------+------+
   | 0      | 4      | 0-3      | 96 GB  | 0000:33:00.0 | 0-11     | 0    |
   +--------+--------+----------+--------+--------------+----------+------+

If the command is not found or lists no
devices, work through :doc:`/torch/get-started/quickstart-torch-neuron` first.


Step 1: Run a model in eager mode
----------------------------------

In this step, you will define a small model, move it to the Neuron device, and run a forward pass
with no compilation and no code changes beyond device placement.

Eager mode dispatches operations as they are called, which makes it the fastest way to confirm that
your model runs at all. Save the following as ``first_model.py``:

.. code-block:: python

   import torch
   import torch.nn as nn


   class TinyMLP(nn.Module):
       def __init__(self, d_in=128, d_hidden=256, d_out=10):
           super().__init__()
           self.stem = nn.Linear(d_in, d_hidden)
           self.act = nn.ReLU()
           self.head = nn.Linear(d_hidden, d_out)

       def forward(self, x):
           return self.head(self.act(self.stem(x)))


   # Device placement is the only change from CPU or CUDA code.
   device = "neuron"
   model = TinyMLP().to(device)
   x = torch.randn(8, 128, dtype=torch.float32).to(device)

   with torch.no_grad():
       logits = model(x)

   print(f"device: {logits.device}")
   print(f"output shape: {tuple(logits.shape)}")
   print("eager mode OK")

.. note::

   ``int64`` is supported. ``float64`` runs as ``float32``, so prefer ``float32`` or ``bfloat16``
   for floating point.

.. note::

   Notice there is no ``import torch_neuronx``. TorchNeuron loads automatically when
   ``torch-neuronx`` is installed, so ``torch`` alone is enough. The explicit import is only
   required when you call NKI kernels. See :doc:`/torch/get-started/quickstart-nki-kernel`.

Run it:

.. code-block:: bash

   python first_model.py

You should see output similar to:

.. code-block:: text

   device: neuron:0
   output shape: (8, 10)
   eager mode OK

The ``neuron:0`` device confirms the forward pass ran on Trainium rather than on CPU.

.. note::

   Eager mode on Neuron enqueues operations asynchronously and preserves execution order.
   Operation concatenation, which combines eligible eager operations, is experimental and off by
   default in 2.33; see :ref:`best-practices-pytorch-neuron`.

Step 2: Compile the model with torch.compile
---------------------------------------------

In this step, you will wrap the same model with ``torch.compile`` and the ``neuron`` backend, then
run it and see what compilation costs.

``torch.compile`` captures an FX graph and hands it to the Neuron compiler, which produces a NEFF
that runs as a single optimized unit. Compilation happens on the first call, not when you wrap the
model. Save the following as ``compiled_model.py``:

.. code-block:: python

   import torch
   import torch.nn as nn


   class TinyMLP(nn.Module):
       def __init__(self, d_in=128, d_hidden=256, d_out=10):
           super().__init__()
           self.stem = nn.Linear(d_in, d_hidden)
           self.act = nn.ReLU()
           self.head = nn.Linear(d_hidden, d_out)

       def forward(self, x):
           return self.head(self.act(self.stem(x)))


   device = "neuron"
   model = TinyMLP().to(device)
   x = torch.randn(8, 128, dtype=torch.float32).to(device)

   # Wrapping is free; the compile happens on the first call below.
   compiled = torch.compile(model, backend="neuron")

   with torch.no_grad():
       logits = compiled(x)

   print(f"device: {logits.device}")
   print(f"output shape: {tuple(logits.shape)}")
   print("torch.compile OK")

Run it:

.. code-block:: bash

   python compiled_model.py


You should see output similar to:

.. code-block:: text

   device: neuron:0
   output shape: (8, 10)
   torch.compile OK

The output is the same as Step 1 — the point is that it took noticeably longer to get there. That
pause was compilation. Every later call with the same input shapes and dtypes reuses the compiled
NEFF and runs at full speed, so in a real workload the one-time cost is amortized over many
iterations. That is why ``torch.compile`` suits production runs while eager mode suits debugging.

.. note::

   Run the script a second time and the pause is gone: the compiled NEFF is cached on disk under
   ``TORCH_NEURONX_NEFF_CACHE_DIR`` (default ``/tmp/neff_cache``). Change the model or the input
   shapes and the full cost returns.

   This quickstart does not measure the speedup. See :ref:`torch-compile-optimize-howto` for that,
   :doc:`/torch/compile/about-torch-compile` for the caching layers, and
   :ref:`torchneuron-env-vars` for every cache directory.

.. note::

   The Neuron backend supports ``mode="default"`` only. Drop CUDA-only arguments such as
   ``mode="max-autotune"``. Add ``fullgraph=True`` so graph breaks raise an error instead of
   silently splitting the model — see :ref:`torch-compile-optimize-howto`.

Step 3: Combine both in mixed mode
-----------------------------------

In the final step, you will run most of the model in eager mode while compiling only the part that
benefits from it.

Mixed mode is the practical middle ground: you keep eager mode's debuggability across the model and
apply ``torch.compile`` to performance-critical components. Save the following as
``mixed_mode.py``:

.. code-block:: python

   import torch
   import torch.nn as nn


   class MLPBlock(nn.Module):
       """The performance-critical component we want compiled."""

       def __init__(self, d_model=256, d_hidden=1024):
           super().__init__()
           self.up = nn.Linear(d_model, d_hidden)
           self.act = nn.GELU()
           self.down = nn.Linear(d_hidden, d_model)

       def forward(self, x):
           return self.down(self.act(self.up(x)))


   class SmallModel(nn.Module):
       def __init__(self, d_in=128, d_model=256, d_out=10):
           super().__init__()
           self.stem = nn.Linear(d_in, d_model)
           self.block = MLPBlock(d_model)
           self.head = nn.Linear(d_model, d_out)

       def forward(self, x):
           x = self.stem(x)     # eager
           x = self.block(x)    # compiled
           return self.head(x)  # eager


   device = "neuron"
   model = SmallModel().to(device)

   # Compile only the hot submodule; the rest of the model stays in eager mode.
   model.block = torch.compile(model.block, backend="neuron")

   x = torch.randn(8, 128, dtype=torch.float32).to(device)

   with torch.no_grad():
       logits = model(x)

   print(f"device: {logits.device}")
   print(f"output shape: {tuple(logits.shape)}")
   print("mixed mode OK")

Run it:

.. code-block:: bash

   python mixed_mode.py


You should see output similar to:

.. code-block:: text

   device: neuron:0
   output shape: (8, 10)
   mixed mode OK

Only ``model.block`` was compiled. You can still set a breakpoint in ``SmallModel.forward`` and
inspect ``x`` around the compiled call.

Confirmation
------------

You have confirmed TorchNeuron works for your use case when all three scripts report a ``neuron:0``
device and print their closing line:

.. code-block:: text

   eager mode OK
   torch.compile OK
   mixed mode OK

Congratulations! You have now run a PyTorch model on Trainium in eager mode, with
``torch.compile``, and in mixed mode. If you encountered any issues, see the **Common issues**
section below.


Common issues
-------------

- **import torch fails or no neuron device after install**: Your virtual environment is not
  active. Run ``source ~/torch-neuronx-venv/bin/activate`` (or the DLAMI equivalent) and retry.

- **Device not found**: Verify you are on a Trainium instance and ``torch-neuronx`` is installed.
  See :doc:`/torch/get-started/quickstart-torch-neuron` for setup instructions.

- **Out of memory**: Reduce the batch size, or check NeuronCore memory use with ``neuron-top``.

- **The first iteration is slow**: This is compilation, not a performance problem. Subsequent
  iterations with the same shapes and dtypes reuse the cached NEFF.

- **Compilation fails with a dynamic shapes error**: Dynamic shapes are not supported. Pass
  ``dynamic=False`` and keep your input shapes fixed between calls.

- **A mode argument is ignored**: Only ``mode="default"`` is supported. CUDA-only modes such
  as ``max-autotune`` are not available on the Neuron backend.

Clean up
--------

Remove the three scripts and the compiled NEFFs when you are done:

.. code-block:: bash

   rm first_model.py compiled_model.py mixed_mode.py
   rm -rf /tmp/neff_cache


Next steps
----------

Now that you've completed this quickstart, explore these topics:

* :doc:`/torch/tutorials/tutorial-simple-training` — Build a complete, copy-ready training loop
* :ref:`torch-compile-optimize-howto` — Tune ``torch.compile`` for throughput
* :ref:`best-practices-pytorch-neuron` — Best practices for execution modes, mixed precision, and
  profiling

Further reading
---------------

- :doc:`/torch/about-pytorch-neuron` — TorchNeuron architecture and execution modes
- :doc:`/torch/compile/reference-torch-compile` — ``torch.compile`` supported parameters for
  Trainium
- :doc:`/torch/get-started/quickstart-nki-kernel` — Drop to a custom NKI kernel when the compiler
  does not reach the performance you need
