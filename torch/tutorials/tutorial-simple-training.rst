.. meta::
   :description: Simple end-to-end training loop on AWS Neuron using torch-neuronx. Eager-mode walk-through of a small GPT-2 with learning-rate scheduling, gradient clipping, logging, and checkpointing, plus an optional torch.compile variant.
   :keywords: TorchNeuron, torch-neuronx, AWS Neuron, Trainium, tutorial, training loop, GPT-2, eager, torch.compile, checkpointing, learning rate schedule
   :date-modified: 2026-10-15

.. _torchneuron-tutorial-simple-training:

===============================================
Tutorial: Train a simple model on Neuron
===============================================

This tutorial walks you through a complete training loop on AWS Trainium
using ``torch-neuronx``. When you have completed it, you will have a working,
copy-ready training-loop reference that runs in eager mode, with an optional
``torch.compile`` variant.

Overview
--------

Most training-loop examples either skip the parts that matter in production or bury them inside a
trainer class. This tutorial keeps the loop explicit: you build the model, feed it batches, and
write the forward pass, loss, backward pass, gradient clipping, optimizer step, learning-rate
schedule, logging, and checkpointing yourself, on a single NeuronCore.

The model is a 4-layer GPT-2 — 8 attention heads, 1024 hidden units, a 2048-token context, about
104M parameters. That is small enough to train on one NeuronCore, yet large enough that the problems
you meet on real models, such as checkpoint size and compilation time, still show up.

The data is a single randomly generated sequence, repeated every iteration. That needs no download
and no tokenizer, and it turns the loss into a wiring check: with the same tokens every step, a
correctly wired loop drives the loss well below its starting point, while a loop with a broken
gradient path cannot. Twenty iterations are not enough to fit the sequence outright — the loss
flattens well above zero — so read the size of the drop rather than expecting convergence. See
`Confirmation`_ for what to look for.

Eager mode is the default throughout. Step 4 adds ``torch.compile`` as a one-line change on top of
the working eager baseline.

.. note::

   This tutorial is about how a training loop is assembled, not about which model to train or how
   to scale it. The model is deliberately small so that the loop is the subject. For runnable
   training scripts and multi-core recipes, start from
   :doc:`/torch/tutorials/tutorial-torch-distributed`.

Before you start
----------------

This tutorial assumes that you have experience in the following areas:

- Writing a basic PyTorch training loop (model, optimizer, data loader).
- Running PyTorch on a non-CPU device.

Prerequisites
-------------

* An EC2 instance with Trainium hardware (``Trn2`` or ``Trn3``)
* `Neuron Deep Learning AMI <https://aws.amazon.com/machine-learning/amis/>`_ (recommended), which
  ships ``torch-neuronx`` preinstalled
* Neuron SDK 2.33 or later with PyTorch 2.13 or 2.14
* A working ``torch-neuronx`` installation — see
  :doc:`/torch/get-started/quickstart-torch-neuron`
* ``transformers`` 5.18 or later, which supplies the GPT-2 model definition:
  ``pip install "transformers>=5.18"``
* About 2 GB of free disk space for the checkpoint

Confirm your environment before you start:

.. code-block:: bash

   neuron-ls

If at least one Neuron device is listed, you are ready. For example, on a ``trn2.3xlarge``:

.. code-block:: text

   instance-type: trn2.3xlarge
   instance-id: i-0c34d7fcbe09ae42a
   logical-neuroncore-config: 2
   +--------+--------+----------+--------+--------------+----------+------+
   | NEURON | NEURON |  NEURON  | NEURON |     PCI      |   CPU    | NUMA |
   | DEVICE | CORES  | CORE IDS | MEMORY |     BDF      | AFFINITY | NODE |
   +--------+--------+----------+--------+--------------+----------+------+
   | 0      | 4      | 0-3      | 96 GB  | 0000:33:00.0 | 0-11     | 0    |
   +--------+--------+----------+--------+--------------+----------+------+

If the command is not found or lists no devices, work through
:doc:`/torch/get-started/quickstart-torch-neuron` first.

Prepare your environment
------------------------

Create a working directory for the script and its checkpoint:

.. code-block:: bash

   mkdir -p ~/simple-training
   cd ~/simple-training

You will build one file, ``train.py``, across Step 1 and Step 2.

Step 1: Pick and wire up the model
----------------------------------

In this step, you will define a small but realistic model and move it to the
Neuron device.

Rather than hand-rolling a transformer, take the GPT-2 definition from ``transformers`` and shrink
it with a custom config. Four layers and a 1024-wide hidden state give you a real GPT-2 at a size
that trains on one NeuronCore.

Start ``train.py`` with the imports, the run configuration, and the model:

.. code-block:: python

   import math
   import os

   import torch
   from transformers import GPT2Config, GPT2LMHeadModel

   SEQ_LENGTH = 2048
   VOCAB_SIZE = 50257
   ITERATIONS = 20
   LEARNING_RATE = 5e-5
   GRAD_CLIP = 1.0
   WARMUP_ITERS = 2
   CKPT_DIR = "checkpoints"

   # Seed before constructing the model so weight init and the fake data are reproducible.
   torch.manual_seed(0)

   # 4 layers, 8 heads, head_dim 128 -> hidden size 1024.
   config = GPT2Config(
       n_layer=4,
       n_head=8,
       n_embd=1024,
       vocab_size=VOCAB_SIZE,
       n_positions=SEQ_LENGTH,
   )

   # Device placement is the only Neuron-specific line in the script.
   device = "neuron"
   model = GPT2LMHeadModel(config).to(device)
   model.loss_type = "ForCausalLM"  # silences a transformers warning; the loss is unchanged

   n_params = sum(p.numel() for p in model.parameters())
   print(f"model on {device}, {n_params / 1e6:.2f}M parameters")

.. note::

   Notice there is no ``import torch_neuronx``. TorchNeuron autoloads on Neuron hardware as of
   PyTorch 2.9, so ``torch`` alone is enough. The explicit import is only required when you call
   NKI kernels. See :doc:`/torch/get-started/quickstart-nki-kernel`.


Step 2: Build the training loop
-------------------------------

In this step, you will assemble data loading, the forward pass, loss, backward,
the optimizer step, and a basic checkpoint.

The dataset generates one random sequence and yields it on every iteration. Append it to
``train.py``:

.. code-block:: python

   class FakeDataset:
       """Fake dataset that repeats a single random sequence."""

       def __init__(self, seq_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE):
           self.input_ids = torch.randint(0, vocab_size, (seq_length,))

       def __iter__(self):
           while True:
               yield {"input_ids": self.input_ids}

The sequence is generated once, in ``__init__``, rather than per iteration. Draw a fresh one each
time and there is nothing to learn: the loss sits flat at ``ln(VOCAB_SIZE)``, and a broken loop looks
exactly like a healthy one.

.. note::

   Weights and activations are ``float32``; token IDs are ``int64``, which the causal-LM loss
   requires. ``int64`` is supported: in eager mode, ops with native ``int64`` support run in 64
   bits on Neuron, and others run on device using the low 32 bits, which is harmless for a
   50257-token vocabulary. See :ref:`torchneuron-guidelines-memory-management` for how 64-bit
   tensors occupy device memory.

Add the schedule and the checkpoint helper. The schedule is linear warmup into cosine decay, which
is what you would reach for on a real run:

.. code-block:: python

   def lr_multiplier(step: int) -> float:
       if step < WARMUP_ITERS:
           return (step + 1) / WARMUP_ITERS
       progress = (step - WARMUP_ITERS) / max(1, ITERATIONS - WARMUP_ITERS)
       return 0.5 * (1.0 + math.cos(math.pi * progress))


   def save_checkpoint(step, model, optimizer, device):
       os.makedirs(CKPT_DIR, exist_ok=True)
       path = os.path.join(CKPT_DIR, f"step-{step:04d}.pt")

       # Unwrap the model if it has been compiled. Step 4 explains why.
       target = getattr(model, "_orig_mod", model)

       # Move the whole module in one call, not tensor by tensor. GPT-2 ties lm_head.weight to
       # the token embedding, and copying each tensor separately would break that shared storage
       # and write the embedding matrix twice -- 1.4 GB instead of 1.2 GB.
       target.to("cpu")
       torch.save(
           {
               "step": step,
               "model": target.state_dict(),
               "optimizer": optimizer.state_dict(),
           },
           path,
       )
       target.to(device)  # restore, so this is safe to call from inside the loop too
       return path

Now the loop itself:

.. code-block:: python

   optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
   scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)

   dataset = FakeDataset()
   model.train()

   for step, sample in enumerate(dataset):
       if step >= ITERATIONS:
           break

       input_ids = sample["input_ids"].unsqueeze(0).to(device)  # add the batch dimension
       attention_mask = torch.ones_like(input_ids)

       # Passing labels makes the model return the causal-LM loss directly.
       outputs = model(
           input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
       )
       loss = outputs.loss

       loss.backward()
       grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
       optimizer.step()
       scheduler.step()
       optimizer.zero_grad()

       print(
           f"step {step:3d} | loss {loss.item():7.4f} "
           f"| grad_norm {grad_norm.item():7.3f} "
           f"| lr {scheduler.get_last_lr()[0]:.2e}"
       )

   print(f"final checkpoint: {save_checkpoint(ITERATIONS, model, optimizer, device)}")

.. note::

   ``loss.item()`` and ``grad_norm.item()`` both synchronize with the device. Logging every step is
   fine at 20 iterations; on a long run, log every N steps instead.

.. note::

   The checkpoint holds the model plus AdamW's two moment buffers — 396 MiB of weights and 793 MiB
   of optimizer state, about 1.2 GB on disk. That is why this loop saves once, at the end. To
   checkpoint periodically, call ``save_checkpoint`` inside the loop and budget the disk
   accordingly.

.. note::

   The optimizer state is what makes the checkpoint resumable rather than just a copy of the
   weights. To pick a run back up:

   .. code-block:: python

      ckpt = torch.load("checkpoints/step-0020.pt", map_location="cpu", weights_only=False)
      model.load_state_dict(ckpt["model"])
      optimizer.load_state_dict(ckpt["optimizer"])
      start_step = ckpt["step"]
      model.to(device)

Step 3: Run in eager mode
-------------------------

In this step, you will launch the training loop end-to-end in eager mode and
observe the expected log lines.

Run the script as written. No environment variables and no launcher are needed for a single
NeuronCore:

.. code-block:: bash

   python train.py

Scroll past any warnings to the step lines:

.. code-block:: text

   step   0 | loss 11.0252 | grad_norm   3.532 | lr 5.00e-05
   step   1 | loss 10.7592 | grad_norm   3.345 | lr 5.00e-05
   step   2 | loss 10.3775 | grad_norm   3.138 | lr 4.96e-05
   step   3 | loss 10.0323 | grad_norm   3.579 | lr 4.85e-05
   step   4 | loss  9.6766 | grad_norm   3.992 | lr 4.67e-05
   step   5 | loss  9.3294 | grad_norm   4.340 | lr 4.42e-05
   step   6 | loss  8.9766 | grad_norm   4.701 | lr 4.11e-05
   step   7 | loss  8.6550 | grad_norm   4.755 | lr 3.75e-05
   step   8 | loss  8.3712 | grad_norm   5.153 | lr 3.36e-05
   step   9 | loss  8.1126 | grad_norm   5.184 | lr 2.93e-05
   step  10 | loss  7.8868 | grad_norm   4.771 | lr 2.50e-05
   step  11 | loss  7.7041 | grad_norm   5.402 | lr 2.07e-05
   step  12 | loss  7.5460 | grad_norm   5.340 | lr 1.64e-05
   step  13 | loss  7.3976 | grad_norm   5.264 | lr 1.25e-05
   step  14 | loss  7.2976 | grad_norm   5.207 | lr 8.93e-06
   step  15 | loss  7.2354 | grad_norm   5.587 | lr 5.85e-06
   step  16 | loss  7.1648 | grad_norm   4.962 | lr 3.35e-06
   step  17 | loss  7.1232 | grad_norm   4.959 | lr 1.51e-06
   step  18 | loss  7.0915 | grad_norm   4.775 | lr 3.80e-07
   step  19 | loss  7.0892 | grad_norm   5.233 | lr 0.00e+00
   final checkpoint: checkpoints/step-0020.pt

Read the three columns together. ``loss`` falls from about ``ln(VOCAB_SIZE)`` — the value you get
when the model has learned nothing — as the model starts to fit the repeated sequence, then
flattens over the last few steps. ``lr`` rises
through warmup and then decays to zero on the cosine schedule, which is why the loss flattens near
the end: the run is out of learning rate, not out of things to learn.

.. note::

   On a ``trn2.3xlarge`` the first run took 2–3 minutes, and every run after it about 1 minute. The
   difference is one-time work the Neuron stack caches on disk — NKI kernel traces and the HLO
   lowering — so eager mode benefits from the compilation caches too, not just
   ``torch.compile``. Both figures include process startup and writing the 1.2 GB checkpoint. See
   :ref:`torchneuron-env-vars` for the cache directories.


A ``loss`` column that ends well below where it started is what you are looking for. See
`Confirmation`_ for the bounds to check against.

Step 4 (optional): Add ``torch.compile``
-----------------------------------------

In the final step, you will add a minimal ``torch.compile`` variant on top of
the eager baseline and confirm the loss trajectory matches.

Get the eager run working first. Then compile the model — this is the entire change:

.. code-block:: python

   model = GPT2LMHeadModel(config).to(device)
   model.loss_type = "ForCausalLM"
   model = torch.compile(model, backend="neuron", dynamic=False)

Everything else in the loop is untouched. Sequence length and batch size are fixed, so the compiled
graph is reused after the first step.

.. note::

   Pass ``dynamic=False``: dynamic shapes are not supported, and ``mode="default"`` is the only
   supported mode. For production runs, add ``fullgraph=True`` so graph breaks raise instead of
   silently splitting the model. See :ref:`torch-compile-optimize-howto`.

.. note::

   Checkpointing needs no changes, and this is the step the unwrap line in ``save_checkpoint`` was
   for. ``torch.compile`` returns a wrapper around your model, and that wrapper prefixes every
   ``state_dict`` key with ``_orig_mod.``. Taking ``getattr(model, "_orig_mod", model)`` first keeps
   the saved keys identical to Step 3's, so checkpoints from either step load into the uncompiled
   model and into each other.

Run it again:

.. code-block:: bash

   python train.py

With the same seed, the same data, and the same schedule, the compiled run tracks the eager one:

.. code-block:: text

   step   0 | loss 11.0211 | grad_norm   3.583 | lr 5.00e-05
   step   1 | loss 10.7711 | grad_norm   3.359 | lr 5.00e-05
   step   2 | loss 10.3858 | grad_norm   3.119 | lr 4.96e-05
   ...
   step  18 | loss  7.1313 | grad_norm   4.986 | lr 3.80e-07
   step  19 | loss  7.1183 | grad_norm   5.074 | lr 0.00e+00

The numbers do not match the eager run exactly. The compiled graph orders operations differently, so
small differences appear and accumulate. The shape is what matters: the same steady descent, ending
in roughly the same place.

Confirmation
------------

The loss curve is the success signal: gradients that reach every parameter move the loss well
below its starting point, and a loop with a broken gradient path does not. The run succeeded if
all of the following hold:

- Every step prints a finite ``loss``, ``grad_norm``, and ``lr`` (no ``nan`` or ``inf``), for all
  20 iterations.
- The loss starts near ``ln(VOCAB_SIZE)`` — about 10.8 for a 50257-token vocabulary, the value you
  get before the model has learned anything — and ends at least 3 below it. A loop that is only
  partly wired can still drift downwards, so the size of the drop is what matters, not the
  direction alone.
- ``grad_norm`` stays in single digits throughout. It should neither collapse towards zero nor
  grow without bound.
- ``checkpoints/step-0020.pt`` exists.


Verify the checkpoint loads back:

.. code-block:: bash

   python -c "
   import torch
   ckpt = torch.load('checkpoints/step-0020.pt', map_location='cpu', weights_only=False)
   print('step:', ckpt['step'])
   print('tensors:', len(ckpt['model']))
   "

.. code-block:: text

   step: 20
   tensors: 53

``weights_only=False`` is needed because the checkpoint carries the optimizer state, which includes
Python objects as well as tensors. Use it only on checkpoints you produced yourself.

Congratulations! You have now trained a simple model on AWS Trainium with
``torch-neuronx``. If you encountered any issues, see the **Common issues**
section below.

Common issues
-------------


- **The loss stays flat**: gradients are not reaching the parameters. Check that
  ``loss.backward()`` runs before ``optimizer.step()``, that ``optimizer.zero_grad()`` comes after
  the step, and that ``FakeDataset`` returns the same ``input_ids`` every iteration.

- **The loss is NaN**: lower the learning rate — ``5e-5`` is safe here — and confirm gradient
  clipping is still in the loop.

- **Out of memory**: halve ``SEQ_LENGTH`` before changing the model config, and check NeuronCore
  memory use with ``neuron-top``.

- **Compilation fails in Step 4**: get the eager run working first, then check that ``SEQ_LENGTH``
  and the batch dimension are constant across steps. A changing shape forces a recompile, and
  dynamic shapes are not supported.

Clean up
--------

Remove the checkpoint and the compilation caches:

.. code-block:: bash

   rm -rf ~/simple-training/checkpoints
   rm -rf /tmp/neff_cache /tmp/hlo_cache /tmp/local_cache

Those three directories are the NEFF cache, the HLO cache, and the local cache that holds filelocks
and scratch. All of them default to a path under ``/tmp``, which does not survive a reboot or an
instance stop. To keep compiled artifacts across restarts, point ``TORCH_NEURONX_NEFF_CACHE_DIR``
and ``TORCH_NEURONX_HLO_CACHE_DIR`` at a persistent location before you run. See
:ref:`torchneuron-env-vars` for every cache variable and :doc:`/torch/compile/about-torch-compile`
for how the cache key is derived.

Stop or terminate the instance when you are finished, because you are billed while it runs.

Next steps
----------

- :doc:`/torch/tutorials/tutorial-torch-distributed` — Take this loop to multiple NeuronCores.
- :ref:`torch-compile-optimize-howto` — Tune ``torch.compile`` for throughput and measure the
  result.

Further reading
---------------

- :doc:`/torch/get-started/quickstart-build-first-model` — The short version: eager mode,
  ``torch.compile``, and mixed mode, without the training loop.
- :doc:`/torch/reference/best-practices-pytorch-neuron` — Execution modes, mixed precision, and
  profiling.
- :ref:`torchneuron-guidelines-memory-management` — Memory behavior behind the dtype and
  checkpoint-size notes above.
