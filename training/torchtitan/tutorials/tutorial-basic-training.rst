.. meta::
   :description: Basic TorchTitan training tutorial on AWS Neuron. A first complete pre-training run on Trainium with TorchNeuron.
   :keywords: TorchTitan, TorchNeuron, AWS Neuron, Trainium, tutorial, pre-training, basic training
   :date-modified: 2026-10-15

.. _torchtitan-neuron-tutorial-basic:

======================================================
Tutorial: Run a basic TorchTitan training on Neuron
======================================================

This tutorial walks you through a first, complete TorchTitan training run on
AWS Trainium using TorchNeuron. When you have completed it, you will have a
working end-to-end TorchTitan training example that trains Qwen3-8B on Trainium
(on a single node and on multiple nodes).

Overview
--------

TorchTitan is PyTorch's native platform for large-model pre-training. This
tutorial runs it on Trainium using Qwen3-8B as the reference model: large
enough to exercise tensor parallelism and FSDP, while staying small enough that
the focus stays on the training example itself.

The training path below runs in **eager mode** — no ``torch.compile``. Eager
mode is the simplest way to confirm the example works end-to-end; once it is
running, the ``torch.compile`` and Simple FSDP paths in
:doc:`/training/torchtitan/tutorials/tutorial-simple-fsdp-training` are the
next step.

Before you start
----------------

This tutorial assumes that you have experience in the following areas:

- Running a single-device PyTorch training loop on Neuron.
- Basic familiarity with ``torchrun`` and distributed data-parallel training.

You do **not** need a prior TorchTitan installation — the setup doc linked in
"Prepare your environment" below walks you through it.

Model details
-------------

This tutorial trains **Qwen3-8B** with the following configuration:

- Module: ``torchtitan_neuron.models.qwen3``
- Config: ``qwen3_8b_tp4_fsdp16_sac``
- Parallelism: tensor-parallel degree 4, data-parallel shard degree ``-1``
  (FSDP fills the remaining cores) — baked into the config
- Execution: eager mode (no ``torch.compile``), with selective activation
  checkpointing (SAC)

The ``qwen3_8b_tp4_fsdp16_sac`` config already sets the TP4 × FSDP16
parallelism layout, the batch size, and the sequence length, so you
do not pass those as command-line flags. You select it by name through the
``CONFIG`` environment variable (see Step 2).

**Limitations:**

- Validated on a single ``trn2.48xlarge`` node (64 NeuronCores, TP4 × FSDP16),
  and on two ``trn2.48xlarge`` nodes (128 NeuronCores, TP4 × FSDP32).
- The batch size shipped in the config is the tested configuration; raising
  it may require more aggressive activation checkpointing to stay within device
  memory.

Prerequisites
-------------

- Trn2 or Trn3 instance
- Neuron SDK 2.33+
- PyTorch 2.14
- ``torch-neuronx`` (native TorchNeuron) verified — run
  :doc:`/torch/get-started/quickstart-torch-neuron` and confirm
  ``device='neuron'`` resolves before continuing

----

Prepare your environment
------------------------

Install and configure TorchTitan on your Trainium instance by following
:doc:`/training/torchtitan/get-started/setup-torchtitan-neuron`. Return here
once ``import torchtitan`` succeeds and the setup doc's verification run
passes.

Step 1: Download the tokenizer
------------------------------

From the ``torchtitan-neuron`` directory you installed in the setup step,
download the Qwen3 tokenizer from Hugging Face:

.. code-block:: bash

   python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-8B --assets tokenizer

Tokenizers are saved to ``./assets/hf/<model-name>/`` by default. TorchTitan
streams the C4 dataset from Hugging Face during training, so you do not
download it separately. If the download fails with HTTP 429, see
:ref:`torchtitan-neuron-tutorial-basic-common-issues`.

Step 2: Launch the training run
-------------------------------

TorchTitan ships a launcher script, ``run_train.sh``, that wraps ``torchrun``.
You select the model and configuration through the ``MODULE`` and ``CONFIG``
environment variables and set the NeuronCore count with ``NGPU``. ``MODULE``
is the model's module path, ``torchtitan_neuron.models.qwen3``:

.. code-block:: bash

   # Train Qwen3-8B on a trn2.48xlarge instance (64 NeuronCores):

   NGPU=64 \
   MODULE=torchtitan_neuron.models.qwen3 \
   CONFIG=qwen3_8b_tp4_fsdp16_sac \
   ./run_train.sh

``run_train.sh`` passes any extra arguments straight through to
``torchtitan.train``, so you can override a config value on the command line
without editing the config — for example, to shorten the run:

.. code-block:: bash

   NGPU=64 MODULE=torchtitan_neuron.models.qwen3 CONFIG=qwen3_8b_tp4_fsdp16_sac \
   ./run_train.sh --training.steps 10 --metrics.log_freq 1

Neuron environment variables of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``NGPU`` — the number of NeuronCores the run uses. ``run_train.sh`` passes
  this to ``torchrun`` as ``--nproc_per_node`` (see :ref:`the GPU/Neuron
  contrast in Step 4 <torchtitan-neuron-tutorial-basic-single-multi-node>`). On a
  ``trn2.48xlarge`` this is 64.
- ``MODULE`` / ``CONFIG`` — select the model
  (``torchtitan_neuron.models.qwen3``) and the registered training config
  (``qwen3_8b_tp4_fsdp16_sac``).

For the full set of TorchNeuron environment variables — compile-cache location,
CPU-fallback controls, and debug directories — see
:doc:`/torch/reference/env-vars`. The ``torch.compile`` backend and its
Neuron-specific flags are covered in
:doc:`/training/torchtitan/tutorials/tutorial-simple-fsdp-training`.

Step 3: Confirm the run succeeds
--------------------------------

Watch the training log for per-step loss lines. A healthy step looks like:

.. code-block:: text

   step:  3  loss: 11.04064  grad_norm:  5.3200  memory:  9.38GiB(26.07%)  tps: 587  tflops: 30.92  mfu: 19.57%

Confirm the run is healthy by checking that:

- Training begins stepping and per-step loss lines appear.
- Loss values are finite (not ``NaN``) and trend downward over the first
  several steps.

Congratulations! You have now completed a first TorchTitan training run on
AWS Trainium. If you encountered any issues, see
:ref:`torchtitan-neuron-tutorial-basic-common-issues`.

.. _torchtitan-neuron-tutorial-basic-single-multi-node:

Step 4 (optional): Scale to multiple nodes
------------------------------------------

Neuron distributed launches differ from GPU in how the process count maps to
hardware:

- **Single-node:** ``NGPU=64`` on a single ``trn2.48xlarge``
  is 64 NeuronCores, arranged here as TP4 × FSDP16. On GPU the equivalent
  ``--nproc_per_node`` maps to the number of GPUs; on Neuron it maps to
  **NeuronCores**.
- **Multi-node:** ``run_train.sh`` reads ``NNODES`` and ``RDZV_ENDPOINT``
  and passes them to ``torchrun``. A launcher, for example MPI,
  starts ``run_train.sh`` once on each node.

.. _torchtitan-neuron-tutorial-basic-multi-node-mpi:

Multi-node with MPI
~~~~~~~~~~~~~~~~~~~~~

Use ``mpirun`` to start ``run_train.sh`` once on each node. ``run_train.sh``
passes ``NNODES`` and ``RDZV_ENDPOINT`` to ``torchrun``, and the ranks of all
nodes meet at the rendezvous endpoint. The example trains Qwen3-8B on two
``trn2.48xlarge`` nodes (128 NeuronCores, TP4 × FSDP32).

Before you start, make sure that:

- Each node has the same environment: ``torch-neuronx``, the
  ``torchtitan-neuron`` directory at the same path, and the tokenizer from
  Step 1.
- A hostfile lists each node with one slot, for example:

  .. code-block:: text

     node-0 slots=1
     node-1 slots=1

- The launcher can start a command on each node. This command prints the
  host name of each node:

  .. code-block:: bash

     mpirun --allow-run-as-root -np 2 --hostfile <hostfile> hostname

- Each node has EFA. This command prints the number of EFA endpoints on each
  node, which must be greater than 0:

  .. code-block:: bash

     mpirun --allow-run-as-root -np 2 --hostfile <hostfile> \
       bash -c 'echo $(hostname) $(/opt/amazon/efa/bin/fi_info -p efa -t FI_EP_RDM | grep -c provider:)'

Run this command on the launcher:

.. code-block:: bash

   mpirun --allow-run-as-root -np 2 --hostfile <hostfile> \
     -x PATH=<venv-path>/bin:$PATH -x CCOM_SOCKET_IFNAME=eth0 \
     -x NEURON_LOGICAL_NC_CONFIG=2 -x NEURON_RT_NUM_CORES=64 \
     bash -c 'cd /path/to/torchtitan-neuron && \
     NGPU=64 NNODES=2 RDZV_ENDPOINT=<node-0 address>:29500 \
     MODULE=torchtitan_neuron.models.qwen3 CONFIG=qwen3_8b_tp4_fsdp16_sac \
     ./run_train.sh'

- ``-np`` and ``NNODES`` are the number of nodes. ``NGPU`` is the number of
  NeuronCores on each node.
- ``RDZV_ENDPOINT`` is an address of the first node that all nodes can reach,
  plus a free port.
- ``-x PATH=<venv-path>/bin:$PATH`` sets ``PATH`` on each node, so that each
  node finds ``torchrun`` in the Python environment of the setup step.
- Use the full path to ``torchtitan-neuron``. ``mpirun`` starts each node's
  process in the launcher's current directory.
- ``CCOM_SOCKET_IFNAME`` sets the network interface for the Neuron collectives
  bootstrap. Set it to the interface of your nodes. It is necessary when the
  interface cannot be found automatically, for example in a container.
- ``NEURON_LOGICAL_NC_CONFIG=2`` sets the logical NeuronCore size (LNC 2),
  and ``NEURON_RT_NUM_CORES=64`` sets the number of NeuronCores per node.
  ``NEURON_LOGICAL_NC_CONFIG`` is valid on both Trn2 and Trn3.

The config sets the data-parallel shard degree to ``-1``, so FSDP fills the
remaining NeuronCores. For four nodes, change ``-np`` and ``NNODES`` to ``4``.
The command does not change in other ways.

.. note::

   - Each rank streams the C4 dataset from Hugging Face. With many ranks, an
     anonymous download can fail with HTTP 429 (rate limit). If you see this
     error, set ``HF_TOKEN`` on each node, for example with ``-x HF_TOKEN``.

----

.. _torchtitan-neuron-tutorial-basic-common-issues:

Common issues
-------------

- **TorchTitan cannot find the Neuron device**: Confirm ``torch-neuronx`` is
  installed and ``device='neuron'`` resolves in a plain PyTorch script first.
  See :doc:`/torch/get-started/quickstart-torch-neuron`.
- **OOM at this batch size**: Enable activation checkpointing before lowering
  the batch size. See :doc:`/training/torchtitan/best-practices`.
- **Dataset download fails with HTTP 429**: Hugging Face limits anonymous
  downloads. Set ``HF_TOKEN`` to your Hugging Face access token.
- **Multi-node run fails at process group initialization with NET/OFI
  errors**: Check EFA on each node with the command in
  :ref:`torchtitan-neuron-tutorial-basic-multi-node-mpi`.
  If EFA is present on each node, contact your cluster administrator about the
  EFA connection between the nodes.

Clean up
--------

Stop the training processes with ``Ctrl-C`` on the launcher: ``torchrun`` for
a single-node run, or the ``mpirun`` process for a multi-node run — ``mpirun``
forwards the interrupt to the per-node processes and tears the job down on
every node. Then, if you no longer need them, remove the checkpoint output
directory TorchTitan wrote under its output path.

.. include:: /includes/torch/trn3-sizing-note.txt

Next steps
----------

- :doc:`/training/torchtitan/tutorials/tutorial-simple-fsdp-training` — FSDP
  training with ``torch.compile`` for Llama and DeepSeek.
- :doc:`/training/torchtitan/tutorials/tutorial-gpt-oss-training` — GPT-OSS
  end-to-end.
- :doc:`/training/torchtitan/best-practices` — Tuning once the example is
  working.

Further reading
---------------

- :doc:`/training/torchtitan/get-started/setup-torchtitan-neuron`
- :doc:`/training/torchtitan/about-torchtitan-neuron`
