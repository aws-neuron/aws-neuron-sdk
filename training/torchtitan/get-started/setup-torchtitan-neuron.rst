.. meta::
   :description: How to install and configure TorchTitan on AWS Neuron for pre-training on Trainium with TorchNeuron.
   :keywords: TorchTitan, TorchNeuron, AWS Neuron, Trainium, setup, install, configuration
   :date-modified: 2026-10-18

.. _torchtitan-neuron-setup:

===========================================
How to set up TorchTitan Neuron
===========================================

Task overview
-------------

This topic shows how to install TorchTitan on an AWS Trainium instance and
verify a minimal training run with TorchNeuron. It covers the supported install
path and a verification run you can use before working through the training
tutorials.

Prerequisites
-------------

- **Trainium instance:** Access to a Trn2 or Trn3 instance.
- **Neuron SDK 2.33+ installed and working.**
- **PyTorch 2.14 with torch-neuronx verified:** Run
  :doc:`/torch/get-started/quickstart-torch-neuron` and confirm the
  verification script passes before continuing. In particular, confirm
  ``device='neuron'`` resolves in a plain PyTorch script.

Instructions
------------

**1:** Install TorchTitan on the Trainium instance

Clone the Neuron fork of TorchTitan and install it, along with its
dependencies, into your Python environment:

.. code-block:: bash

   git clone https://github.com/aws-neuron/torchtitan-neuron.git
   cd torchtitan-neuron
   pip install -r requirements.txt
   pip install -e .

The ``requirements.txt`` install pulls in TorchTitan's dependencies; the
editable install (``-e .``) makes the ``torchtitan`` package importable while
leaving the source tree editable.

**2:** Verify the installation with a minimal run

Run the shipped ``qwen3_8b_tp4_fsdp16_sac`` config for a few steps to confirm
the install trains end-to-end. Override the tokenizer and dataset to the
bundled test tokenizer (``tests/assets/tokenizer``) and the small ``c4_test``
split shipped in the repository, so this step needs no model-asset or dataset
download. ``run_train.sh`` wraps ``torchrun``; select the model and config
through the ``MODULE`` and ``CONFIG`` environment variables, set the NeuronCore
count with ``NGPU``, and shorten the run with ``--training.steps``:

.. code-block:: bash

   NGPU=64 MODULE=torchtitan_neuron.models.qwen3 CONFIG=qwen3_8b_tp4_fsdp16_sac \
   ./run_train.sh --training.steps 10 --metrics.log_freq 1 \
     --dataloader.dataset c4_test --hf_assets_path=./tests/assets/tokenizer

This is the same model and config as the basic training tutorial (see
:doc:`/training/torchtitan/tutorials/tutorial-basic-training`), run for only a
few steps here to confirm the install works. It uses all 64 NeuronCores on a
``trn2.48xlarge`` (TP4 × FSDP16). The tutorial covers training with the real
Qwen3-8B tokenizer and C4 dataset.

Confirm your work
-----------------

To confirm TorchTitan is installed and configured correctly:

- The import ``import torchtitan`` succeeds in Python.
- The verification run from Step 2 begins stepping and completes the requested
  number of steps.
- Per-step loss lines appear and loss values are finite (not ``NaN``). A healthy
  step line looks like this (values will vary):

  .. code-block:: text

     step: 10  loss:  6.95994  grad_norm: 28.3022  memory:  9.14GiB(38.10%)  tps: 496  tflops: 26.13  mfu: 16.54%

Common issues
-------------

Uh oh! Did you encounter an error or other issue while working through this
task? Here are some commonly encountered issues and how to address them.

.. rubric:: ``import torchtitan`` fails

- **Possible solution:** Confirm the editable install (``pip install -e .``)
  completed and that you are running in the same Python environment TorchTitan
  was installed into.

.. rubric:: A dependency is missing at import time

- **Possible solution:** Re-run ``pip install -r requirements.txt``
  from the repository root. The requirements file lists all of TorchTitan's
  dependencies.

.. rubric:: Verification run can't find the Neuron device

- **Possible solution:** Confirm torch-neuronx is installed and
  ``device='neuron'`` resolves in a plain PyTorch script first. If it does not,
  fix the TorchNeuron install before returning here.

Related information
-------------------

- :doc:`/training/torchtitan/about-torchtitan-neuron` — Conceptual overview.
- :doc:`/training/torchtitan/tutorials/tutorial-basic-training` — Your first
  end-to-end training run, which builds on this setup.
- :doc:`/training/torchtitan/tutorials/index` — Training tutorials for the
  supported model families.
- :doc:`/torch/get-started/quickstart-torch-neuron` — Verify the underlying
  TorchNeuron install before running TorchTitan.
