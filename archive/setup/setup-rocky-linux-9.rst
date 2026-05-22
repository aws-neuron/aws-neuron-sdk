.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-05-15

.. _setup-rocky-linux-9:

PyTorch Neuron setup for Rocky Linux 9 — Archived
==================================================

.. warning::

   This document is archived. Rocky Linux 9 is no longer a supported install
   target for the AWS Neuron SDK. It is provided here for reference only.

The original page documented installing PyTorch Neuron (``torch-neuronx`` and
``torch-neuron``) on the Rocky-9-EC2-Base AMI. The flow installed the Neuron
driver and tools using the Rocky Linux 9 package set, then deferred to the
Amazon Linux 2023 guide for EFA and PyTorch Neuron installation.

For current setup paths, see:

- :doc:`Set up environments </setup/index>` for the supported install matrix
  (Ubuntu 22.04, Ubuntu 24.04, Amazon Linux 2023).
- :doc:`Install PyTorch Neuron manually </setup/pytorch/manual>` for the
  current PyTorch (``torch-neuronx``) install path.
- :doc:`Install PyTorch Neuron for Inf1 (legacy) </setup/legacy-inf1/pytorch>`
  for the legacy ``torch-neuron`` (Inf1) install path.
