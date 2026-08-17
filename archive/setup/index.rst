.. meta::
   :noindex:
   :nofollow:
   :description: Archived AWS Neuron SDK setup and installation guides.
   :keywords: AWS Neuron SDK, archived setup, legacy installation, previous releases
   :date-modified: 2026-08-04

.. _archive-setup-index:

Archived setup guides
=====================

.. warning::

   The pages listed here are archived and no longer maintained. They cover
   unsupported install targets (for example, Ubuntu 20.04, Amazon Linux 2,
   Rocky Linux 9) or superseded Neuron SDK releases. They are provided for
   reference only and may not work with current Neuron releases.

For current installation and setup instructions, see
:doc:`Set up environments </setup/index>`.

Legacy install targets
-----------------------

.. list-table::
   :header-rows: 1

   * - Page
     - Reason archived
     - Date archived
   * - :doc:`setup-rocky-linux-9`
     - Rocky Linux 9 is no longer a supported install target
     - Archived on: 5/15/2026
   * - :doc:`pytorch-install-prev-u20`
     - Ubuntu 20.04 is no longer a supported install target
     - Archived on: 8/4/2026
   * - :doc:`pytorch-install-prev-al2`
     - Amazon Linux 2 is no longer a supported install target
     - Archived on: 8/4/2026
   * - :doc:`Inf1 installation (legacy) <legacy-inf1/index>`
     - Inf1 (NeuronCore v1) is legacy hardware; ``torch-neuron`` is superseded by ``torch-neuronx``
     - Archived on: 8/4/2026

Previous-release PyTorch NeuronX installs
------------------------------------------

.. list-table::
   :header-rows: 1

   * - Page
     - Last release covered
     - Date archived
   * - :doc:`neuronx-2.9.0-pytorch-install`
     - Neuron 2.9.0
     - Archived on: 8/4/2026
   * - :doc:`neuronx-2.8.0-pytorch-install`
     - Neuron 2.8.0
     - Archived on: 8/4/2026
   * - :doc:`neuronx-2.7.0-pytorch-install`
     - Neuron 2.7.0
     - Archived on: 8/4/2026

.. toctree::
   :maxdepth: 1
   :hidden:

   setup-rocky-linux-9
   legacy-inf1/index
   pytorch-install-prev-u20
   pytorch-install-prev-al2
   neuronx-2.9.0-pytorch-install
   neuronx-2.8.0-pytorch-install
   neuronx-2.7.0-pytorch-install
