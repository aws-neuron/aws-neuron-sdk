.. post:: March 12, 2026
    :language: en
    :tags: announce-correction-neuron-driver-inf1, neuron-driver-version, inf1

.. _announce-correction-neuron-driver-inf1-support:


Correction: Neuron Driver support for Inf1 — version 2.24 (not 2.21)
---------------------------------------------------------------------

We are correcting a previous announcement regarding last Neuron Driver version to support Inf1. The last supported version is 2.24

Neuron driver versions above 2.24 only support non-Inf1 instances (such as ``Trn1``, ``Inf2``, or other instance types).
For ``Inf1`` instance users, only Neuron driver version 2.24 will remain supported with regular security patches.

As part of this correction, Neuron Driver version **2.24.13.0** has been released as a patch for ``Inf1`` users, adding compatibility with Linux kernel 6.18.

``Inf1`` instance users are advised to pin the Neuron driver version to ``2.24.*`` in their installation script:

For Ubuntu:

.. code-block:: bash

    sudo apt-get install aws-neuronx-dkms=2.24.* -y

For Amazon Linux 2 / Amazon Linux 2023:

.. code-block:: bash

    sudo yum install aws-neuronx-dkms-2.24.* -y

Refer to the :ref:`Neuron Driver release notes <runtime_rn>` for more details.
