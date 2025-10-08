.. _neuron-setup-troubleshooting:

Neuron Setup Troubleshooting
============================

.. contents:: Table of contents
   :local:
   :depth: 2

.. _gpg_key_update:

How to update Neuron repository GNU Privacy Guard (GPG) key for Ubuntu installation
-----------------------------------------------------------------------------------

Description
^^^^^^^^^^^

The GPG key for the Neuron repository (https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB) is installed on the Ubuntu (Canonical) server, the key was uploaded originally with an expiry date of three (3) years, which has expired on 11/10/22.

Any customer of Ubuntu or Debian using Neuron ``apt`` repository will get the following error:

.. code::

   While running an apt-get update command on an AWS deep learning image (us-east-1/ami-01fce297f68912e45) I get this output:

   Err:6 https://apt.repos.neuron.amazonaws.com (https://apt.repos.neuron.amazonaws.com/) bionic InRelease
   The following signatures were invalid: EXPKEYSIG 5749CAD8646D9185 Amazon AWS Neuron <neuron-maintainers@amazon.com>
   Fetched 172 kB in 1s (161 kB/s)
   Reading package lists... Done
   W: An error occurred during the signature verification. The repository is not updated and the previous index files will be used. GPG error:https://apt.repos.neuron.amazonaws.com (https://apt.repos.neuron.amazonaws.com/) bionic InRelease: The following signatures were invalid: EXPKEYSIG 5749CAD8646D9185 Amazon AWS Neuron <neuron-maintainers@amazon.com>

Solution
^^^^^^^^

To solve this issue, you need to run the following commands to fetch the new key before running ``apt-get update``


.. code::

   wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

   # Update OS packages
   sudo apt-get update -y




``pip install --upgrade`` wouldn't upgrade ``neuron-cc``
--------------------------------------------------------

Description
^^^^^^^^^^^

When trying to upgrade to a newer Neuron release, for example by calling: 

``pip install --upgrade torch-neuron neuron-cc[tensorflow] torchvision``

``neuron-cc`` is not upgraded.

This can be a result of a bug in certain ``pip`` versions, for example `pip install upgrade will not upgrade package if extras_require specified <https://github.com/pypa/pip/issues/10173>`_

Solution
^^^^^^^^

To solve this issue you can either upgrade to a newer ``pip`` version or use ``--force`` when trying to upgrade, for example:

``pip install --force torch-neuron neuron-cc[tensorflow] torchvision``

