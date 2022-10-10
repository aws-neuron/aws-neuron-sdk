.. _neuron-setup-troubleshooting:

Neuron Setup Troubleshooting
============================

.. _contents::
   :local:
   :depth: 1

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

