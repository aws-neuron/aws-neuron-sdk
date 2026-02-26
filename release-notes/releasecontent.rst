
.. _latest-neuron-release-artifacts:

Release Content
===============

This page contains the packages, libraries, and other artifacts (and the versions of them) that ship in the latest AWS Neuron SDK release.

.. contents:: Table of contents
   :local:
   :depth: 2

<< :ref:`Back to the release notes <latest-neuron-release>`

Neuron 2.28.0 (02/26/2026)
---------------------------

Trn1 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.28.0

Trn2 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.28.0


Inf2 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.28.0

Inf1 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.28.0

Supported Python Versions for Inf1 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.28.0

Supported Python Versions for Inf2/Trn1/Trn2 packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.28.0

Supported NumPy Versions
^^^^^^^^^^^^^^^^^^^^^^^^

Neuron currently supports NumPy versions 2.X. Neuron continues to support NumPy versions >= 1.21.6, as well.

Supported vLLM Versions
^^^^^^^^^^^^^^^^^^^^^^^

Neuron currently supports vLLM version 0.13.0.

Supported Hugging Face Transformers Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
+----------------------------------+----------------------------------+
| Package                          | Supported Hugging Face           |
|                                  | Transformers Versions            |
+==================================+==================================+
| torch-neuronx                    | >= 4.52                          |
+----------------------------------+----------------------------------+
| neuronx-distributed-inference    | >= 4.57                          |
+----------------------------------+----------------------------------+
| vllm                             | >= 4.56.0, < 5                   |
+----------------------------------+----------------------------------+

Supported Protobuf Versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+----------------------------------+----------------------------------+
| Package                          | Supported Protobuf versions      |
+==================================+==================================+
| neuronx-cc                       | > 3                              |
+----------------------------------+----------------------------------+
| torch-neuronx                    | >= 3.20                          |
+----------------------------------+----------------------------------+
| torch-neuron                     | < 3.20                           |
+----------------------------------+----------------------------------+
| neuronx-distributed              | >= 3.20                          |
+----------------------------------+----------------------------------+
| tensorflow-neuronx               | < 3.20                           |
+----------------------------------+----------------------------------+
| tensorflow-neuron                | < 3.20                           |
+----------------------------------+----------------------------------+
  
Previous Neuron Releases Content
--------------------------------

* :ref:`pre-release-content`
* :ref:`pre-n1-release-content`
