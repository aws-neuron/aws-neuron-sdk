.. meta::
   :description: Packages, libraries, and artifacts (with versions) that ship in the latest AWS Neuron SDK release for Trainium and Inferentia instances.
   :date-modified: 2026-10-19
   :keywords: AWS Neuron, release content, packages, versions, Trainium, Inferentia, vLLM Neuron

.. _latest-neuron-release-artifacts:

Release content
===============

This page contains the packages, libraries, and other artifacts (and the versions of them) that ship in the latest AWS Neuron SDK release.

.. contents:: Table of contents
   :local:
   :depth: 2

<< :ref:`Back to the release notes <latest-neuron-release>`


Neuron 2.33.0 (10/19/2026)
---------------------------

Trn1 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.33.0

Trn1 supported Python versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=trn1 --file=src/helperscripts/n2-manifest.json --neuron-version=2.33.0

Trn2 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.33.0

Trn2 supported Python versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=trn2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.33.0

Trn3 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=trn3 --file=src/helperscripts/n2-manifest.json --neuron-version=2.33.0

Trn3 supported Python versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=trn3 --file=src/helperscripts/n2-manifest.json --neuron-version=2.33.0

Inf2 packages
^^^^^^^^^^^^^

.. program-output:: python3 src/helperscripts/n2-helper.py --list=packages --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.33.0

Inf2 supported Python versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. program-output:: python3 src/helperscripts/n2-helper.py --list=pyversions --instance=inf2 --file=src/helperscripts/n2-manifest.json --neuron-version=2.33.0

Supported NumPy versions
^^^^^^^^^^^^^^^^^^^^^^^^

Neuron currently supports NumPy versions 2.X. Neuron continues to support NumPy versions >= 1.21.6, as well.

Supported vLLM versions
^^^^^^^^^^^^^^^^^^^^^^^

Neuron currently supports vLLM version 0.29.0. Both vLLM Neuron (``vllm-neuron`` 0.29.0.1.2.0) and vLLM Omni Neuron (``vllm-omni-neuron`` 0.29.0.0.2.0) are built on vLLM 0.29.0.

Supported Hugging Face Transformers versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------------------------+----------------------------------+
| Package                          | Supported Hugging Face           |
|                                  | Transformers versions            |
+==================================+==================================+
| torch-neuronx                    | >= 4.52                          |
+----------------------------------+----------------------------------+
| vllm                             | >= 4.56.0, < 5                   |
+----------------------------------+----------------------------------+

Supported Protobuf versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+----------------------------------+----------------------------------+
| Package                          | Supported Protobuf versions      |
+==================================+==================================+
| neuronx-cc                       | > 3                              |
+----------------------------------+----------------------------------+
| torch-neuronx                    | >= 3.20                          |
+----------------------------------+----------------------------------+
| jax-neuronx                      | >= 3.20                          |
+----------------------------------+----------------------------------+

Previous Neuron releases content
--------------------------------

* :ref:`pre-release-content`
* :ref:`pre-n1-release-content`
