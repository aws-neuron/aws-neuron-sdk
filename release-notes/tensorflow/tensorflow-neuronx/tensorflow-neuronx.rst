.. _tensorflow-neuronx-release-notes:

TensorFlow Neuron (``tensorflow-neuronx``) Release Notes
========================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for the tensorflow-neuronx 2.x packages.

.. _tfx-known-issues-and-limitations:

Known Issues and Limitations - updated 02/24/2023

tensorflow-neuronx 2.10 release [2.10.1.2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 03/28/2023

The second release of tensorflow-neuronx 2.10 includes the following features:

* Dynamic batching

The following features are not included in this release:

* Support for tracing models larger than 2 GB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tensorflow-neuronx 2.10 release [2.10.1.1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 2/24/2023

The initial release of tensorflow-neuronx 2.10 includes the following features:

* Initial support for TensorFlow 2.10 inference on Inf2 and Trn1
* Trace API (tensorflow_neuronx.trace)
* Automatic partitioning of model into CPU vs NeuronCore parts
* Automatic data parallel on multiple NeuronCores (experimental)
* Python 3.7, 3.8 and 3.9 support
* HuggingFace Roberta tutorial

The following features are not included in this release:

* Dynamic batching
* Support for tracing models larger than 2 GB
