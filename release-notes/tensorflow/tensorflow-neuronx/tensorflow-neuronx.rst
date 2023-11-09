.. _tensorflow-neuronx-release-notes:

TensorFlow Neuron (``tensorflow-neuronx``) Release Notes
========================================================

.. contents:: Table of contents
   :local:
   :depth: 1

This document lists the release notes for the tensorflow-neuronx 2.x packages.



tensorflow-neuronx 2.x release [2.1.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 09/15/2023

* Minor updates

Date: 05/1/2023

* Added support for tracing models larger than 2 GB through the environment variable ``NEURON_CC_FLAGS='--extract-weights INSTANCE_TYPE'`` for all trn1 and inf2 instance types.
* tensorflow-neuronx now supports tensorflow 2.7, 2.8, and 2.9 (In addition to the already supported 2.10).
* Neuron release 2.10 release will be the last release that will include support for tensorflow-neuronx version 2.7. Future Neuron releases will not include tensorflow-neuronx version 2.7.

tensorflow-neuronx 2.10 release [2.0.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 03/28/2023

The second release of tensorflow-neuronx 2.10 includes the following features:

* Dynamic batching

The following features are not included in this release:

* Support for tracing models larger than 2 GB

tensorflow-neuronx 2.10 release [1.0.0]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date: 2/24/2023

The initial release of tensorflow-neuronx 2.10 includes the following features:

* Initial support for TensorFlow 2.10 inference on Inf2 and Trn1
* Trace API (tensorflow_neuronx.trace)
* Automatic partitioning of model into CPU vs NeuronCore parts
* Automatic data parallel on multiple NeuronCores (beta)
* Python 3.7, 3.8 and 3.9 support
* HuggingFace Roberta tutorial

The following features are not included in this release:

* Dynamic batching
* Support for tracing models larger than 2 GB
