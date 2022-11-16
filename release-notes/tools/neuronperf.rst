.. _neuronperf_rn:

NeuronPerf 1.x Release Notes
============================

:ref:`NeuronPerf <neuronperf>` is a lightweight Python library with a simple API that enables fast measurements of performance when running models using Neuron.


.. contents:: Table of Contents
   :local:
   :depth: 1


NeuronPerf release [1.6.0.0]
----------------------------

Date: 11/23/2022

* New Evaluation + metrics API (see NeuronPerf Evaluation Guide)
* Support map and iterable-type torch datasets
* Support custom torch DataLoader args via dataloader_kwargs
* New get_report_by_tag utility to identify specific configurations
* Python 3.7+ now default from 3.6
* Pricing and sizing info updated for inf1 + trn1

Bug fixes

* GPU inputs are now moved correctly


NeuronPerf release [1.3.0.0]
----------------------------

Date: 04/29/2022


* Minor updates

NeuronPerf release [1.2.0.0]
----------------------------

Date: 03/25/2022


* Initial release of NeuronPerf
* Supports PyTorch, TensorFlow, and Apache MXNet (Incubating).
* Supports customizable JSON and CSV reports
* See :ref:`neuronperf` for more information.
