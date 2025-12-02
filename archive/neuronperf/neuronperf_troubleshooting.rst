.. _neuronperf_troubleshooting:

.. meta::
   :noindex:
   :nofollow:
   :description: This tutorial for the AWS Neuron SDK is currently archived and not maintained. It is provided for reference only.
   :date-modified: 12-02-2025

NeuronPerf Troubleshooting
==========================

.. contents:: Table of contents
   :local:
   :depth: 2

Compilation issues
^^^^^^^^^^^^^^^^^^

Model fails to compile
~~~~~~~~~~~~~~~~~~~~~~

Please `file a bug <https://github.com/aws/aws-neuron-sdk/issues>`_ with as much information as possible.

Benchmarking Issues
^^^^^^^^^^^^^^^^^^^

Benchmarking terminates early with errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Scroll up and read the output. Most likely causes are:
   - invalid input shapes or
   - not enough memory to load the requested number of model copies on the device. Try passing ``n_models=1`` to ``benchmark`` again to test for memory issues.

Other Issues or Feature Requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please file a bug on `Github <https://github.com/aws/aws-neuron-sdk/issues>`_.