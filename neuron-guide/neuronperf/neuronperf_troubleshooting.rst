.. _neuronperf_troubleshooting:

NeuronPerf Troubleshooting
==========================

**Compilation issues**
   * Model fails to compile.
      - Please `file a bug <https://github.com/aws/aws-neuron-sdk/issues>`_ with as much information as possible.

**Benchmarking Issues**
   * Benchmarking terminates early with errors
      - Scroll up and read the output. Most likely causes are:
         - invalid input shapes or
         - not enough memory to load the requested number of model copies on the device. Try passing ``n_models=1`` to ``benchmark`` again to test for memory issues.

Other Issues or Feature Requests
--------------------------------

Please file a bug on `Github <https://github.com/aws/aws-neuron-sdk/issues>`_.