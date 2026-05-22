.. _neuron-tools-tutorials:

Tutorials
============

.. toctree::
    :hidden:
    :maxdepth: 1

    performance-profiling-vllm
    tutorial-neuron-monitor-mnist

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Profiling a vLLM Inference Workload
      :link: /tools/tutorials/performance-profiling-vllm
      :link-type: doc
      :class-card: sd-border-1

      Learn how to capture and analyze device-level and system-level profiles for vLLM inference workloads on AWS Trainium. 

   .. grid-item-card:: Profiling a NKI Kernel
      :link: /nki/guides/use-neuron-profile
      :link-type: doc
      :class-card: sd-border-1

      Learn how to profile a NKI kernel with Neuron Explorer.

   .. grid-item-card:: Track System Resource Utilization during Training with Neuron Monitor
      :link: tutorial-neuron-monitor-mnist
      :link-type: doc
      :class-card: sd-border-1

      Learn how to monitor resource utilization using neuron-monitor, Prometheus and Grafana while running a multi-layer perceptron MNIST model on Trainium using PyTorch Neuron.
