.. _neuron-tools-tutorials:

Tutorials
============

.. toctree::
    :hidden:
    :maxdepth: 1

    performance-profiling-vllm
    multinode-training-model-profiling
    torch-neuronx-profiling-with-tb
    tutorial-tensorboard-scalars-mnist
    tutorial-neuron-monitor-mnist

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Profiling Multi-Node Training Jobs
      :link: /tools/tutorials/multinode-training-model-profiling
      :link-type: doc
      :class-card: sd-border-1

      Learn how to analyze performance during multi-node training on AWS Trainium instances with SLURM job scheduling.

   .. grid-item-card:: Profiling a vLLM Inference Workload
      :link: /tools/tutorials/performance-profiling-vllm
      :link-type: doc
      :class-card: sd-border-1

      Learn how to capture and analyze device-level and system-level profiles for vLLM inference workloads on AWS Trainium. 

   .. grid-item-card:: Profiling a NKI Kernel
      :link: /nki/how-to-guides/use-neuron-profile
      :link-type: doc
      :class-card: sd-border-1

      Learn how to profile a NKI kernel with Neuron Explorer.

   .. grid-item-card:: Profiling PyTorch Neuron with TensorBoard
      :link: tutorial-tensorboard-scalars-mnist
      :link-type: doc
      :class-card: sd-border-1

      Learn how to use Neuron's plugin for TensorBoard that allows users to measure and visualize performance on a torch runtime level or an operator level.

   .. grid-item-card:: Track System Resource Utilization during Training with Neuron Monitor
      :link: tutorial-neuron-monitor-mnist
      :link-type: doc
      :class-card: sd-border-1

      Learn how to monitor resource utilization using neuron-monitor, Prometheus and Grafana while running a multi-layer perceptron MNIST model on Trainium using PyTorch Neuron.

   .. grid-item-card:: Track Training Progress in TensorBoard using PyTorch Neuron
      :link: torch-neuronx-profiling-with-tb
      :link-type: doc
      :class-card: sd-border-1

      Learn how to track training progress in TensorBoard while running a multi-layer perceptron MNIST model on Trainium using PyTorch Neuron.
