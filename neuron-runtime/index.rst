.. _neuron_runtime:

Neuron Runtime
==============

Neuron runtime consists of kernel driver and C/C++ libraries which provides APIs to access Inferentia and Trainium Neuron devices. The Neuron ML frameworks plugins for TensorFlow, PyTorch and Apache MXNet use the Neuron runtime to load and run models on the NeuronCores. Neuron runtime loads compiled deep learning models, also referred to as Neuron Executable File Format (NEFF) to the Neuron devices and is optimized for high-throughput and low-latency. 


.. dropdown::  API Reference Guide
      :class-title: sphinx-design-class-title-med
      :class-body: sphinx-design-class-body-small
      :animate: fade-in
      :open:

      .. toctree::
         :maxdepth: 1

         Runtime Configuration </neuron-runtime/nrt-configurable-parameters>


.. dropdown::  
      :class-title: sphinx-design-class-title-med
      :class-body: sphinx-design-class-body-small
      :animate: fade-in
      :open:

      .. toctree::
         :maxdepth: 1

         Troubleshooting on Trn1 </neuron-runtime/nrt-troubleshoot-trn1>
         Troubleshooting on Inf1 </neuron-runtime/nrt-troubleshoot>
         FAQ </neuron-runtime/faq>
         /release-notes/runtime/aws-neuronx-runtime-lib/index
         /release-notes/runtime/aws-neuronx-dkms/index
         /release-notes/runtime/aws-neuronx-collectives/index       









