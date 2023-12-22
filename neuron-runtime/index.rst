.. _neuron_runtime:

NeuronX Runtime
==============

NeuronX runtime consists of kernel driver and C/C++ libraries which provides APIs to access Inferentia and Trainium Neuron devices. The Neuron ML frameworks plugins for TensorFlow, PyTorch and Apache MXNet use the Neuron runtime to load and run models on the NeuronCores. Neuron runtime loads compiled deep learning models, also referred to as Neuron Executable File Format (NEFF) to the Neuron devices and is optimized for high-throughput and low-latency. 



.. toctree::
    :maxdepth: 1
    :hidden:

    /neuron-runtime/api-reference-guide


.. toctree::
    :maxdepth: 1
    :hidden:
      
    /neuron-runtime/configuration-guide


.. toctree::
    :maxdepth: 1
    :hidden:

    Misc </neuron-runtime/misc-runtime>


.. dropdown::  API Reference Guide
      :class-title: sphinx-design-class-title-med
      :class-body: sphinx-design-class-body-small
      :animate: fade-in
      :open:

      * :ref:`Runtime API <nrt-api-guide>`

.. dropdown::  Configuration Guide
      :class-title: sphinx-design-class-title-med
      :class-body: sphinx-design-class-body-small
      :animate: fade-in
      :open:

      * :ref:`Runtime Configuration <nrt-configuration>`


.. dropdown::  Misc
      :class-title: sphinx-design-class-title-med
      :class-body: sphinx-design-class-body-small
      :animate: fade-in
      :open:

      * :ref:`Troubleshooting on Inf1 and Trn1 <nrt-troubleshooting>`
      * :ref:`FAQ <neuron-runtime-faq>`
      * :ref:`neuron-runtime-rn`
      * :ref:`neuron-driver-release-notes`
      * :ref:`neuron-collectives-rn`









