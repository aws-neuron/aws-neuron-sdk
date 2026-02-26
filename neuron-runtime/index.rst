.. _neuron_runtime:

NeuronX Runtime
================

The NeuronX Runtime is a high-performance execution engine that enables deep learning models to run on AWS Inferentia and Trainium accelerators. It consists of a kernel driver and C/C++ libraries that provide low-level APIs for accessing Neuron devices, managing model execution, and coordinating collective communications across NeuronCores.

The Neuron Runtime serves as the foundation for all ML framework integrations (TensorFlow, PyTorch, JAX, and Apache MXNet), loading compiled models in Neuron Executable File Format (NEFF) and orchestrating their execution on Neuron hardware. It is optimized for high-throughput and low-latency inference and training workloads, with features including:

* **Efficient model execution**: Loads and executes NEFF files on NeuronCores with optimized memory management
* **Multi-model support**: Manages multiple models across multiple NeuronCores with flexible allocation strategies
* **Collective communications**: Provides high-performance collective operations for distributed training and inference
* **Device management**: Handles NeuronCore allocation, device discovery, and resource management
* **Debugging support**: Offers core dump generation, debug streams, and detailed logging for troubleshooting
* **Configuration flexibility**: Extensive environment variables for fine-tuning runtime behavior

The Neuron Runtime is typically used transparently through ML framework plugins, but also provides direct C/C++ APIs for developers building custom frameworks or requiring low-level device control. 

.. toctree::
    :maxdepth: 2
    :hidden:

    Overview </neuron-runtime/about/index>
    Get Started </neuron-runtime/about/core-dump>
    Deep Dives </neuron-runtime/explore/index>
    /neuron-runtime/configuration-guide
    Developer Guide </neuron-runtime/nrt-developer-guide>
    API Reference </neuron-runtime/api/index>
    NRT Debug Stream </neuron-runtime/api/debug-stream-api>
    Troubleshooting on Inf1 and Trn1 </neuron-runtime/nrt-troubleshoot>
    Release Notes </release-notes/components/runtime>
    FAQ </neuron-runtime/faq>

Get Started
------------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: About the NeuronX Runtime
        :link: neuron-runtime-about
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Learn about the AWS Neuron Runtime, its features, and capabilities for accessing Inferentia and Trainium Neuron devices.

    .. grid-item-card:: Quickstart: Generate a Core Dump
        :link: runtime-core-dump-quickstart
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Learn how to generate a Neuron runtime core dump for debugging runtime failures and analyzing device state.

Reference
------------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Runtime Developer Guide
        :link: nrt-api-guide
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Comprehensive guide to the Neuron Runtime API for developers building custom frameworks that call libnrt APIs directly.

    .. grid-item-card:: Runtime API Reference Documentation
        :link: /neuron-runtime/api/index
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Documentation of the APIs in the public headers for the Neuron Runtime.

    .. grid-item-card:: Runtime Configuration
        :link: nrt-configuration
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Learn how to configure the Neuron Runtime using environment variables to control NeuronCore allocation, logging, and more.

    .. grid-item-card:: Troubleshooting on Inf1 and Trn1
        :link: nrt-troubleshooting
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Solutions for common issues encountered when using the Neuron Runtime on Inferentia and Trainium instances.

    .. grid-item-card:: Frequently Asked Questions
        :link: neuron-runtime-faq
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Answers to common questions about the Neuron Runtime, including compatibility, configuration, and usage.

Learn More
------------

.. grid:: 1
    :gutter: 2

    .. grid-item-card:: Explore the Neuron Runtime
        :link: neuron-runtime-explore-home
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Deep dives into the Neuron Runtime, including NEFF files, compute-communication overlap, device memory, and core dumps.

Collectives
------------

.. grid:: 1
    :gutter: 2

    .. grid-item-card:: About Collectives
        :link: /neuron-runtime/about/collectives
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Learn about Neuron Runtime collectives.

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Deep Dive: Inter-node Collective Communication
        :link: /neuron-runtime/explore/internode-collective-comm
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Explore and understand techniques for communication across nodes in the Neuron Runtime.

    .. grid-item-card:: Deep dive: Intra-node Collective Communication
        :link: /neuron-runtime/explore/intranode-collective-comm
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Explore and understand techniques for communication within nodes in the Neuron Runtime.

Release Notes
--------------

.. grid:: 1
    :gutter: 2

    .. grid-item-card:: Runtime Release Notes
        :link: /release-notes/components/runtime
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Latest updates, improvements, and bug fixes for the Neuron Runtime library, driver, and collectives.

