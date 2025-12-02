.. _neuron_runtime:

NeuronX Runtime
================

The NeuronX Runtime consists of a kernel driver and C/C++ libraries which provides APIs to access Inferentia and Trainium Neuron devices. The Neuron ML frameworks plugins for TensorFlow, PyTorch and Apache MXNet use the Neuron runtime to load and run models on the NeuronCores. Neuron runtime loads compiled deep learning models, also referred to as Neuron Executable File Format (NEFF) to the Neuron devices and is optimized for high-throughput and low-latency. 

.. toctree::
    :maxdepth: 2
    :hidden:

    Overview </neuron-runtime/about/index>
    Get Started </neuron-runtime/about/core-dump>
    Deep Dives </neuron-runtime/explore/index>
    /neuron-runtime/configuration-guide
    /neuron-runtime/api-reference-guide
    Runtime API <nrt-api-guide>
    NRT Debug Stream </neuron-runtime/api/debug-stream-api>
    Resources </neuron-runtime/misc-runtime>

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

    .. grid-item-card:: Runtime API Reference
        :link: nrt-api-guide
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Comprehensive guide to the Neuron Runtime API for developers building custom frameworks that call libnrt APIs directly.

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

    .. grid-item-card:: Runtime Configuration
        :link: nrt-configuration
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

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

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Runtime Release Notes
        :link: neuron-runtime-rn
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Latest updates, improvements, and bug fixes for the Neuron Runtime library.

    .. grid-item-card:: Driver Release Notes
        :link: neuron-driver-release-notes
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Release notes for the Neuron kernel driver that enables access to Neuron devices.

    .. grid-item-card:: Collectives Release Notes
        :link: neuron-collectives-rn
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Release notes for the Neuron Collective Communication Library used for distributed training and inference.
