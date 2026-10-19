.. _inference-torch-neuronx:


.. meta::
   :description: Inference with ``torch-neuronx`` (Inf2 & Trn1/Trn2) - AWS Neuron SDK documentation
   :keywords: AWS Neuron, Inferentia, PyTorch, Trainium, inference, torch-neuronx
   :date-modified: 2026-03-13


Inference with ``torch-neuronx`` (Inf2 & Trn1/Trn2)
====================================================

.. note:: Maintenance status

   The XLA-based ``torch-neuronx`` library documented on this page is in **maintenance
   status** as of Neuron SDK 2.32 — it continues to receive critical bug fixes and security
   patches, and this documentation remains available. It is **not** end-of-support. For new
   PyTorch work on Trainium, use :ref:`TorchNeuron, the native PyTorch backend
   <torch-neuron-home>`; the task-oriented :ref:`Training on Neuron <training-home>` page
   shows where it fits.

Deploy inference workloads using PyTorch NeuronX on Inf2, Trn1, and Trn2 instances.

.. toctree::
    :maxdepth: 1
    :hidden:

    Tutorials </archive/frameworks/torch/torch-neuronx/tutorials/inference/tutorials-torch-neuronx>
    Additional Examples </archive/frameworks/torch/torch-neuronx/additional-examples-inference-torch-neuronx>
    API Reference Guide </archive/frameworks/torch/torch-neuronx/api-reference-guide/inference/inference-api-guide-torch-neuronx>
    Developer Guide  </archive/frameworks/torch/torch-neuronx/programming-guide/inference/index>
    Misc  </archive/frameworks/torch/torch-neuronx/misc-inference-torch-neuronx>

Get Started
------------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Setup (``torch-neuronx``)
        :link: setup-torch-neuronx
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Install and configure PyTorch NeuronX for inference workloads on Inf2, Trn1, and Trn2 instances.

Tutorials
----------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Inference Tutorials
        :link: /archive/frameworks/torch/torch-neuronx/tutorials/inference/tutorials-torch-neuronx
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Step-by-step tutorials including BERT, TorchServe, LibTorch C++, ResNet50, and T5 inference.

Reference
----------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: API Reference Guide
        :link: /archive/frameworks/torch/torch-neuronx/api-reference-guide/inference/inference-api-guide-torch-neuronx
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Inference API reference for PyTorch NeuronX, including trace, replace weights, core placement, and data parallel APIs.

    .. grid-item-card:: Developer Guide
        :link: /archive/frameworks/torch/torch-neuronx/programming-guide/inference/index
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        In-depth developer guide covering core placement, trace vs XLA, data parallelism, and auto-bucketing.

Additional Resources
---------------------

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Additional Examples
        :link: /archive/frameworks/torch/torch-neuronx/additional-examples-inference-torch-neuronx
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        More inference examples and sample code from the AWS Neuron Samples repository.

    .. grid-item-card:: Misc
        :link: /archive/frameworks/torch/torch-neuronx/misc-inference-torch-neuronx
        :link-type: doc
        :class-header: sd-bg-primary sd-text-white

        Supported operators, release notes, and additional inference resources.
