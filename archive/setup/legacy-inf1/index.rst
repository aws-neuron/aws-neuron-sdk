.. meta::
   :noindex:
   :nofollow:
   :description: Archived legacy installation guide for AWS Inferentia 1 (Inf1) instances
   :keywords: neuron, inf1, legacy, installation, inferentia
   :instance-types: inf1
   :status: legacy
   :content-type: legacy-guide
   :date-modified: 2026-08-04

.. _legacy-inf1:

Inf1 installation (legacy) — Archived
======================================

.. warning::

   This page is archived and no longer maintained. It is provided for
   reference only.

   **Legacy hardware**: Inf1 instances use NeuronCore v1 architecture.
   
   **For new projects, use Inf2, Trn1, Trn2, or Trn3 instances** with NeuronCore v2 for:
   
   - 3x better price-performance than Inf1
   - Broader framework support (PyTorch 2.x, JAX)
   - Active development and feature updates
   - Latest Neuron SDK features
   
   See :ref:`setup-guide-index` for current instance options.

.. admonition:: When to use Inf1
   :class: tip
   
   Use Inf1 only if you:
   
   - Maintain existing Inf1 deployments
   - Have compiled models for NeuronCore v1
   - Require specific Inf1 cost optimization for inference workloads

Migration to Inf2
-----------------

Consider migrating to Inf2 for better performance and support:

- Inf2 offers 3x better price-performance
- Broader framework support including PyTorch 2.x and JAX
- Active development with monthly SDK releases
- See :ref:`setup-guide-index` for current installation options

Choose your framework
---------------------

.. note::
   
   JAX is not supported on Inf1 instances. Use Inf2, Trn1, Trn2, or Trn3 for JAX workloads.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: PyTorch (Inf1)
      :link: pytorch
      :link-type: doc
      :class-card: sd-border-2
      
      PyTorch 1.x with torch-neuron
      
      Inference on Inf1 instances using NeuronCore v1
      
      :bdg-warning:`Legacy`

   .. grid-item-card:: TensorFlow (Inf1)
      :link: /archive/tensorflow/setup-legacy-inf1-tensorflow
      :link-type: doc
      :class-card: sd-border-2
      
      TensorFlow 2.x with tensorflow-neuron (archived)
      
      :bdg-danger:`Archived`

Additional resources
--------------------

- :doc:`/setup/torch-neuron` - Original PyTorch Neuron setup (Inf1)
- :doc:`/archive/tensorflow/tensorflow-neuron-inference` - TensorFlow Neuron inference (Inf1)
- :doc:`/release-notes/index` - Version compatibility

.. toctree::
   :hidden:
   :maxdepth: 1
   
   pytorch
