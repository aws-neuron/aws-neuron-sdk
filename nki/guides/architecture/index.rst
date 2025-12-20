.. meta::
    :description: NKI and Neuron Architectures.
    :keywords: NKI, AWS Neuron, Architecture, Trainium, trn1, trn2, trn3, inf2
    :date-modified: 12/14/2025

.. _nki-architecture-guides:

NKI and Neuron Architecture
----------------------------

NKI currently supports the following NeuronDevice generations:

* Trainium/Inferentia2, available on AWS ``trn1``, ``trn1n`` and ``inf2`` instances
* Trainium2, available on AWS ``trn2`` instances and UltraServers
* Trainium3, available on AWS ``trn3`` instances and UltraServers

The documents below provide an architecture deep dive of each NeuronDevice generation,
with a focus on areas that NKI developers can directly control through kernel implementation.

* :doc:`Trainium/Inferentia2 Architecture Guide </nki/guides/architecture/trainium_inferentia2_arch>` serves as a foundational architecture guide for understanding the basics of any NeuronDevice generation.
* :doc:`Trainium2 Architecture Guide </nki/guides/architecture/trainium2_arch>` walks through the architecture enhancements when compared to the previous generation.
* :doc:`Trainium3 Architecture Guide </nki/guides/architecture/trainium3_arch>` covers the enhancements for the next-generation Trainium ML accelerators.
  
Neuron recommends new NKI developers start with :doc:`Trainium/Inferentia2 Architecture Guide </nki/guides/architecture/trainium_inferentia2_arch>` before exploring newer NeuronDevice architecture.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Trainium/Inferentia2 Architecture Guide
      :link: trainium_inferentia2_arch
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Foundational architecture guide for understanding NeuronDevice basics.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Trainium2 Architecture Guide
      :link: trainium2_arch
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Architecture enhancements and improvements in the Trainium2 generation.

   .. grid-item-card:: Trainium3 Architecture Guide
      :link: trainium3_arch
      :link-type: ref
      :class-body: sphinx-design-class-title-small

      Latest architecture features and capabilities in Trainium3 devices.

.. toctree::
   :maxdepth: 1
   :hidden:

   Trainium/Inferentia2 Guide <trainium_inferentia2_arch>
   Trainium2 Guide <trainium2_arch>
   Trainium3 Guide <trainium3_arch>

