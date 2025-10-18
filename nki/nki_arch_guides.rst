.. _nki_arch_guides:

NeuronDevice Architecture Guide for NKI
==========================================

NKI currently supports the following NeuronDevice generations:

* Trainium/Inferentia2, available on AWS ``trn1``, ``trn1n`` and ``inf2`` instances
* Trainium2, available on AWS ``trn2`` instances and UltraServers

The documents below provide an architecture deep dive of each NeuronDevice generation,
with a focus on areas that NKI developers can directly control through kernel implementation.
:doc:`Trainium/Inferentia2 Architecture Guide <arch/trainium_inferentia2_arch>`
serves as a foundational architecture guide for understanding basics of any
NeuronDevice generation, while :doc:`Trainium2 Architecture Guide <arch/trainium2_arch>`
walks through architecture enhancements compared to the previous generation in details.
Therefore, we suggest new NKI developers start with
:doc:`Trainium/Inferentia2 Architecture Guide <arch/trainium_inferentia2_arch>` before
exploring newer NeuronDevice architecture.


.. grid:: 2
      :margin: 4 1 0 0

      .. grid-item::

            .. card:: Trainium/Inferentia2 Architecture Guide
                  :link: trainium_inferentia2_arch
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small


      .. grid-item::

            .. card:: Trainium2 Architecture Guide
                  :link: trainium2_arch
                  :link-type: ref
                  :class-body: sphinx-design-class-title-small



.. toctree::
      :maxdepth: 1
      :hidden:

      arch/trainium_inferentia2_arch.rst
      arch/trainium2_arch.rst
