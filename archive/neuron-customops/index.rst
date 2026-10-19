.. meta::
   :robots: noindex, nofollow
   :description: Archived documentation for Neuron Custom C++ Operators, which run on the GPSIMD engines of Inf2 and Trn1 and are in maintenance mode starting with Neuron 2.33. Use NKI for new custom operators.
   :date-modified: 2026-10-18

:nosearch:

.. _neuron_c++customops:

Neuron Custom C++ Operators [Beta] (archived)
=============================================

.. warning::

   Neuron Custom C++ Operators are in maintenance mode starting with Neuron 2.33. The ``aws-neuronx-gpsimd-customop-lib`` library receives critical security and bug fixes only, and this documentation is archived and no longer updated. The feature runs only on NeuronCore-v2 (Inf2 and Trn1) through the PyTorch/XLA path. To implement new custom operators, use the :doc:`Neuron Kernel Interface (NKI) </nki/index>`. See :ref:`announce-maintenance-gpsimd-customop-lib-2-33`.


.. include:: /archive/neuron-customops/customops-intro.txt


.. note:: 

        Neuron Custom C++ Operators feature is currently supported on NeuronCore-v2 architecture only, which is found in Trainium (Trn1) and second-generation Inferentia (Inf2) chips.


.. toctree::
    :maxdepth: 1
    :hidden:

    /archive/neuron-customops/api-reference-guide/api-reference-guide


.. toctree::
    :maxdepth: 1
    :hidden:
      
    /archive/neuron-customops/programming-guide/programming-guide


.. toctree::
    :maxdepth: 1
    :hidden:

    /archive/neuron-customops/tutorials/tutorials


.. toctree::
    :maxdepth: 1
    :hidden:

    /archive/neuron-customops/misc-customops




.. dropdown::  API Reference Guide
      :class-title: sphinx-design-class-title-med
      :class-body: sphinx-design-class-body-small
      :animate: fade-in
      :open:

      * :ref:`custom-ops-api-ref-guide`       


.. dropdown::  Developer Guide
      :class-title: sphinx-design-class-title-med
      :class-body: sphinx-design-class-body-small
      :animate: fade-in
      :open:

      * :ref:`feature-custom-operators-devguide`


.. dropdown::  Tutorials
      :class-title: sphinx-design-class-title-med
      :class-body: sphinx-design-class-body-small
      :animate: fade-in
      :open:

      * :ref:`neuronx-customop-mlp-tutorial`
      * :ref:`neuronx-customop-mlp-perf`


.. dropdown::  Misc
      :class-title: sphinx-design-class-title-med
      :class-body: sphinx-design-class-body-small
      :animate: fade-in
      :open:

  
      * :ref:`gpsimd-customop-tools-rn`
      * :ref:`gpsimd-customop-lib-rn`











