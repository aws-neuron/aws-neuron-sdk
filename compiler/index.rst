.. _neuron_cc:

Neuron Compiler
===============

The Neuron Compiler accepts Machine Learning models in various formats (TensorFlow, MXNet, PyTorch, XLA HLO) and optimizes them to run on Neuron devices.

The Neuron compiler is invoked within the ML framework, where ML models are sent to
the compiler by the Neuron Framework plugin. The resulting compiler artifact is called
a NEFF file (Neuron Executable File Format) that in turn is loaded by the Neuron runtime to the Neuron device.


.. toctree::
    :maxdepth: 1
    :hidden:

    /compiler/neuronx-cc


.. toctree::
    :maxdepth: 1
    :hidden:

    /compiler/neuron-cc




.. tab-set::

   .. tab-item:: Neuron Compiler for Trn1 & Inf2

         .. dropdown::  API Reference Guide
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in

               * :ref:`Neuron Compiler CLI Reference Guide <neuron-compiler-cli-reference-guide>`



         .. dropdown::  Developer Guide
                  :class-title: sphinx-design-class-title-med
                  :class-body: sphinx-design-class-body-small
                  :animate: fade-in


                  * :ref:`neuronx-cc-training-mixed-precision`


         .. dropdown::  Misc
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in
               :open:

               * :ref:`FAQ <neuronx_compiler_faq>`
               * :ref:`What's New <neuronx-cc-rn>`

   .. tab-item:: Neuron Compiler for Inf1


         .. dropdown::  API Reference Guide
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in

               * :ref:`neuron-compiler-cli-reference`


         .. dropdown::  Developer Guide
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in


               * :ref:`neuron-cc-training-mixed-precision`



         .. dropdown::  Misc
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in
               :open:

               * :ref:`FAQ <neuron_compiler_faq>`
               * :ref:`What's New <neuron-cc-rn>`
               * :ref:`neuron-supported-operators`
         