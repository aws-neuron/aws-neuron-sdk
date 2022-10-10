.. _neuron_cc:

Neuron Compiler
===============

The Neuron Compiler accepts Machine Learning models in various formats (TensorFlow, MXNet, PyTorch, XLA HLO) and optimizes them to run on Neuron devices.

The Neuron compiler is invoked within the ML framework, where ML models are sent to
the compiler by the Neuron Framework plugin. The resulting compiler artifact is called
a NEFF file (Neuron Executable File Format) that in turn is loaded by the Neuron runtime to the Neuron device.

.. tab-set::

   .. tab-item:: Neuron Compiler for Trn1

         .. dropdown::  API Reference Guide
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in

               .. toctree::
                  :maxdepth: 1

                  Neuron Compiler CLI Reference Guide </compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide>


         .. dropdown::  Developer Guide
                  :class-title: sphinx-design-class-title-med
                  :class-body: sphinx-design-class-body-small
                  :animate: fade-in


                  .. toctree::
                        :maxdepth: 1

                        /general/appnotes/neuronx-cc/neuronx-cc-training-mixed-precision


         .. dropdown::  
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in
               :open:

               .. toctree::
                  :maxdepth: 1

                  FAQ </compiler/neuronx-cc/faq>
                  What's New </release-notes/compiler/neuronx-cc/index>

   .. tab-item:: Neuron Compiler for Inf1


         .. dropdown::  API Reference Guide
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in

               .. toctree::
                  :maxdepth: 1

                  /compiler/neuron-cc/command-line-reference

         .. dropdown::  Developer Guide
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in


               .. toctree::
                  :maxdepth: 1

                  /general/appnotes/neuron-cc/mixed-precision


         .. dropdown::  
               :class-title: sphinx-design-class-title-med
               :class-body: sphinx-design-class-body-small
               :animate: fade-in
               :open:


               .. toctree::
                  :maxdepth: 1

                  FAQ </compiler/neuron-cc/faq>
                  What's New </release-notes/compiler/neuron-cc/neuron-cc>
                  /release-notes/compiler/neuron-cc/neuron-cc-ops/index
