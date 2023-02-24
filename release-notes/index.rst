.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.8.0-whatsnew:
.. _neuron-2.7.0-whatsnew:

Neuron 2.8.0 (02/24/2023)
-------------------------

.. contents:: Table of contents
   :local:
   :depth: 3

What's New
^^^^^^^^^^

This release adds support for `EC2 Inf2 <https://aws.amazon.com/ec2/instance-types/inf2/>`_ instances, introduces initial inference support with TensorFlow 2.x Neuron (``tensorflow-neuronx``) on Trn1 and Inf2, and introduces minor enhancements and bug fixes.

This release introduces the following:

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details

   * - Support for `EC2 Inf2 <https://aws.amazon.com/ec2/instance-types/inf2/>`_ instances
     - * Inference support for Inf2 instances in PyTorch Neuron (``torch-neuronx``)      
    
       * Inference support for Inf2 instances in TensorFlow 2.x Neuron (``tensorflow-neuronx``)
        
       * Overall documentation update to include Inf2 instances
  

   * - TensorFlow 2.x Neuron (``tensorflow-neuronx``) support
     - * This releases introduces initial inference support with TensorFlow 2.x Neuron (``tensorflow-neuronx``) on Trn1 and Inf2


   * - New Neuron GitHub samples
     - * New sample scripts for deploying LLM models with ``transformer-neuronx`` under       `aws-neuron-samples <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference>`_  GitHub repository.
      
       * New sample scripts for deploying models with ``torch-neuronx`` under `aws-neuron-samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx>`_  GitHub repository.

   * - Minor enhancements and bug fixes.
     - * See :ref:`components-rn`

   * - Release included packages
     - * see :ref:`neuron-release-content`

For more detailed release notes of the new features and resolved issues, see :ref:`components-rn`.


.. _components-rn:

Neuron Components Release Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inf1, Trn1 and Inf2 common packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size


   * - Component
     - Instance/s
     - Package/s
     - Details


   * - Neuron Runtime
     - Trn1, Inf1, Inf2
     - * Trn1: ``aws-neuronx-runtime-lib`` (.deb, .rpm)

       * Inf1: Runtime is linked into the ML frameworks packages
       
     - * :ref:`neuron-runtime-rn`

   * - Neuron Runtime Driver
     - Trn1, Inf1, Inf2
     - * ``aws-neuronx-dkms``  (.deb, .rpm)
       
     - * :ref:`neuron-driver-release-notes`

   * - Neuron System Tools
     - Trn1, Inf1, Inf2
     - * ``aws-neuronx-tools``  (.deb, .rpm)
     - * :ref:`neuron-tools-rn`



   * - Containers
     - Trn1, Inf1, Inf2
     - * ``aws-neuronx-k8-plugin`` (.deb, .rpm)

       * ``aws-neuronx-k8-scheduler`` (.deb, .rpm)
       
       * ``aws-neuronx-oci-hooks`` (.deb, .rpm)

     - * :ref:`neuron-k8-rn`

       * :ref:`neuron-containers-release-notes`

   * - NeuronPerf (Inference only)
     - Trn1, Inf1, Inf2
     - * ``neuronperf`` (.whl)
     - * :ref:`neuronperf_rn`


   * - TensorFlow Model Server Neuron
     - Trn1, Inf1, Inf2
     - * ``tensorflow-model-server-neuronx`` (.deb, .rpm)
     - * :ref:`tensorflow-modeslserver-neuronx-rn`


Trn1 and Inf2 only packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   
   * - Component
     - Instance/s
     - Package/s
     - Details



   * - PyTorch Neuron
     - Trn1, Inf2
     - * ``torch-neuronx`` (.whl)
     - * :ref:`torch-neuronx-rn`

       * :ref:`pytorch-neuron-supported-operators`
       

   * - TensorFlow Neuron
     - Trn1, Inf2
     - * ``tensorflow-neuronx`` (.whl)
     - * :ref:`tensorflow-neuronx-release-notes`


   * - Neuron Compiler (Trn1 only)
     - Trn1, Inf2
     - * ``neuronx-cc`` (.whl)
     - * :ref:`neuronx-cc-rn`

   * - Collective Communication library
     - Trn1, Inf2
       
     - * ``aws-neuronx-collective`` (.deb, .rpm)

     - * :ref:`neuron-collectives-rn`


   * - Neuron Custom C++ Operators
     - Trn1, Inf2
  
     - * ``aws-neuronx-gpsimd-customop`` (.deb, .rpm)
  
       * ``aws-neuronx-gpsimd-tools`` (.deb, .rpm)
  
     - * :ref:`gpsimd-customop-lib-rn`

       * :ref:`gpsimd-customop-tools-rn`


   * - ``transformers-neuronx``
     - Trn1, Inf2
       
     - * GitHub repository `(link) <https://github.com/aws-neuron/transformers-neuronx>`_

     - * `Release Notes <https://github.com/aws-neuron/transformers-neuronx/blob/master/releasenotes.md>`_


.. note::

   In next releases ``aws-neuronx-tools`` and ``aws-neuronx-runtime-lib`` will add support for Inf1.


Inf1 only packages
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   

   * - Component
     - Instance/s
     - Package/s
     - Details


   * - PyTorch Neuron
     - Inf1
     - * ``torch-neuron`` (.whl)
     - * :ref:`pytorch-neuron-rn`

       * :ref:`neuron-cc-ops-pytorch`


   * - TensorFlow Neuron
     - Inf1
     - * ``tensorflow-neuron`` (.whl)
     - * :ref:`tensorflow-neuron-rn`

       * :ref:`neuron-cc-ops-tensorflow`



   * - Apache MXNet (Incubating)
     - Inf1
     - * ``mx_neuron`` (.whl)
     - * :ref:`mxnet-neuron-rn`

       * :ref:`neuron-cc-ops-mxnet`


   * - Neuron Compiler (Inf1 only)
     - Inf1
     - * ``neuron-cc`` (.whl)
     - * :ref:`neuron-cc-rn`

       * :ref:`neuron-supported-operators`


Release Content
~~~~~~~~~~~~~~~

* :ref:`neuron-release-content`


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`prev-n1-rn`

