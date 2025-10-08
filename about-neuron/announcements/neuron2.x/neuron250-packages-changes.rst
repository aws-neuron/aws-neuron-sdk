.. post:: Nov 22, 2022 03:00
    :language: en
    :tags: neuron2.x

.. _neuron250-packages-changes:

Introducing Neuron packaging and installation changes for Inf1 customers
------------------------------------------------------------------------

Starting with :ref:`Neuron release 2.5 <neuron-2.5.0-whatsnew>`, Neuron introduces changes in Neuron packages and installation instructions for Inf1, the following  Neuron packages will change names: 


.. list-table:: Neuron package with changed names for Inf1
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size   

   * - New name
     - Old name (deprecated package)
     - Package Type
     - Description
     - Supported Instances 

   * - ``aws-neuronx-tools``
     - ``aws-neuron-tools``
     - .deb (apt), .rpm (yum)
     - System Tools
     - Trn1, Inf1

   * - ``aws-neuronx-dkms``
     - ``aws-neuron-dkms``
     - .deb (apt), .rpm (yum)
     - Neuron Driver
     - Trn1, Inf1     

   * - ``aws-neuronx-k8-plugin``
     - ``aws-neuron-k8-plugin``
     - .deb (apt), .rpm (yum)
     - Neuron Kubernetes plugin
     - Trn1, Inf1

   * - ``aws-neuronx-k8-scheduler``
     - ``aws-neuron-k8-scheduler``
     - .deb (apt), .rpm (yum)
     - Neuron Scheduler plugin
     - Trn1, Inf1

   * - ``tensorflow-model-server-neuronx``
     - ``tensorflow-model-server-neuron``
     - .deb (apt), .rpm (yum)
     - tensorflow-model-server
     - Trn1, Inf1


Please follow the :ref:`Neuron setup guide <setup-guide-index>` to update to latest Neuron releases.

