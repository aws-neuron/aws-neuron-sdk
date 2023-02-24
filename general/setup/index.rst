.. _setup-guide-index:

Setup Guide
===========
This section walks you through the various options to install Neuron. You have to install Neuron on Trainium and Inferentia powered instances to enable deep-learning acceleration. 

.. dropdown::  Launch the Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. include:: /general/setup/install-templates/launch-instance.txt

.. dropdown::  Install Drivers and Tools
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. tab-set::

       .. tab-item:: Amazon Linux 2

        .. include :: /src/helperscripts/installationScripts/python_instructions.txt
            :start-line: 2
            :end-line: 3

       .. tab-item:: Ubuntu 20

        .. include :: /src/helperscripts/installationScripts/python_instructions.txt
            :start-line: 5
            :end-line: 6


.. tab-set::


   .. tab-item:: Pytorch
        :name:

        .. dropdown::  torch-neuron (``Inf1``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in

                * :ref:`Fresh install <install-neuron-pytorch>`
                * :ref:`Update to latest release <update-neuron-pytorch>`
                * :ref:`Install previous releases <install-prev-neuron-pytorch>`
                * :ref:`pytorch-install-cxx11`

                .. include:: /general/setup/install-templates/trn1-ga-warning.txt

        .. dropdown::  torch-neuronx (``Trn1, Inf2``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in

                * :ref:`Fresh install <pytorch-neuronx-install>`
                * :ref:`Update to latest release <pytorch-neuronx-update>`
                * :ref:`Install previous releases <pytorch-neuronx-install-prev>`

                .. include:: /general/setup/install-templates/trn1-ga-warning.txt

   .. tab-item:: Tensorflow
        :name: 

        .. dropdown::  tensorflow-neuron (``Inf1``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in

                * :ref:`Fresh install <install-neuron-tensorflow>`
                * :ref:`Update to Latest release <update-neuron-tensorflow>`
                * :ref:`Install previous releases <install-prev-neuron-tensorflow>`

        .. dropdown::  tensorflow-neuronx (``Trn1, Inf2``)
                :class-title: sphinx-design-class-title-med
                :class-body: sphinx-design-class-body-small
                :animate: fade-in

                * :ref:`Fresh install <install-tensorflow-neuronx>`
                * :ref:`Update to Latest release <update-tensorflow-neuronx>`

   .. tab-item:: MXNet
        :name:

        .. dropdown::  mxnet-neuron (``Inf1``)
            :class-title: sphinx-design-class-title-med
            :class-body: sphinx-design-class-body-small
            :animate: fade-in

            * :ref:`Fresh install <install-neuron-mxnet>`
            * :ref:`Update to latest release <update-neuron-mxnet>`
            * :ref:`Install previous releases <install-prev-neuron-mxnet>`
