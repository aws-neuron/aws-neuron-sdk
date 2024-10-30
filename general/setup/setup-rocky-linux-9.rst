.. _setup-rocky-linux-9:

.. card:: Select a Different Platform for Setup
    :link: setup-guide-index
    :link-type: ref
    :class-body: sphinx-design-class-title-small

PyTorch Neuron Setup Guide for Rocky Linux 9
============================================


.. contents:: Table of contents
    :local:
    :depth: 1

Get Started with Latest Release of PyTorch Neuron 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section provides links that will assist you to quickly start with a fresh installation of PyTorch Neuron (``torch-neuronx`` , ``torch-neuron``).


.. dropdown:: Launch the Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    * Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to launch an instance. When choosing the instance type at the EC2 console, please make sure to select the correct instance type.
    * To get more information about instances sizes and pricing see: `Trn1 web page <https://aws.amazon.com/ec2/instance-types/trn1/>`_, `Inf2 web page <https://aws.amazon.com/ec2/instance-types/inf2/>`_
    * Select Rocky-9-EC2-Base AMI
    * When launching a Trn1, please adjust your primary EBS volume size to a minimum of 512GB.
    * After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_ to connect to the instance

.. dropdown:: Install Drivers and Tools
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. include:: /src/helperscripts/installationScripts/python_instructions.txt
        :start-line: 218
        :end-line: 219

Please continue with the installation instructions for EFA and PyTorch Neuron by following the corresponding AL2023 setup guide below (please skip the "Launch the Instance" and "Install Drivers and Tools" sections). 


.. card:: Pytorch Neuron (``torch-neuronx``) Setup on Amazon Linux 2023
            :link: setup-torch-neuronx-al2023
            :link-type: ref
            :class-body: sphinx-design-class-title-small

.. card:: Pytorch Neuron (``torch-neuron``) Setup on Amazon Linux 2023
            :link: setup-torch-neuron-al2023
            :link-type: ref
            :class-body: sphinx-design-class-title-small