.. _setup-torch-neuron-al2:

.. card:: Select a Different Framework or Platform for Setup
    :link: setup-guide-index
    :link-type: ref
    :class-body: sphinx-design-class-title-small


PyTorch Neuron ("torch-neuron") Setup on Amazon Linux 2
=========================================================


.. contents:: Table of contents
	:local:
	:depth: 2


Get Started with Latest Release of PyTorch Neuron (``torch-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section provide links that will assist you to quickly start with a fresh installation of :ref:`setup-torch-neuron` for Inference.


.. dropdown::  Launch the Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    * Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to launch an instance. When choosing the instance type at the EC2 console, please make sure to select the correct instance type.
    * To get more information about instances sizes and pricing see: `Inf1 web page <https://aws.amazon.com/ec2/instance-types/inf1/>`_
    * Select Amazon Linux 2 AMI(HVM) - Kernel 5.10
    * After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_ to connect to the instance  

.. dropdown::  Install Drivers and Tools
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. include :: /src/helperscripts/installationScripts/python_instructions.txt
        :start-line: 2
        :end-line: 3


.. include:: /general/quick-start/tab-inference-torch-neuron-al2.txt

.. include :: /frameworks/torch/torch-neuron/setup/pytorch-update-al2.rst

.. include :: /frameworks/torch/torch-neuron/setup/pytorch-install-prev-al2.rst