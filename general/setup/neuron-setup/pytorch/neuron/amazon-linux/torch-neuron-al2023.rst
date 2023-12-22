.. _setup-torch-neuron-al2023:

.. card:: Select a Different Framework or Platform for Setup
    :link: setup-guide-index
    :link-type: ref
    :class-body: sphinx-design-class-title-small


PyTorch Neuron ("torch-neuron") Setup on Amazon Linux 2023
===========================================================


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
    * Select Amazon Linux 2023 AMI
    * After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_ to connect to the instance  

.. dropdown::  Install Drivers and Tools
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --framework=pytorch --framework-version=1.13.1 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami --category=driver_runtime_tools


.. include:: /general/quick-start/tab-inference-torch-neuron-al2023.txt