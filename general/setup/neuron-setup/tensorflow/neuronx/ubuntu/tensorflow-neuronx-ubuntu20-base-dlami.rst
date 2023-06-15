.. _setup-tensorflow-neuronx-u20-base-dlami:

.. card:: Select a Different Framework or Platform for Setup
    :link: setup-guide-index
    :link-type: ref
    :class-body: sphinx-design-class-title-small


TensorFlow Neuron ("tensorflow-neuronx") Setup on Ubuntu 20 with DLAMI Base
===========================================================================


.. contents:: Table of contents
	:local:
	:depth: 2


Get Started with Latest Release of TensorFlow Neuron (``tensorflow-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section provide links that will assist you to quickly start with a fresh installation of :ref:`tensorflow-neuronx-main`.



.. dropdown::  Launch the Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    * Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to launch an instance. When choosing the instance type at the EC2 console. please make sure to select the correct instance type.
    * To get more information about instance sizes and pricing see: `Trn1 web page <https://aws.amazon.com/ec2/instance-types/trn1/>`_, `Inf2 web page <https://aws.amazon.com/ec2/instance-types/inf2/>`_
    * Check for the latest version of the `DLAMI Base AMI <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-base-neuron-ubuntu-20-04/>`_ and copy the AMI name that starts with "Deep Learning Base Neuron AMI (Ubuntu 20.04) <latest_date>" from "AMI Name:" section
    * Search for the copied AMI name in the AMI Search , you should see a matching AMI with the AMI name in Community AMIs. Select the AMI and use it to launch the instance.
    * When launching a Trn1, please adjust your primary EBS volume size to a minimum of 512GB.
    * After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_ to connect to the instance 

.. dropdown::  Install Drivers and Tools
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. include :: /src/helperscripts/installationScripts/python_instructions.txt
        :start-line: 5
        :end-line: 6


.. include:: /general/quick-start/tab-inference-tensorflow-neuronx-u20.txt

.. include:: /frameworks/tensorflow/tensorflow-neuronx/setup/tensorflow-update-u20.rst

.. include:: /frameworks/tensorflow/tensorflow-neuronx/setup/tensorflow-install-prev-u20.rst
