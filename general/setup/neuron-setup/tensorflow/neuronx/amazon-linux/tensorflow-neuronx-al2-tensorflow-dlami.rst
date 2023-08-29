.. _setup-tensorflow-neuronx-al2-dlami-tensorflow:

.. card:: Select a Different Framework or Platform for Setup
    :link: setup-guide-index
    :link-type: ref
    :class-body: sphinx-design-class-title-small


Tensorflow Neuron ("tensorflow-neuronx") Setup on Amazon Linux 2 with DLAMI Tensorflow
=======================================================================================


.. contents:: Table of contents
	:local:
	:depth: 2


Get Started with Latest Release of Neuron Tensorflow (``tensorflow-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section provide links that will assist you to quickly start with a fresh installation of :ref:`tensorflow-neuronx-main`.


.. dropdown::  Launch the Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    * Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to launch an instance. When choosing the instance type at the EC2 console. please make sure to select the correct instance type.
    * To get more information about instances sizes and pricing see: `Trn1 web page <https://aws.amazon.com/ec2/instance-types/trn1/>`_, `Inf2 web page <https://aws.amazon.com/ec2/instance-types/inf2/>`_
    * Check for the latest version of the `Deep Learning AMI Neuron TensorFlow 2.10 <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-neuron-tensorflow-2-10-amazon-linux-2/>`_ and copy the AMI name that starts with "Deep Learning AMI Neuron TensorFlow 2.10 (Amazon Linux 2) <latest_date>" from "AMI Name:" section
    * Search for the copied AMI name in the AMI Search , you should see an exact matching AMI with the AMI name in Community AMIs. Select the AMI and use it to launch the instance.
    * When launching a Trn1, please adjust your primary EBS volume size to a minimum of 512GB.
    * After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_ to connect to the instance 


.. dropdown::  Update Neuron Drivers
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small

    .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=driver_runtime_tools --framework=pytorch --framework-version=1.13.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1


.. dropdown::  Get Started With Tensorflow DLAMI
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small

    .. include:: /src/helperscripts/installationScripts/python_instructions.txt
            :start-line: 92
            :end-line: 93

.. card:: Visit TensorFlow Neuron(``tensorflow-neuronx``) for Inference section
    :link: inference-tensorflow-neuronx
    :link-type: ref
    :class-body: sphinx-design-class-title-small

.. card:: Visit TensorFlow Neuron section for more
        :class-body: sphinx-design-class-body-small
        :link: tensorflow-neuron-main
        :link-type: ref

.. include:: /frameworks/tensorflow/tensorflow-neuronx/setup/tensorflow-update-al2-dlami.rst

.. include:: /frameworks/tensorflow/tensorflow-neuronx/setup/tensorflow-install-prev-al2.rst