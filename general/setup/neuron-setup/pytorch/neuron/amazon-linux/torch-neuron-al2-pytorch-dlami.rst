.. _setup-torch-neuron-al2-pytorch-dlami:

.. card:: Select a Different Framework or Platform for Setup
    :link: setup-guide-index
    :link-type: ref
    :class-body: sphinx-design-class-title-small


PyTorch Neuron ("torch-neuron") Setup on Amazon Linux 2 with Pytorch DLAMI
=========================================================================

.. note::
   As of 2.20.0, Neuron Runtime no longer supports AL2. Upgrade to AL2023 following the :ref:`AL2 Migration guide <eos-al2>`

.. contents:: Table of contents
	:local:
	:depth: 2

.. include:: /general/setup/install-templates/al2-python.rst

Get Started with Latest Release of PyTorch Neuron (``torch-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section provide links that will assist you to quickly start with a fresh installation of :ref:`setup-torch-neuron`.


.. dropdown::  Launch the Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    * Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to launch an instance. When choosing the instance type at the EC2 console. please make sure to select the correct instance type.
    * To get more information about instances sizes and pricing see: `Inf1 web page <https://aws.amazon.com/ec2/instance-types/inf1/>`_
    * Check for the latest version of the `DLAMI Neuron Pytorch 1.13 AMI <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-neuron-pytorch-1-13-amazon-linux-2/>`_ and copy the AMI name that starts with "Deep Learning AMI Neuron PyTorch 1.13 (Amazon Linux 2) <latest_date>" from "AMI Name:" section
    * Search for the copied AMI name in the AMI Search , you should see an exact matching AMI with the AMI name in Community AMIs. Select the AMI and use it to launch the instance.
    * After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_ to connect to the instance 

.. dropdown::  Update Neuron Drivers
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small

    .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=driver_runtime_tools --framework=pytorch --framework-version=1.13.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1

.. dropdown::  Get Started With Pytorch DLAMI
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small

    .. include:: /src/helperscripts/installationScripts/python_instructions.txt
            :start-line: 98
            :end-line: 99

.. card:: Visit PyTorch Neuron(``torch-neuron``) for Inference section
    :link: inference-torch-neuron
    :link-type: ref
    :class-body: sphinx-design-class-title-small

.. card:: Visit PyTorch Neuron section for more
    :class-body: sphinx-design-class-body-small
    :link: neuron-pytorch
    :link-type: ref

.. include:: /frameworks/torch/torch-neuron/setup/pytorch-update-al2-dlami.rst

.. include:: /frameworks/torch/torch-neuron/setup/pytorch-install-prev-al2.rst