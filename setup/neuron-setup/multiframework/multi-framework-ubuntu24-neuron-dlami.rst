.. _setup-ubuntu24-multi-framework-dlami:

.. card:: Select a Different Framework or Platform for Setup
    :link: setup-guide-index
    :link-type: ref
    :class-body: sphinx-design-class-title-small


Get Started with Neuron on Ubuntu 24 with Neuron Multi-Framework DLAMI
======================================================================

You can quickly get started on Ubuntu 24 using the Neuron Deep Learning AMI (DLAMI). Then, start using one of the multiple frameworks or libraries that Neuron SDK supports by
activating the corresponding virtual environment. Each virtual environment comes pre-installed with Neuron libraries needed for you to get started. The Neuron DLAMI supports all Neuron instances (Inf2/Trn1/Trn1n/Trn2)
and is updated with each Neuron SDK release. To start using the latest version of the Neuron DLAMI, use the following steps:

Step 1:  Launch the instance using Neuron DLAMI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you open the `EC2 Console <https://console.aws.amazon.com/ec2>`_, select your desired AWS region and choose "Launch Instance". Under AMI selection select the "Quick Start"
and "Ubuntu", choose the "Deep Learning AMI Neuron (Ubuntu 24.04)"(see screenshot below). Once you have selected the AMI, select the desired Neuron Instance(Inf2/Trn1/Trn1n/Trn2) , 
configure disk size and other criteria, launch the instance

.. image:: /images/neuron-multi-framework-dlami-U24-quick-start.png
    :scale: 20%
    :align: center


.. note::
  If you are looking to use the Neuron DLAMI in your cloud automation flows , Neuron also supports :ref:`SSM parameters <ssm-parameter-neuron-dlami>` to easily retrieve the latest DLAMI id.



Step 2: Activate the desired virtual environment 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  

You can activate one of the virtual environments depending on the library or framework you are interested in:

1. Get the desired virtual environment name for the framework/library by referring to :ref:`the Neuron DLAMI overview <neuron-dlami-multifw-venvs>`.
2. Activate the virtual environment by using:

  ::

    source /opt/<name_of_virtual_environment>/bin/activate


After you have activated the desired virtual environment , you can try out one of the tutorials listed in the corresponding framework or library training and inference section.
















