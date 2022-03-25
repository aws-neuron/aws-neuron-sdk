.. _neuron-install-guide:
.. _non-dlami-setup:

Setup Guide
===========

.. _pytorch-setup:

PyTorch Setup 
-------------

* :ref:`Fresh install <install-neuron-pytorch>`
* :ref:`Update to latest release <update-neuron-pytorch>`
* :ref:`Install previous releases <install-prev-neuron-pytorch>`

.. _tensorflow-setup:

TensorFlow Setup 
----------------

* :ref:`Fresh install <install-neuron-tensorflow>`
* :ref:`Update to latest release <update-neuron-tensorflow>`
* :ref:`Install previous releases <install-prev-neuron-tensorflow>`

.. _mxnet-setup:

Apache MxNet Setup 
------------------

* :ref:`Fresh install <install-neuron-mxnet>`
* :ref:`Update to latest release <update-neuron-mxnet>`
* :ref:`Install previous releases <install-prev-neuron-mxnet>`


Troubleshooting
---------------

* :ref:`neuron-setup-troubleshooting`

.. _dlami-section:

Additional Setup Resources
---------------------------

Common Developer Flows
^^^^^^^^^^^^^^^^^^^^^^^

:ref:`neuron-devflows` secion describe the common flows to develop with Neuron.

More information about DLAMI and Inferentia be found also at the `DLAMI Documentation <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia.html>`_ .


Deep Learning AMI (DLAMI)
^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron packages are installed within Conda environments in  `Ubuntu 18.04 DLAMI <https://docs.aws.amazon.com/dlami/latest/devguide/ubuntu18-04.html>`_ and `Amazon Linux 2 DLAMI <https://docs.aws.amazon.com/dlami/latest/devguide/al2.html>`_ . For DLAMI release notes see:

* `AWS Deep Learning AMI (Amazon Linux 2) <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-amazon-linux-2/>`_
* `AWS Deep Learning AMI (Ubuntu 18.04) <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-ubuntu-18-04/>`_
* `AWS Deep Learning Base AMI (Amazon Linux 2) <https://aws.amazon.com/releasenotes/aws-deep-learning-base-ami-amazon-linux-2/>`_
* `AWS Deep Learning Base AMI (Ubuntu 18.04) <https://aws.amazon.com/releasenotes/aws-deep-learning-base-ami-ubuntu-18-04/>`_


Deep Learning (DL) Containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For containerized applications, it is recommended to use the neuron-rtd
container, more details at :ref:`neuron-containers`.

Jupyter notebook setup
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   /neuron-intro/setup-jupyter-notebook-steps-troubleshooting
   /neuron-intro/running-jupyter-notebook-as-script

Launching Inf1 Instance from AWS CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   /neuron-intro/install-templates/launch-inf1-dlami-aws-cli.rst

Tensorboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   /neuron-intro/install-tensorboard



