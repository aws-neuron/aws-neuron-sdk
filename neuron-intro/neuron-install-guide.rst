.. _neuron-install-guide:

Setup Guide
===========

.. _non-dlami-setup:

PyTorch Setup 
--------------

* :ref:`Fresh install <install-neuron-pytorch>`
* :ref:`Update to latest release <update-neuron-pytorch>`
* :ref:`Install previous releases <install-prev-neuron-pytorch>`

TensorFlow Setup 
--------------

* :ref:`Fresh install <install-neuron-tensorflow>`
* :ref:`Update to latest release <update-neuron-tensorflow>`
* :ref:`Install previous releases <install-prev-neuron-tensorflow>`


Apache MxNet Setup 
--------------

* :ref:`Fresh install <install-neuron-mxnet>`
* :ref:`Update to latest release <update-neuron-mxnet>`
* :ref:`Install previous releases <install-prev-neuron-mxnet>`


.. _dlami-section:

Additional Setup Resources
---------------------------

Common Developer Flows
^^^^^^^^^^^^^^^^^^^^^^^

:ref:`neuron-devflows` secion describe the common flows to develop with Neuron.

More information about DLAMI and Inferentia be found also at the `DLAMI Documentation <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia.html>`_ .


Deep Learning AMI (DLAMI)
^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron packages are installed within Conda environments in `AWS Deep Learning AMI (DLAMI) with Conda <https://docs.aws.amazon.com/dlami/latest/devguide/conda.html>`_, and `DLAMI <https://docs.aws.amazon.com/dlami/index.html>`_ is the recommended AMI to use with Neuron SDK. 


For more information about Neuron and DLAMI:

.. toctree::
   :maxdepth: 1

   /neuron-intro/dlami/dlami-neuron-matrix.rst
   /neuron-intro/dlami/dlami-neuron-conda-pip.rst
   /release-notes/dlami-release-notes



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



