.. _neuron-install-guide:

Setup Guide
===========

.. contents:: Table of Contents
   :local:
   :depth: 2



Recommended Developer Flows
---------------------------

:ref:`neuron-devflows` secion describe the recommended flows to develop with Neuron.

More information about Deep Learning AMI (DLAMI) and other flows can be found in this Setup Guide.


.. _dlami-section:

Deep Learning AMI (DLAMI)
-------------------------

Neuron packages are installed within Conda environments in `AWS Deep Learning AMI (DLAMI) with Conda <https://docs.aws.amazon.com/dlami/latest/devguide/conda.html>`_, and `DLAMI <https://docs.aws.amazon.com/dlami/index.html>`_ is the recommended AMI to use with Neuron SDK. 


For more information about Neuron and DLAMI:

.. toctree::
   :maxdepth: 1

   /neuron-intro/dlami/dlami-neuron-matrix.rst
   /neuron-intro/dlami/dlami-neuron-conda-pip.rst
   /release-notes/dlami-release-notes


More information about DLAMI and Inferentia be found also at the `DLAMI Documentation <https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-inferentia.html>`_ .


DL Containers
-------------

For containerized applications, it is recommended to use the neuron-rtd
container, more details at :ref:`neuron-containers`.

Inferentia support for `AWS DL
Containers <https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-ec2.html>`__
is coming soon.


Jupyter notebook
----------------

.. toctree::
   :maxdepth: 1

   /neuron-intro/setup-jupyter-notebook-steps-troubleshooting
   /neuron-intro/running-jupyter-notebook-as-script

Launching Inf1 Instance from AWS CLI
------------------------------------

.. toctree::
   :maxdepth: 1

   /neuron-intro/install-templates/launch-inf1-dlami-aws-cli.rst

Tensorboard
-----------

.. toctree::
   :maxdepth: 1

   /neuron-intro/install-tensorboard

.. _non-dlami-setup:

non-DLAMI Setup
---------------

In addition to DLAMI Neuron SDK can be installed on Ubuntu or Amazon Linux using 
standard package managers (apt, yum, pip, and conda) to install and keep 
updates current


.. toctree::
   :maxdepth: 1

   install-pytorch
   install-tensorflow
   install-mxnet
