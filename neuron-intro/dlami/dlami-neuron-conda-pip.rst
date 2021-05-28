.. _dlami-neuron-conda-env-pip-packages:

Neuron Pip Packages within DLAMI Conda Environments FAQ
=======================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _how-to-update-to-latest-Neuron-Conda-Env:

How to update to latest Neuron packages in DLAMI Conda Environments?
--------------------------------------------------------------------

If the DLAMI Conda Environments donot include the latest Neuron packages, update the packages as follows:

* To upgrade Neuron PyTorch:

 .. code::

    source activate aws_neuron_pytorch_p36
    pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    pip install --upgrade torch-neuron neuron-cc[tensorflow] torchvision

* To upgrade Neuron TensorFlow:

 .. code::

    source activate aws_neuron_tensorflow_p36
    pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    pip install --upgrade tensorflow-neuron tensorboard-neuron neuron-cc

* To upgrade Neuron MXNet:

 .. code::

    source activate aws_neuron_mxnet_p36
    pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    pip install --upgrade mxnet-neuron neuron-cc

.. note::

   To avoid breaking an existing DLAMI enviroment, backup your DLAMI enviroment by creating an AMI from the existing DLAMI environment. Follow instructions at `Create an AMI from an Amazon EC2 Instance <https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/tkv-create-ami-from-instance.html>`_  to save the DLAMI before updating the Neuron Conda packages or upgrading to the latest DLAMI.

What DLAMI versions include Neuron Conda environments?
------------------------------------------------------

Starting with the DLAMI v26.0, the `Deep Learning AMI with Conda Options <https://docs.aws.amazon.com/dlami/latest/devguide/conda.html>`_ include Neuron Conda packages.

Starting with Neuron SDK 1.14.0, pip packages (Neuron pip packages) are used to install Neuron SDK framework in DLAMI conda environments. To upgrade Neuron SDK framework DLAMI users should use pip upgrade commands instead of conda update commands.
Instructions are in :ref:`how-to-update-to-latest-Neuron-Conda-Env`. For more information, see https://aws.amazon.com/blogs/developer/neuron-conda-packages-eol/.

.. note::

   Only Ubuntu 18 and Amazon Linux2 DLAMI are supported (Amazon Linux and Ubuntu 16 are not supported)


What version of Neuron packages are included in latest DLAMI version?
---------------------------------------------------------------------

Both the DLAMI and Neuron have a monthly release cadence. When there is a new DLAMI release, it will include the latest Neuron Conda packages at the release time. This means that the latest DLAMI version include either the latest Neuron packages or the previous. See :ref:`dlami-neuron-matrix` for latest DLAMI information.


Should I update to latest Neuron packages?
------------------------------------------

Update to the latest Neuron packages if the tutorial or the machine learning application you intend to run
require a feature or bug fix from the latest Neuron version. See :ref:`neuron-whatsnew` for information on the latest Neuron version.

.. _dlami-version-howto:

How to know which DLAMI version I am running?
----------------------------------------------

You see the version of the running DLAMI by inspecting the README file on the user's home folder, or at the start of a new terminal session. In the example below the DLAMI version is 35.0

 .. code::

    cat ~/README

    ubuntu@ip-172-31-88-188:~/$ cat ~/README
    =============================================================================
           __|  __|_  )
           _|  (     /   Deep Learning AMI (Ubuntu 18.04) Version 35.0
          ___|\___|___|
    =============================================================================