.. _neuron-conda-packages:

Deep Learning AMI (DLAMI) Neuron Conda Packages FAQ
===================================================

.. contents:: Table of Contents
   :local:
   :depth: 2


.. _how-to-update-to-latest-Neuron-Conda:

How to update to latest Neuron Conda packages?
-----------------------------------------------

.. _launch-new-instance-with-dlami:

If Launching a new instance with DLAMI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the DLAMI doesn't include the latest Neuron Conda packages, update the Conda packages as follows:


* PyTorch

 .. code::

    conda update numpy -y
    conda update torch-neuron

* TensorFlow

 .. code::

    conda update numpy -y
    conda update tensorflow-neuron


* MXNet

 .. code::

    conda update numpy -y
    conda update mxnet-neuron


.. note::

   To avoid breaking an existing DLAMI enviroment, backup your DLAMI enviroment by creating an AMI from the existing DLAMI environment. Follow instructions at `Create an AMI from an Amazon EC2 Instance <https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/tkv-create-ami-from-instance.html>`_  to save the DLAMI before updating the Neuron Conda packages or upgrading to the latest DLAMI.


If running old DLAMI version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To update to latest Neuron version it is recommended to upgrade to latest DLAMI as described at `Upgrading to a New DLAMI Version <https://docs.aws.amazon.com/dlami/latest/devguide/upgrading-dlami.html>`_ and then follow instructions at :ref:`launch-new-instance-with-dlami` .


If you choose not to upgrade to latest DLAMI, follow the instructions described in :ref:`non-dlami-setup` to update to latest Neuron packages.

.. note::

   To avoid breaking an existing DLAMI enviroment, backup your DLAMI enviroment by creating an AMI from the existing DLAMI environment. Follow instructions at `Create an AMI from an Amazon EC2 Instance <https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/tkv-create-ami-from-instance.html>`_  to save the DLAMI before updating the Neuron Conda packages or upgrading to the latest DLAMI.


What DLAMI versions support Neuron Conda Packages?
--------------------------------------------------

Starting with the DLAMI v26.0, the `Deep Learning AMI with Conda Options <https://docs.aws.amazon.com/dlami/latest/devguide/conda.html>`_ include Neuron Conda packages.

.. note::

   Only Ubuntu 16,18 and Amazon Linux2 DLAMI are supported (Amazon Linux is not supported)   


What version of Neuron Conda packages are included in latest DLAMI version? 
----------------------------------------------------------------------------

Both the DLAMI and Neuron have a monthly release cadence. When there is a new DLAMI release, it will include the latest Neuron Conda packages at the release time. This means that the latest DLAMI version include either the latest Neuron Conda packages or the previous. See :ref:`dlami-neuron-matrix` for latest DLAMI information.


Should I update to latest Neuron Conda packages?
-------------------------------------------------

Update to the latest Neuron Conda packages if the tutorial or the machine learning application you intend to run require a feature or bug fix from the latest Neuron version. See :ref:`neuron-whatsnew` for information on the latest Neuron version.


.. _dlami-version-howto:

How to know which DLAMI version I am running?
----------------------------------------------

You see the version of the running DLAMI by inspecting the README file on the user's home folder, or at the start of a new terminal session. In the example below the DLAMI version is 35.0

 .. code::

    source activate aws_neuron_mxnet_p36
    cat ~/README
    
    (aws_neuron_mxnet_p36) ubuntu@ip-172-31-88-188:~/aws-neuron-sdk/src/examples$ cat ~/README
    =============================================================================
           __|  __|_  )
           _|  (     /   Deep Learning AMI (Ubuntu 18.04) Version 35.0
          ___|\___|___|
    =============================================================================

.. _neuron-conda-version-howto:

How to know which packages are available in the DLAMI Conda environment I am running?
---------------------------------------------------------------------------------------

 .. code::

    conda list | grep neuron

.. _latest-neuron-conda-version-howto:

How to find available Conda packages to install (latest Neuron Conda packages)?
--------------------------------------------------------------------------------

* PyTorch

 .. code::

    conda search torch-neuron


* TensorFlow

 .. code::

    conda search tensorflow-neuron


* MXNet

 .. code::

    conda search mxnet-neuron



