.. _software-maintenance:

Software Maintenance and Deprecation
====================================

.. contents::
	:local:
	:depth: 1



.. _eol-ncgs-env_2:

MM/DD/YYY - Announcing end of support for ``NEURONCORE_GROUP_SIZES`` starting with Neuron 1.20.0 release
--------------------------------------------------------------------------------------------------------

MM/DD/YYY - Starting with Neuron SDK 1.20.0, ``NEURONCORE_GROUP_SIZES`` environment variable will no longer be supported. Setting 
``NEURONCORE_GROUP_SIZES`` environment variable will no longer affect applications behavior.
Current customers using ``NEURONCORE_GROUP_SIZES`` environment variable are advised to use ``NEURON_RT_VISIBLE_CORES`` environment variable  or ``NEURON_RT_NUM_CORES`` environment variable instead.

See :ref:`eol-ncg`, :ref:`nrt-configuration` and :ref:`neuron-migrating-apps-neuron-to-libnrt` for more information.


.. _announce-eol-pt-1-5:

MM/DD/YYY - End of support for TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4
-------------------------------------------------------------------------------

MM/DD/YYYY - Starting with *Neuron 1.18.0* release, *TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4* will no longer be supported, and  
no further releases of *TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4* will be issued.  Current users of those versions are advised to migrate to 
*TensorFlow Neuron version 2.5 or above*.


.. _announce-eol-pt-1-5:

MM/DD/YYY - Announcing end of support for PyTorch Neuron version 1.5 starting with Neuron 1.19.0 release
--------------------------------------------------------------------------------------------------------

MM/DD/YYYY - Starting with *Neuron 1.19.0* release, *PyTorch Neuron version 1.5* will no longer be supported. Last release of *PyTorch Neuron version 1.5* will be issued
as part of *Neuron 1.18.0* release. Current users of those versions are advised to migrate to *PyTorch Neuron version 1.10 or above*.


.. _maintenance_tf21_tf24:

02/17/2022 - TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4 enter maintenance mode
------------------------------------------------------------------------------------

02/17/2022 - Starting with *Neuron 1.17.2* release, *TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4* are entering maintenance mode.  Future releases of 
*TensorFlow Neuron versions 2.1, 2.2, 2.3 and 2.4* will address critical security issues only. Current users of those versions are advised to migrate to 
*TensorFlow Neuron version 2.5 or above*.


.. _maintenance_rtd:

10/27/2021 - Neuron Runtime 1.x (``neuron-rtd``) enters maintenance mode
------------------------------------------------------------------------

10/27/2021 - Starting with *Neuron 1.16.0* release, *Neuron Runtime 1.x* (``neuron-rtd``) is entering maintenance mode and replaced 
with *Neuron Runtime 2.x*, a shared library named ``libnrt.so``. 
Future releases of *Neuron Runtime 1.x* (``neuron-rtd``) will address critical bug fixes and security issues only. Previous releases of 
*Neuron Runtime 1.x* (``neuron-rtd``) will continue to be available via ``rpm`` and ``deb`` packages.

For more information please see:

	* :ref:`introduce-libnrt`
	* :ref:`neuron-install-guide`
	* :ref:`neuron-maintenance-policy`


.. _maintenance_mxnet_1_5:

10/27/2021 - Neuron support for *Apache MXNet 1.5* enters maintenance mode
--------------------------------------------------------------------------

10/27/2021 - Starting *Neuron release 1.16.0*,  Neuron support for *MXNet 1.5* is entering maintenance mode.
Future releases of Neuron supporting *MXNet 1.5*  will address critical bug fixes and security issues only.
Previous releases of *Apache MXNet 1.5* will continue to be available via ``pip`` packages.

Current users of *Neuron MXNet 1.5* can migrate their applications to *Neuron MXNet 1.8*, for more information 
about Neuron MXNet support and how to upgrade to latest *Neuron MXNet 1.8*, please see visit :ref:`neuron-mxnet`.


.. _maintenance_neuron-cli:

10/27/2021 - ``neuron-cli`` enters maintenance mode
---------------------------------------------------

10/27/2021 - Starting *Neuron release 1.16.0*, with the introduction of *Neuron Runtime 2.x*, ``neuron-cli`` is entering maintenance mode. ``neuron-cli`` 
functionality will be available only if *Neuron Runtime 1.x* (``neuron-rtd``) is being used by the application. If the application is using 
*Neuron Runtime 2.x* shared library(``libnrt.so``), ``neuron-cli`` functionality will not be available.


If you have used ``neuron-cli`` in previous releases, and you are migrating to
newer Neuron releases where applications require *Neuron Runtime 2.x* shared library, please see the below :ref:`neuron-cli-mntnce-faq`.
Future releases of ``neuron-cli`` will address 
critical bug fixes and security issues only. Previous releases of ``neuron-cli`` will continue to be available via ``rpm`` and ``deb`` packages.


.. _eol-ncg:

10/27/2021 - End of support for NeuronCore Groups (NCG)
-------------------------------------------------------

10/27/2021 -  Before the introduction of *Neuron Runtime 2.x*, :ref:`NeuronCore Group (NCG) <neuron-core-group>` has been used by Neuron Runtime 1.x 
to define an execution group of one or more NeuronCores where models can be loaded and executed. It also provided separation between processes.
   
With the introduction of *Neuron Runtime 2.x*, the strict separation of NeuronCores into groups is no longer needed and NeuronCore Groups (NCG) is 
deprecated.  *Neuron Runtime 2.x* enables each process to own a set of NeuronCores, and within each process, Neuron Runtime 2.x supports loading and 
executing multiple models on separate , different or overlapping sets of NeuronCores.

Please note that ``NEURONCORE_GROUP_SIZES`` environment variable is in the process of being :ref:`deprecated <eol-ncgs-env>`, and for a transition period 
``NEURONCORE_GROUP_SIZES`` can be used to preserve the old NeuronCore Group behavior. The frameworks internally would convert ``NEURONCORE_GROUP_SIZES`` to 
use runtime's new mode of mapping models to NeuronCores.

For more information see details about ``NEURON_RT_VISIBLE_CORES`` at :ref:`nrt-configuration` and  and :ref:`neuron-migrating-apps-neuron-to-libnrt`.


.. _eol-ncgs-env:

10/27/2021 - Announcing end of support for ``NEURONCORE_GROUP_SIZES``
---------------------------------------------------------------------

10/27/2021 - ``NEURONCORE_GROUP_SIZES`` environment variable is in the process of being deprecated, future Neuron releases may no longer support
the ``NEURONCORE_GROUP_SIZES`` environment variable. Please start
using ``NEURON_RT_VISIBLE_CORES`` instead.

See :ref:`eol-ncg`, :ref:`nrt-configuration` and :ref:`neuron-migrating-apps-neuron-to-libnrt` for more information.




.. _neuron-cli-mntnce-faq:

Frequently Asked questions (FAQ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Is there another tool that provide the same functionality as ``neuron-cli list-model``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, please see :ref:`neuron-ls-ug` or :ref:`neuron-monitor-ug`.

Is there another tool that provide the same functionality as ``neuron-cli create-ncg``, ``neuron-cli destroy-ncg``, and ``neuron-cli list-ncg``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, these functionalities are no longer needed with *Neuron Runtime 2.x*,NeuronCore Groups (NCG) :ref:`is deprecated <eol-ncg>` and ``NEURONCORE_GROUP_SIZES`` environment variable :ref:`is in the process of being deprecated <eol-ncgs-env>`, Please start using ``NEURON_RT_VISIBLE_CORES`` instead. See :ref:`nrt-configuration` and :ref:`neuron-migrating-apps-neuron-to-libnrt` 

for more information.

Is there another tool that provide the same functionality as ``neuron-cli reset``?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, this functionality is no longer needed with *Neuron Runtime 2.x*. Before introducing ``libnrt.so``, in certain cases after an application 
crashed  models had to be unloaded manually by calling neuron-cli reset.

With ``libnrt.so``, applications runs in the context of the ``libnrt.so`` shared library and when an application exits the Neuron driver will free all resources associated with the application.


For more information please see:

	* :ref:`introduce-libnrt`
	* :ref:`neuron-tools`
	* :ref:`neuron-install-guide`
	* :ref:`neuron-maintenance-policy`


.. _eol-conda-packages:

05/28/2021 - End of support for Neuron Conda packages in Deep Learning AMI starting Neuron 1.14.0
-------------------------------------------------------------------------------------------------

05/28/2021 - Starting with Neuron SDK 1.14.0, we will no longer support conda packages to install Neuron SDK framework in DLAMI and we will no longer update conda packages used to install Neuron SDK framework (Neuron conda packages) with new versions.

Starting with Neuron SDK 1.14.0, pip packages (Neuron pip packages) will be used to install Neuron SDK framework in DLAMI conda environment. To upgrade Neuron SDK framework DLAMI users should use pip upgrade commands instead of conda update commands. Instructions are available in this blog and in Neuron SDK documentation (https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-intro/neuron-install-guide.html#deep-learning-ami-dlami).


Starting with Neuron SDK 1.14.0, run one of the following commands to upgrade to latest Neuron framework of your choice:

* To upgrade Neuron PyTorch:

.. code-block::

    source activate aws_neuron_pytorch_p36
    pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    pip install --upgrade torch-neuron neuron-cc[tensorflow] torchvision

* To upgrade Neuron TensorFlow:

.. code-block::

   source activate aws_neuron_tensorflow_p36
   pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
   pip install --upgrade tensorflow-neuron tensorboard-neuron neuron-cc

* To upgrade Neuron MXNet:

.. code-block::

   source activate aws_neuron_mxnet_p36
   pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
   pip install --upgrade mxnet-neuron neuron-cc

For more information please check the `blog <https://aws.amazon.com/blogs/developer/neuron-conda-packages-eol/>`__.



.. _eol-ubuntu16:

05/01/2021 - End of support for Ubuntu 16 starting Neuron 1.14.0
----------------------------------------------------------------

05/01/2021 - Ubuntu 16.04 entered end of life phase officially in April 2021 (see https://ubuntu.com/about/release-cycle) and will not receive any public software or security updates. Starting with Neuron SDK 1.14.0, Ubuntu 16 is no longer supported for Neuron, users who are using Ubuntu 16 are requested to migrate to Ubuntu18 or Amazon Linux 2.

Customers who choose to upgrade libc on Ubuntu 16 to work with Neuron v1.13.0 (or higher versions) are highly discouraged from doing that since Ubuntu 16 will no longer receive public security updates.

.. _eol-classic-tensorboard:

05/01/2021 - End of support for classic TensorBoard-Neuron starting Neuron 1.13.0 and introducing Neuron Plugin for TensorBoard 
-------------------------------------------------------------------------------------------------------------------------------

05/01/2021 - Starting with Neuron SDK 1.13.0, we are introducing :ref:`Neuron Plugin for TensorBoard <neuron-plugin-tensorboard>` and we will no longer support classic TensorBoard-Neuron. Users are required to migrate to Neuron Plugin for TensorBoard.

Starting with Neuron SDK 1.13.0, if you are using TensorFlow-Neuron within DLAMI Conda environment, attempting to run ``tensorboard`` with the existing version of TensorBoard will fail.  Please update the TensorBoard version before installing the Neuron plugin by running ``pip install TensorBoard --force-reinstall``, for installation instructions see :ref:`neuron-plugin-tensorboard`.

Users who are using Neuron SDK releases before 1.13.0,  can find classic TensorBoard-Neuron documentation at `Neuron 1.12.2 documentation <https://awsdocs-neuron.readthedocs-hosted.com/en/1.12.2/neuron-guide/neuron-tools/getting-started-tensorboard-neuron.html>`__.


For more information see see :ref:`neuron-tensorboard-rn` and :ref:`neuron-plugin-tensorboard`.

.. _eol_python_3_5:

02/24/2021 - End of support for Python 3.5 
-----------------------------------------

02/24/2021 - As Python 3.5 reached end-of-life in October 2020, and many packages including TorchVision and Transformers have
stopped support for Python 3.5, we will begin to stop supporting Python 3.5 for frameworks, starting with
PyTorch-Neuron version :ref:`neuron-torch-11170` in this release. You can continue to use older versions with Python 3.5.


11/17/2020 - End of support for ONNX 
------------------------------------

11/17/2020 - ONNX support is limited and from this version onwards we are not
planning to add any additional capabilities to ONNX. We recommend
running models in TensorFlow, PyTorch or MXNet for best performance and
support.


07/16/2020 - End of support for PyTorch 1.3 
------------------------------------------

07/16/2020 - Starting this release we are ending the support of PyTorch 1.3 and migrating to PyTorch 1.5.1, customers are advised to migrate to PyTorch 1.5.1.



