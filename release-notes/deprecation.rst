.. _software-deprecation:

Software deprecation
====================

.. contents::
	:local:
	:depth: 1
	

.. _eol-conda-packages:

End of support for Neuron Conda packages in Deep Learning AMI starting Neuron 1.14.0
------------------------------------------------------------------------------------

04/30/2021 - Starting with Neuron SDK 1.14.0, we will no longer support conda packages to install Neuron SDK framework in DLAMI and we will no longer update conda packages used to install Neuron SDK framework (Neuron conda packages) with new versions.

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

End of support for Ubuntu 16 starting Neuron 1.14.0
---------------------------------------------------

04/30/2021 - Ubuntu 16.04 entered end of life phase officially in April 2021 (see https://ubuntu.com/about/release-cycle) and will not receive any public software or security updates. Starting with Neuron SDK 1.14.0, Ubuntu 16 is no longer supported for Neuron, users who are using Ubuntu 16 are requested to migrate to Ubuntu18 or Amazon Linux 2.

Customers who choose to upgrade libc on Ubuntu 16 to work with Neuron v1.13.0 (or higher versions) are highly discouraged from doing that since Ubuntu 16 will no longer receive public security updates.

.. _eol-classic-tensorboard:

End of support for classic TensorBoard-Neuron starting Neuron 1.13.0 and introducing Neuron Plugin for TensorBoard 
-------------------------------------------------------------------------------------------------------------------

04/30/2021 - Starting with Neuron SDK 1.13.0, we are introducing :ref:`Neuron Plugin for TensorBoard <neuron-plugin-tensorboard>` and we will no longer support classic TensorBoard-Neuron. Users are required to migrate to Neuron Plugin for TensorBoard.

Starting with Neuron SDK 1.13.0, if you are using TensorFlow-Neuron within DLAMI Conda environment, attempting to run ``tensorboard`` with the existing version of TensorBoard will fail.  Please update the TensorBoard version before installing the Neuron plugin by running ``pip install TensorBoard --force-reinstall``, for installation instructions see :ref:`neuron-plugin-tensorboard`.

Users who are using Neuron SDK releases before 1.13.0,  can find classic TensorBoard-Neuron documentation at `Neuron 1.12.2 documentation <https://awsdocs-neuron.readthedocs-hosted.com/en/1.12.2/neuron-guide/neuron-tools/getting-started-tensorboard-neuron.html>`__.


For more information see see :ref:`neuron-tensorboard-rn` and :ref:`neuron-plugin-tensorboard`.


End of support for Python 3.5 
-----------------------------

2/24/2021 - As Python 3.5 reached end-of-life in October 2020, and many packages including TorchVision and Transformers have
stopped support for Python 3.5, we will begin to stop supporting Python 3.5 for frameworks, starting with
PyTorch-Neuron version :ref:`neuron-torch-11170` in this release. You can continue to use older versions with Python 3.5.


End of support for ONNX 
------------------------

11/17/2020 - ONNX support is limited and from this version onwards we are not
planning to add any additional capabilities to ONNX. We recommend
running models in TensorFlow, PyTorch or MXNet for best performance and
support.


End of support for PyTorch 1.3 
------------------------------

7/16/2020 - Starting this release we are ending the support of PyTorch 1.3 and migrating to PyTorch 1.5.1, customers are advised to migrate to PyTorch 1.5.1.

