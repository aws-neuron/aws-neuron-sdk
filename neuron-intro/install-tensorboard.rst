.. _install-tensorboard:

Install Neuron Plugin for TensorBoard
=====================================

The Neuron plugin for TensorBoard is available starting with Neuron v1.13.0.

.. _tensorboard-plugin-neuron-dlami-conda:

To install the Neuron plugin, first enable ML framework Conda environment of your choice, by running one of the following:

* Enable PyTorch-Neuron Conda enviroment:

 .. include:: /neuron-intro/install-templates/dlami-enable-neuron-pytorch.rst

* Enable TensorFlow-Neuron Conda enviroment:

  .. include:: /neuron-intro/install-templates/dlami-enable-neuron-tensorflow.rst

* Enable MXNet-Neuron Conda enviroment:

  .. include:: /neuron-intro/install-templates/dlami-enable-neuron-mxnet.rst

Then run the following:

.. include:: /neuron-intro/install-templates/tensorboard-plugin-neuron-pip-install.rst


Install Neuron TensorBoard (Deprecated)
=======================================

.. warning::

  TensorBoard-Neuron is deprecated and no longer compatible with Neuron tools version 1.5 and higher.
  Neuron tools version 1.5 is first introduced in Neuron v1.13.0 release.
  Please use the Neuron plugin for TensorBoard instead.

To install Tensorboard, first enable ML framework Conda environment of your choice, by running one of the following:

* Enable PyTorch-Neuron Conda enviroment:

 .. include:: /neuron-intro/install-templates/dlami-enable-neuron-pytorch.rst

* Enable TensorFlow-Neuron Conda enviroment:

  .. include:: /neuron-intro/install-templates/dlami-enable-neuron-tensorflow.rst

* Enable Neuron Conda enviroment for Neuron Apache MXNet (Incubating):

  .. include:: /neuron-intro/install-templates/dlami-enable-neuron-mxnet.rst

Then run the following:

.. code:: bash

   pip install tensorboard-neuron

-  Installing ``tensorflow-neuron<=1.15.5.1.2.9.0`` will automatically install
   ``tensorboard-neuron`` as a dependency.  The final version, ``1.15.5.1.2.9.0``, is
   part of Neuron v1.12.2 release.
-  To verify ``tensorboard-neuron`` is installed correctly, run
   ``tensorboard_neuron -h | grep run_neuron_profile``. If nothing is
   shown, please retry installation with the ``--force-reinstall``
   option.


