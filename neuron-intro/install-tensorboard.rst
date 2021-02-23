.. _install-tensotboard:

Install Neuron Tensorboard
==========================

To install Tensorboard, first enable ML framework Conda environment of your choice, by running one of the following:

* Enable PyTorch-Neuron Conda enviroment:

 .. include:: /neuron-intro/install-templates/dlami-enable-neuron-pytorch.rst

* Enable TensorFlow-Neuron Conda enviroment:

  .. include:: /neuron-intro/install-templates/dlami-enable-neuron-tensorflow.rst

* Enable MXNet-Neuron Conda enviroment:

  .. include:: /neuron-intro/install-templates/dlami-enable-neuron-mxnet.rst

Then run the following:

.. code:: bash

   pip install tensorboard-neuron

-  Installing ``tensorflow-neuron`` will automatically install
   ``tensorboard-neuron`` as a dependency
-  To verify ``tensorboard-neuron`` is installed correctly, run
   ``tensorboard_neuron -h | grep run_neuron_profile``. If nothing is
   shown, please retry installation with the ``--force-reinstall``
   option.


