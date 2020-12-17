.. _install-tensotboard:

Install Neuron Tensorboard
==========================

To install Tensorboard first install your neuron framework of choice:

- :ref:`install-neuron-pytorch`
- :ref:`install-neuron-tensorflow`
- :ref:`install-neuron-mxnet`

Then run the following:

.. code:: bash

   pip install tensorboard-neuron

-  Installing ``tensorflow-neuron`` will automatically install
   ``tensorboard-neuron`` as a dependency
-  To verify ``tensorboard-neuron`` is installed correctly, run
   ``tensorboard_neuron -h | grep run_neuron_profile``. If nothing is
   shown, please retry installation with the ``--force-reinstall``
   option.


