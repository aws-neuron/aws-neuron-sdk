.. _neuronperf_framework_notes:

==========================
NeuronPerf Framework Notes
==========================

PyTorch
=======

  * Requires: ``torch-neuron``
  	- Versions: 1.7.x, 1.8.x, 1.9.x, 1.10.x
  * Input to ``compile``: ``torch.nn.Module``
  * Model inputs: ``Any``.


TensorFlow 1.x
==============

  * Requires: ``tensorflow-neuron``
  	- Versions: All
  * Input to ``compile``: Path to uncompiled model dir from ``saved_model.simple_save``
  * Model inputs: Tensors must be provided as ``numpy.ndarray``

.. note::

	Although TensorFlow *tensors* must be ``ndarray``, this doesn't stop you from wrapping them inside of data structures that traverse process boundaries safely. For example, you can still pass an input ``dict`` like ``{'input_0': np.zeros((2, 1))}``.

TensorFlow 2.x
==============

  * Requires: ``tensorflow-neuron``
  	- Versions: All
  * Input to ``compile``: ``tf.keras.Model``
  * Model inputs: Tensors must be provided as ``numpy.ndarray``

.. note::

	Although TensorFlow *tensors* must be ``ndarray``, this doesn't stop you from wrapping them inside of data structures that traverse process boundaries safely. For example, you can still pass an input ``dict`` like ``{'input_0': np.zeros((2, 1))}``.

Apache MXNet (Incubating)
=========================

  * Requires: ``mxnet-neuron``
  	- Versions 1.5, 1.8
  * Input to ``compile``: ``tuple(sym, args, aux)``
  * Inputs: Tensors must be provided as ``mxnet.ndarray`` or ``numpy.ndarray``
