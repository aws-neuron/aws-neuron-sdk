.. _neuronperf_framework_notes:

.. meta::
   :noindex:
   :nofollow:
   :description: This tutorial for the AWS Neuron SDK is currently archived and not maintained. It is provided for reference only.
   :date-modified: 12-02-2025

==========================
NeuronPerf Framework Notes
==========================

PyTorch
=======

  * Requires: ``torch-neuron`` or ``torch-neuronx``
	- Versions: 1.7.x, 1.8.x, 1.9.x, 1.10.x, 1.11.x, 1.12.x, 1.13.x
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

  * Requires: ``tensorflow-neuron`` or ``tensorflow-neuronx``
  	- Versions: All
  * Input to ``compile``: ``tf.keras.Model``
  * Model inputs: Tensors must be provided as ``numpy.ndarray``

.. note::

	Although TensorFlow *tensors* must be ``ndarray``, this doesn't stop you from wrapping them inside of data structures that traverse process boundaries safely. For example, you can still pass an input ``dict`` like ``{'input_0': np.zeros((2, 1))}``.

Apache MXNet
=============

  * Requires: ``mxnet-neuron``
  	- Versions 1.5, 1.8
  * Input to ``compile``: ``tuple(sym, args, aux)``
  * Inputs: Tensors must be provided as ``mxnet.ndarray`` or ``numpy.ndarray``
