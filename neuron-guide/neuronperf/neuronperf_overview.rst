.. _neuronperf_overview:

===================
NeuronPerf Overview
===================

NeuronPerf is a lightweight Python library that can help you easily benchmark your models with Neuron hardware.

NeuronPerf supports Neuron releases for PyTorch, Tensorflow, and MXNet. It is used internally by the Neuron team to generate performance benchmarking numbers.

When interacting with NeuronPerf, you will typically import the base package along with one of the submodule wrappers, for example:

.. code:: python

	import neuronperf
	import neuronperf.torch

You may then benchmark and/or compile one or more models with NeuronPerf. For example,

.. code:: python

	reports = neuronperf.torch.benchmark(model, inputs, ...)

The ``compile`` and ``benchmark`` methods must be accessed through one of the supported framework submodules.

Benchmarking
============

All NeuronPerf ``benchmark`` calls require a minimum of two arguments:

	1. A filename
	2. Inputs

The filename may refer to:

	1. A Neuron-compiled model (e.g. ``my_model.pt``)
	2. A :ref:`Model Index <neuronperf_model_index_guide>`.

A Model Index is useful for benchmarking more than one model in a single session.

Compiling
=========

NeuronPerf also provides a standard interface to all Neuron frameworks through the ``compile`` API.

.. code:: python

	model_index = neuronperf.torch.compile(model, inputs, ...)

This is completely optional. You may use the standard compilation guides for supported frameworks.

Next Steps
==========

Take a look at the simple :ref:`neuronperf_examples`, :ref:`neuronperf_benchmark_guide`, :ref:`neuronperf_compile_guide`, and :ref:`neuronperf_api`.