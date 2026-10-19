.. meta::
   :robots: noindex, nofollow
   :description: Archived FAQ about using ONNX models with Neuron through torch-neuron.
   :date-modified: 2026-10-13

:nosearch:

.. _onnx-faq:

ONNX FAQ (archived)
-------------------

.. warning::

   This FAQ is archived and is no longer maintained. It recommends ``torch_neuron.trace``, which is part of the retired torch-neuron (Inf1) framework. For current PyTorch support, see :doc:`TorchNeuron </torch/index>`.

.. contents:: Table of contents
   :local:
   :depth: 1


Can I use ONNX models with Neuron ? If not, what should I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AWS Neuron does not directly support compilation of models in the ONNX file format. The recommended way to compile a model that is in the ONNX file format is to first convert the model to PyTorch using a publicly available tool
like  `onnx2pytorch <https://github.com/ToriML/onnx2pytorch>`_ . Once the ONNX model is converted to PyTorch, it can then be compiled with the :func:`torch_neuron.trace` function to produce a model that can run on Neuron.


