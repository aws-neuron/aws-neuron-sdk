.. _torch-neuronx-dataparallel-app-note:

Data Parallel Inference on torch_neuronx
=======================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
------------

This guide introduces :func:`torch_neuronx.DataParallel`, a Python API that
implements data parallelism on :class:`~torch.jit.ScriptModule` models created by the
:ref:`torch_neuronx_trace_api`.
The following sections explain how data parallelism can improve the performance of
inference workloads on Inferentia, including how :func:`torch_neuronx.DataParallel`
uses dynamic batching to run inference on variable input sizes. It covers an
overview of the :func:`torch_neuronx.DataParallel` module and provides a few
:ref:`example data parallel applications <data_parallel_examples_torch_neuronx>`.

Data parallel inference
-------------------------

Data Parallelism is a form of parallelization across multiple devices or cores,
referred to as nodes. Each node contains the same model and parameters, but
data is distributed across the different nodes. By distributing the
data across multiple nodes, data parallelism reduces the total
execution time of large batch size inputs compared to sequential execution.
Data parallelism works best for smaller models in latency sensitive
applications that have large batch size requirements.


torch_neuronx.DataParallel
-------------------------

To fully leverage the Inferentia hardware, we want to use all available
NeuronCores. An inf2.xlarge and inf2.8xlarge have two NeuronCores, an
inf2.24xlarge has 12 NeuronCores, and an inf2.48xlarge has 24 NeuronCores.
For maximum performance on Inferentia hardware, we can use
:func:`torch_neuronx.DataParallel` to utilize all available NeuronCores.

:func:`torch_neuronx.DataParallel` implements data parallelism at the module
level by replicating the Neuron model on all available NeuronCores
and distributing data across the different cores for parallelized inference.
This function is analogous to :class:`~torch.nn.DataParallel` in PyTorch.
:func:`torch_neuronx.DataParallel` requires PyTorch >= 1.8.

The following sections provide an overview of some of the features
of :func:`torch_neuronx.DataParallel` that enable maximum performance on
Inferentia.

NeuronCore selection
^^^^^^^^^^^^^^^^^^^^

By default, DataParallel will try to use all NeuronCores allocated to the
current process to fully saturate the Inferentia hardware for maximum performance.
It is more efficient to make the batch dimension divisible by the number of
NeuronCores. This will ensure that NeuronCores are not left idle during
parallel inference and the Inferentia hardware is fully utilized.

In some applications, it is advantageous to use a subset of the
available NeuronCores for DataParallel inference. DataParallel has a
``device_ids`` argument that accepts a list of :obj:`int` or ``'nc:#'``
that specify the NeuronCores to use for parallelization. See
:ref:`Specifying NeuronCores <dataparallel_example_specify_ncs_torch_neuronx>`
for an example of how to use ``device_ids`` argument.

Batch dim
^^^^^^^^^

DataParallel accepts a ``dim`` argument that denotes the batch dimension used
to split the input data for distributed inference. By default,
DataParalell splits the inputs on ``dim = 0`` if the ``dim`` argument is not
specified. For applications with a non-zero batch dim, the ``dim`` argument
can be used to specify the inference-time input batch dimension.
:ref:`DataParallel with dim ! = 0 <dataparallel_example_dim_neq_zero_torch_neuronx>` provides an
example of data parallel inference on inputs with batch dim = 2.

.. _dynamic_batching_description_torch_neuronx:

Dynamic batching
^^^^^^^^^^^^^^^^

Batch size has a direct impact on model performance. The Inferentia chip is optimized
to run with small batch sizes. This means that a Neuron compiled model can outperform
a GPU model, even if running single digit batch sizes.

As a general best practice, we recommend optimizing your model's throughput by
compiling the model with a small batch size and gradually increasing it to
find the peak throughput on Inferentia.

Dynamic batching is a feature that allows you to use tensor batch sizes that the
Neuron model was not originally compiled against. This is necessary because the
underlying Inferentia hardware will always execute inferences with the batch
size used during compilation. Fixed batch size execution allows tuning the
input batch size for optimal performance. For example, batch size 1 may be
best suited for an ultra-low latency on-demand inference application, while
batch size > 1 can be used to maximize throughput for offline inferencing.
Dynamic batching is implemented by slicing large input tensors into chunks
that match the batch size used during the :func:`torch_neuronx.trace` compilation call.

The :func:`torch_neuronx.DataParallel` class automatically enables dynamic batching on
eligible models. This allows us to run inference in applications that have
inputs with a variable batch size without needing to recompile the model. See
:ref:`Dynamic batching <dataparallel_example_dynamic_batching_torch_neuronx>` for an example
of how DataParallel can be used to run inference on inputs with a dynamic batch
size without needing to recompile the model.

Dynamic batching using small batch sizes can result in sub-optimal throughput
because it involves slicing tensors into chunks and iteratively sending data
to the hardware. Using a larger batch size at compilation time can use the
Inferentia hardware more efficiently in order to maximize throughput. You can
test the tradeoff between individual request latency and total throughput by
fine-tuning the input batch size.

Automatic batching in the DataParallel module can be disabled using the
``disable_dynamic_batching()`` function as follows:

.. code-block:: python

   >>> model_parallel = torch_neuronx.DataParallel(model_neuron)
   >>> model_parallel.disable_dynamic_batching()

If dynamic batching is disabled, the compile-time batch size must be equal to
the inference-time batch size divided by the number of NeuronCores.
:ref:`DataParallel with dim != 0 <dataparallel_example_dim_neq_zero_torch_neuronx>` and
:ref:`Dynamic batching disabled <dataparallel_example_disable_dynamic_batching_torch_neuronx>`
provide examples of running DataParallel inference with dynamic batching
disabled.


Performance optimizations
^^^^^^^^^^^^^^^^^^^^^^^^^

The DataParallel module has a ``num_workers`` attribute that can be used to
specify the number of worker threads used for multithreaded inference. By
default, ``num_workers = 2 * number of NeuronCores``. This value can be
fine tuned to optimize DataParallel performance.

DataParallel has a ``split_size`` attribute that dictates the size of the input
chunks that are distributed to each NeuronCore. By default,
``split_size = max(1, input.shape[dim] // number of NeuronCores)``. This value
can be modified to optimally match the inference input chunk size with the
compile-time batch size.

.. _data_parallel_examples_torch_neuronx:

Examples
--------

The following sections provide example usages of the
:func:`torch_neuronx.DataParallel` module.


.. _dataparallel_example_default_torch_neuronx:

Default usage
^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/torch-neuronx-dataparallel-example-default.rst

.. _dataparallel_example_specify_ncs_torch_neuronx:

Specifying NeuronCores
^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/torch-neuronx-dataparallel-example-specify-ncs.rst


.. _dataparallel_example_dim_neq_zero_torch_neuronx:

DataParallel with dim != 0
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/torch-neuronx-dataparallel-example-dim-neq-zero.rst


.. _dataparallel_example_dynamic_batching_torch_neuronx:

Dynamic batching
^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/torch-neuronx-dataparallel-example-dynamic-batching.rst


.. _dataparallel_example_disable_dynamic_batching_torch_neuronx:

Dynamic batching disabled
^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/torch-neuronx-dataparallel-example-disable-dynamic-batching.rst

