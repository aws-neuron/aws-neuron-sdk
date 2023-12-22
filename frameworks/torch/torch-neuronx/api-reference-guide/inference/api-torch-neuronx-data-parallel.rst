.. _torch_neuronx_dataparallel_api:

PyTorch NeuronX DataParallel API
==================================

The :func:`torch_neuronx.DataParallel` Python API implements data parallelism on
:class:`~torch.jit.ScriptModule` models created by 
:ref:`torch_neuronx_trace_api`.
This function is analogous to :class:`~torch.nn.DataParallel` in PyTorch.
The :ref:`torch-neuronx-dataparallel-app-note` application note provides an
overview of how :func:`torch_neuronx.DataParallel` can be used to improve
the performance of inference workloads on Inferentia.

.. py:function:: torch_neuronx.DataParallel(model, device_ids=None, dim=0, set_dynamic_batching=True)

    Applies data parallelism by replicating the model on
    available NeuronCores and distributing data across the different
    NeuronCores for parallelized inference.

    By default, DataParallel will use all available NeuronCores
    allocated for the current process for parallelism. DataParallel will
    apply parallelism on ``dim=0`` if ``dim`` is not specified.

    DataParallel automatically enables
    :ref:`dynamic batching <dynamic_batching_description_torch_neuronx>` on
    eligible models if ``dim=0``. Dynamic batching can be disabled using
    :func:`torch_neuronx.DataParallel.disable_dynamic_batching`, or by setting
    ``set_dynamic_batching=False`` when initializing the DataParallel object.
    If dynamic batching is not enabled, the batch size at compilation-time must
    be equal to the batch size at inference-time divided by the number of
    NeuronCores being used. Specifically, the following must be true when
    dynamic batching is disabled:
    ``input.shape[dim] / len(device_ids) == compilation_input.shape[dim]``.

    :func:`torch.neuron.DataParallel` requires PyTorch >= 1.8.

    *Required Arguments*

    :arg ~torch.jit.ScriptModule model: Model created by the
        :ref:`torch_neuronx_trace_api` to be parallelized.

    *Optional Arguments*

    :arg list device_ids: List of :obj:`int` or ``'nc:#'`` that specify the
        NeuronCores to use for parallelization (default: all NeuronCores).
        Refer to the :ref:`device_ids note <device_ids_note_torch_neuronx>` for a description
        of how ``device_ids`` indexing works.
    :arg int dim: Dimension along which the input tensor is scattered across
        NeuronCores (default ``dim=0``).
    :arg bool set_dynamic_batching: Whether to enable dynamic batching.

    *Attributes*

    :arg int num_workers: Number of worker threads used for
        multithreaded inference (default: ``2 * number of NeuronCores``).
    :arg int split_size: Size of the input chunks
        (default: ``max(1, input.shape[dim] // number of NeuronCores)``).


.. py:function:: torch.neuron.DataParallel.disable_dynamic_batching()

    Disables automatic dynamic batching on the DataParallel module. See
    :ref:`Dynamic batching disabled <dataparallel_example_disable_dynamic_batching_api_torch_neuronx>`
    for example of how DataParallel can be used with dynamic batching disabled.
    Use as follows:

        >>> model_parallel = torch_neuronx.DataParallel(model_neuron)
        >>> model_parallel.disable_dynamic_batching()

.. _device_ids_note_torch_neuronx:

.. note::

    ``device_ids`` uses per-process NeuronCore granularity and zero-based
    indexing. Per-process granularity means that each Python process "sees"
    its own view of the world. Specifically, this means that ``device_ids``
    only "sees" the NeuronCores that are allocated for the current process.
    Zero-based indexing means that each Python process will index its
    allocated NeuronCores starting at 0, regardless of the "global" index of
    the NeuronCores. Zero-based indexing makes it possible to redeploy the exact
    same code unchanged in different process. This behavior is analogous to
    the ``device_ids`` argument in the PyTorch
    :class:`~torch.nn.DataParallel` function.

    As an example, assume DataParallel is run on an inf2.48xlarge, which
    contains 12 Inferentia chips each of which contains two NeuronCores:

    * If ``NEURON_RT_VISIBLE_CORES`` is not set, a single process can access
      all 24 NeuronCores. Thus specifying ``device_ids=["nc:0"]`` will
      correspond to chip0:core0 and ``device_ids=["nc:13"]`` will correspond
      to chip6:core1.

    * However, if two processes are launched where: process 1 has
      ``NEURON_RT_VISIBLE_CORES=0-11`` and process 2 has
      ``NEURON_RT_VISIBLE_CORES=12-23``, ``device_ids=["nc:13"]``
      cannot be specified in either process. Instead, chip6:core1 can only be
      accessed in process 2. Additionally, chip6:core1 is specified in process 2
      with ``device_ids=["nc:1"]``. Furthermore, in process 1,
      ``device_ids=["nc:0"]`` would correspond to chip0:core0; in process 2
      ``device_ids=["nc:0"]`` would correspond to chip6:core0.


Examples
--------

The following sections provide example usages of the
:func:`torch_neuronx.DataParallel` module.

Default usage
^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/torch-neuronx-dataparallel-example-default.rst

Specifying NeuronCores
^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/torch-neuronx-dataparallel-example-specify-ncs.rst

DataParallel with dim != 0
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/torch-neuronx-dataparallel-example-dim-neq-zero.rst

Dynamic batching
^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/torch-neuronx-dataparallel-example-dynamic-batching.rst

.. _dataparallel_example_disable_dynamic_batching_api_torch_neuronx:

Dynamic batching disabled
^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuronx/torch-neuronx-dataparallel-example-disable-dynamic-batching.rst

