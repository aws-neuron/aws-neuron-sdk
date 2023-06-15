.. _api_torch_neuron_dataparallel_api:

torch.neuron.DataParallel API
=============================

The :func:`torch.neuron.DataParallel` Python API implements data parallelism on
:class:`~torch.jit.ScriptModule` models created by the
:ref:`torch_neuron_trace_api`.
This function is analogous to :class:`~torch.nn.DataParallel` in PyTorch.
The :ref:`torch-neuron-dataparallel-app-note` application note provides an
overview of how :func:`torch.neuron.DataParallel` can be used to improve
the performance of inference workloads on Inferentia.

.. py:function:: torch.neuron.DataParallel(model, device_ids=None, dim=0)

    Applies data parallelism by replicating the model on
    available NeuronCores and distributing data across the different
    NeuronCores for parallelized inference.

    By default, DataParallel will use all available NeuronCores
    allocated for the current process for parallelism. DataParallel will
    apply parallelism on ``dim=0`` if ``dim`` is not specified.

    DataParallel automatically enables
    :ref:`dynamic batching <dynamic_batching_description>` on
    eligible models if ``dim=0``. Dynamic batching can be dsiabled using
    :func:`torch.neuron.DataParallel.disable_dynamic_batching`.
    If dynamic batching is not enabled, the batch size at compilation-time must
    be equal to the batch size at inference-time divided by the number of
    NeuronCores being used. Specifically, the following must be true when
    dynamic batching is disabled:
    ``input.shape[dim] / len(device_ids) == compilation_input.shape[dim]``.
    DataParallel will throw a warning if dynamic batching cannot be enabled.

    DataParallel will try load all of a model’s NEFFs onto
    a single NeuronCore, only if all of the NEFFs can fit on a single
    NeuronCore. DataParallel does not currently support models that
    have been compiled with :ref:`neuroncore-pipeline`.

    :func:`torch.neuron.DataParallel` requires PyTorch >= 1.8.

    *Required Arguments*

    :arg ~torch.jit.ScriptModule model: Model created by the
        :ref:`torch_neuron_trace_api`
        to be parallelized.

    *Optional Arguments*

    :arg list device_ids: List of :obj:`int` or ``'nc:#'`` that specify the
        NeuronCores to use for parallelization (default: all NeuronCores).
        Refer to the :ref:`device_ids note <device_ids_note>` for a description
        of how ``device_ids`` indexing works.
    :arg int dim: Dimension along which the input tensor is scattered across
        NeuronCores (default ``dim=0``).

    *Attributes*

    :arg int num_workers: Number of worker threads used for
        multithreaded inference (default: ``2 * number of NeuronCores``).
    :arg int split_size: Size of the input chunks
        (default: ``max(1, input.shape[dim] // number of NeuronCores)``).


.. py:function:: torch.neuron.DataParallel.disable_dynamic_batching()

    Disables automatic dynamic batching on the DataParallel module. See
    :ref:`Dynamic batching disabled <dataparallel_example_disable_dynamic_batching_api>`
    for example of how DataParallel can be used with dynamic batching disabled.
    Use as follows:

        >>> model_parallel = torch.neuron.DataParallel(model_neuron)
        >>> model_parallel.disable_dynamic_batching()

.. _device_ids_note:

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

    As an example, assume DataParallel is run on an inf1.6xlarge, which
    contains four Inferentia chips each of which contains four NeuronCores:

    * If ``NEURON_RT_VISIBLE_CORES`` is not set, a single process can access
      all 16 NeuronCores. Thus specifying ``device_ids=["nc:0"]`` will
      correspond to chip0:core0 and ``device_ids=["nc:14"]`` will correspond
      to chip3:core2.

    * However, if two processes are launched where: process 1 has
      ``NEURON_RT_VISIBLE_CORES=0-6`` and process 2 has
      ``NEURON_RT_VISIBLE_CORES=7-15``, ``device_ids=["nc:14"]``
      cannot be specified in either process. Instead, chip3:core2 can only be
      accessed in process 2. Additionally, chip3:core2 is specified in process 2
      with ``device_ids=["nc:7"]``. Furthermore, in process 1,
      ``device_ids=["nc:0"]`` would correspond to chip0:core0; in process 2
      ``device_ids=["nc:0"]`` would correspond to chip1:core3.


Examples
--------

The following sections provide example usages of the
:func:`torch.neuron.DataParallel` module.

Default usage
^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuron/torch-neuron-dataparallel-example-default.rst

Specifying NeuronCores
^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuron/torch-neuron-dataparallel-example-specify-ncs.rst

DataParallel with dim != 0
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuron/torch-neuron-dataparallel-example-dim-neq-zero.rst

Dynamic batching
^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuron/torch-neuron-dataparallel-example-dynamic-batching.rst

.. _dataparallel_example_disable_dynamic_batching_api:

Dynamic batching disabled
^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: /frameworks/torch/torch-neuron/torch-neuron-dataparallel-example-disable-dynamic-batching.rst

Full tutorial with torch.neuron.DataParallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For an end-to-end tutorial that uses DataParallel, see the
:ref:`PyTorch Resnet Tutorial </src/examples/pytorch/resnet50.ipynb>`.
