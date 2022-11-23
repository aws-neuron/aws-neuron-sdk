"""
.. _torch_neuron_core_placement_api:

Experimental: NeuronCore Placement APIs
=======================================

.. warning::

    The following functionality is experimental and **will not be supported** in
    future releases of the NeuronSDK. This module serves only as a preview for
    future functionality. In future releases, equivalent functionality may
    be moved directly to the :code:`torch_neuron` module and will no longer be
    available in the :code:`torch_neuron.experimental` module.

Functions which enable placement of :class:`torch.jit.ScriptModule` to specific
NeuronCores. Two sets of functions are provided which can be used
interchangeably but have different performance characteristics and advantages:

- The :func:`~torch_neuron.experimental.multicore_context` &
  :func:`~torch_neuron.experimental.neuron_cores_context` functions are context
  managers that allow a model to be placed on a given NeuronCore at
  :func:`torch.jit.load` time. These functions are the most efficient way of
  loading a model since the model is loaded directly to a NeuronCore. The
  alternative functions described below require that a model is unloaded from
  one core and then reloaded to another.
- The :func:`~torch_neuron.experimental.set_multicore` &
  :func:`~torch_neuron.experimental.set_neuron_cores` functions allow a model
  that has already been loaded to a NeuronCore to be moved to a different
  NeuronCore. This functionality is less efficient than directly loading a model
  to a NeuronCore within a context manager but allows device placement to be
  fully dynamic at runtime. This is analogous to the :meth:`torch.nn.Module.to`
  function for device placement.

.. important::

    A prerequisite to enable placement functionality is that
    the loaded :class:`torch.jit.ScriptModule` has already been compiled with
    the :func:`torch_neuron.trace` API. Attempting to place a regular
    :class:`torch.nn.Module` onto a NeuronCore prior to compilation will do
    nothing.
"""
import contextlib


def set_neuron_cores(trace: 'torch.jit.ScriptModule', start_nc: int=-1, nc_count: int=-1):
    """
    Set the NeuronCore start/count for all Neuron subgraphs in a torch Module.

    This will unload the model from an existing NeuronCore if it is already
    loaded.

    *Requires Torch 1.8+*

    Arguments:
        trace: A torch module which contains one or more Neuron subgraphs.
        start_nc: The starting NeuronCore index where the Module is placed. The
            value ``-1`` automatically loads to the optimal NeuronCore (least
            used). Note that this index is always relative to NeuronCores
            visible to this process.
        nc_count: The number of NeuronCores to use. The value ``-1`` will load
            a model to exactly the number of cores required by that model (1 for
            most models, >1 when using NeuronCore Pipeline). If ``nc_count``
            is greater than the number of NeuronCores required by the
            model, the model will be replicated across multiple
            NeuronCores. ``(replications = floor(nc_count / cores_per_model))``

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized.
        ValueError: If the ``nc_count`` is an invalid number of NeuronCores.

    Examples:

        *Single Load*: Move a model to the first visible NeuronCore after
        loading.

        >>> model = torch.jit.load('example_neuron_model.pt')
        >>> torch_neuron.experimental.set_neuron_cores(model, start_nc=0, nc_count=1)
        >>> model(example) # Executes on NeuronCore 0
        >>> model(example) # Executes on NeuronCore 0
        >>> model(example) # Executes on NeuronCore 0

        *Multiple Core Replication*: Replicate a model to 2 NeuronCores after
        loading. This allows a single :class:`torch.jit.ScriptModule` to
        use multiple NeuronCores by running round-robin executions.

        >>> model = torch.jit.load('example_neuron_model.pt')
        >>> torch_neuron.experimental.set_neuron_cores(model, start_nc=2, nc_count=2)
        >>> model(example) # Executes on NeuronCore 2
        >>> model(example) # Executes on NeuronCore 3
        >>> model(example) # Executes on NeuronCore 2

        *Multiple Model Load*: Move and pin 2 models to separate NeuronCores.
        This causes each :class:`torch.jit.ScriptModule` to always execute on
        a specific NeuronCore.

        >>> model1 = torch.jit.load('example_neuron_model.pt')
        >>> torch_neuron.experimental.set_neuron_cores(model1, start_nc=2)
        >>> model2 = torch.jit.load('example_neuron_model.pt')
        >>> torch_neuron.experimental.set_neuron_cores(model2, start_nc=0)
        >>> model1(example) # Executes on NeuronCore 2
        >>> model1(example) # Executes on NeuronCore 2
        >>> model2(example) # Executes on NeuronCore 0
        >>> model2(example) # Executes on NeuronCore 0
    """


def set_multicore(trace: 'torch.jit.ScriptModule'):
    """
    Loads all Neuron subgraphs in a torch Module to all visible NeuronCores.

    This loads each Neuron subgraph within a :class:`torch.jit.ScriptModule`
    to multiple NeuronCores without requiring multiple calls to
    :func:`torch.jit.load`. This allows a single
    :class:`torch.jit.ScriptModule` to use multiple NeuronCores for
    concurrent threadsafe inferences. Executions use a round-robin strategy
    to distribute across NeuronCores.

    This will unload the model from an existing NeuronCore if it is already
    loaded.

    *Requires Torch 1.8+*

    Arguments:
        trace: A torch module which contains one or more Neuron subgraphs.

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized.

    Examples:

        *Multiple Core Replication*: Move a model across all visible
        NeuronCores after loading. This allows a single
        :class:`torch.jit.ScriptModule` to use all NeuronCores by
        running round-robin executions.

        >>> model = torch.jit.load('example_neuron_model.pt')
        >>> torch_neuron.experimental.set_multicore(model)
        >>> model(example) # Executes on NeuronCore 0
        >>> model(example) # Executes on NeuronCore 1
        >>> model(example) # Executes on NeuronCore 2
    """


@contextlib.contextmanager
def neuron_cores_context(start_nc: int=-1, nc_count: int=-1):
    """
    A context which sets the NeuronCore start/count for all Neuron subgraphs.

    Any calls to :func:`torch.jit.load` will cause any underlying Neuron
    subgraphs to load to the specified NeuronCores within this context.
    This context manager only needs to be used during the model load.
    After loading, inferences do not need to occur in this context in order
    to use the correct NeuronCores.

    Note that this context is *not* threadsafe. Using multiple core placement
    contexts from multiple threads may not correctly place models.

    Arguments:
        start_nc: The starting NeuronCore index where the Module is placed. The
            value ``-1`` automatically loads to the optimal NeuronCore (least
            used). Note that this index is always relative to NeuronCores
            visible to this process.
        nc_count: The number of NeuronCores to use. The value ``-1`` will load
            a model to exactly the number of cores required by that model (1 for
            most models, >1 when using NeuronCore Pipeline). If ``nc_count``
            is greater than the number of NeuronCores required by the
            model, the model will be replicated across multiple
            NeuronCores. ``(replications = floor(nc_count / cores_per_model))``

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized.
        ValueError: If the ``nc_count`` is an invalid number of NeuronCores.

    Examples:

        *Single Load*: Directly load a model from disk to the first visible
        NeuronCore.

        >>> with torch_neuron.experimental.neuron_cores_context(start_nc=0, nc_count=1):
        >>>     model = torch.jit.load('example_neuron_model.pt')
        >>> model(example) # Executes on NeuronCore 0
        >>> model(example) # Executes on NeuronCore 0
        >>> model(example) # Executes on NeuronCore 0

        *Multiple Core Replication*: Directly load a model from disk to 2
        NeuronCores. This allows a single :class:`torch.jit.ScriptModule` to
        use multiple NeuronCores by running round-robin executions.

        >>> with torch_neuron.experimental.neuron_cores_context(start_nc=2, nc_count=2):
        >>>     model = torch.jit.load('example_neuron_model.pt')
        >>> model(example) # Executes on NeuronCore 2
        >>> model(example) # Executes on NeuronCore 3
        >>> model(example) # Executes on NeuronCore 2

        *Multiple Model Load*: Directly load 2 models from disk and pin them to
        separate NeuronCores. This causes each :class:`torch.jit.ScriptModule`
        to always execute on a specific NeuronCore.

        >>> with torch_neuron.experimental.neuron_cores_context(start_nc=2):
        >>>     model1 = torch.jit.load('example_neuron_model.pt')
        >>> with torch_neuron.experimental.neuron_cores_context(start_nc=0):
        >>>     model2 = torch.jit.load('example_neuron_model.pt')
        >>> model1(example) # Executes on NeuronCore 2
        >>> model1(example) # Executes on NeuronCore 2
        >>> model2(example) # Executes on NeuronCore 0
        >>> model2(example) # Executes on NeuronCore 0
    """


@contextlib.contextmanager
def multicore_context():
    """
    A context which loads all Neuron subgraphs to all visible NeuronCores.

    This loads each Neuron subgraph within a :class:`torch.jit.ScriptModule`
    to multiple NeuronCores without requiring multiple calls to
    :func:`torch.jit.load`. This allows a single
    :class:`torch.jit.ScriptModule` to use multiple NeuronCores for
    concurrent threadsafe inferences. Executions use a round-robin strategy
    to distribute across NeuronCores.

    Any calls to :func:`torch.jit.load` will cause any underlying Neuron
    subgraphs to load to the specified NeuronCores within this context.
    This context manager only needs to be used during the model load.
    After loading, inferences do not need to occur in this context in order
    to use the correct NeuronCores.

    Note that this context is *not* threadsafe. Using multiple core placement
    contexts from multiple threads may not correctly place models.

    Raises:
        RuntimeError: If the Neuron runtime cannot be initialized.

    Examples:

        *Multiple Core Replication*: Directly load a model to all visible
        NeuronCores. This allows a single  :class:`torch.jit.ScriptModule`
        to use all NeuronCores by running round-robin executions.

        >>> with torch_neuron.experimental.multicore_context():
        >>>     model = torch.jit.load('example_neuron_model.pt')
        >>> model(example) # Executes on NeuronCore 0
        >>> model(example) # Executes on NeuronCore 1
        >>> model(example) # Executes on NeuronCore 2
    """
