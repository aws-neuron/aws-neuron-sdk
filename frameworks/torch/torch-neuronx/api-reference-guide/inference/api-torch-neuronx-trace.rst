.. _torch_neuronx_trace_api:

PyTorch NeuronX Tracing API for Inference
===========================================

.. py:function:: torch_neuronx.trace(func, example_inputs, *_, input_output_aliases={}, compiler_workdir=None, compiler_args=None, partitioner_config=None, inline_weights_to_neff=True)
    
    Trace and compile operations in the ``func`` by executing it using
    ``example_inputs``.

    This function is similar to a :func:`torch.jit.trace` since it produces a
    :class:`~torch.jit.ScriptModule` that can be saved with
    :func:`torch.jit.save` and reloaded with :func:`torch.jit.load`. The
    resulting module is an optimized fused graph representation of the ``func``
    that is *only* compatible with Neuron.

    Tracing a module produces a more efficient *inference-only* version of the
    model. XLA Lazy Tensor execution should be used during training. See:
    :ref:`trace-vs-xla-lazytensor`

    .. warning::

        Currently this only supports |NeuronCore-v2| type instances
        (e.g. |trn1|, inf2). To compile models compatible with |NeuronCore-v1|
        (e.g. |inf1|), please see :func:`torch_neuron.trace`

    :arg ~torch.nn.Module,callable func: The function/module that that will be
       run using the ``example_inputs`` arguments in order to record the
       computation graph.
    :arg ~torch.Tensor,tuple[~torch.Tensor] example_inputs: A tuple of example
       inputs that will be passed to the ``func`` while tracing.

    :keyword dict input_output_aliases: Marks input tensors as state tensors
       which are device tensors. 
    :keyword str compiler_workdir: Work directory used by
       |neuronx-cc|. This can be useful for debugging and/or inspecting
       intermediary |neuronx-cc| outputs
    :keyword str,list[str] compiler_args: List of strings representing
       |neuronx-cc| compiler arguments. See :ref:`neuron-compiler-cli-reference-guide`
       for more information about compiler options.
    :keyword PartitionerConfig partitioner_config: A PartitionerConfig object,
        which can be optionally supplied if there are unsupported ops in the model 
        that need to be partitioned out to CPU.
    :keyword bool inline_weights_to_neff: A boolean indicating whether the weights should be
        inlined to the neff. If set to False, weights will be seperated from the neff.
        The default is `True`.

    :returns: The traced :class:`~torch.jit.ScriptModule` with the embedded
       compiled Neuron graph. Operations in this module will execute on Neuron.
    :rtype: ~torch.jit.ScriptModule

    .. warning::

      Behavior Change! The use of using args for kwargs is deprecated starting from release 2.15.0 (``torch-neuronx==1.13.1.1.12.0``).
      The current behavior is that a warning will be raised, but ``torch_neuronx.trace()`` will attempt to infer the keyword
      arguments. This is likely to become an error in future releases, so to avoid the warning/error, assign kwargs as kwargs and
      not args.

    .. rubric:: Notes

    This function records operations using `torch-xla`_ to create a HloModule
    representation of the ``func``. This fixed graph representation is
    compiled to the Neuron Executable File Format (NEFF) using the |neuronx-cc|
    compiler. The NEFF binary executable is embedded into an optimized
    :class:`~torch.jit.ScriptModule` for `torchscript`_ execution.

    In contrast to a regular :func:`torch.jit.trace` that produces a graph of
    many separate operations, tracing with Neuron produces a graph with a single
    fused operator that is executed entirely on device. In `torchscript`_
    this appears as a stateful ``neuron::Model`` component with an associated
    ``neuron::forward*`` operation.

    Tracing can be performed on any EC2 machine with sufficient memory and
    compute resources, but inference can only be executed on a Neuron instance.

    Unlike some devices (such as `torch-xla`_) that use
    :meth:`~torch.Tensor.to` to move :class:`~torch.nn.parameter.Parameter` and
    :class:`~torch.Tensor` data between CPU and device, upon loading a
    Neuron traced :class:`~torch.jit.ScriptModule`, the model binary executable
    is automatically moved to a NeuronCore. When the underlying
    ``neuron::Model`` is initialized after tracing or upon
    :func:`torch.jit.load`, it is loaded to a Neuron device without specifying
    a device or ``map_location`` argument.

    Furthermore, the Neuron traced :class:`~torch.jit.ScriptModule` expects
    to consume CPU tensors and produces CPU tensors. The underlying operation
    performs all data transfers to and from the Neuron device without explicit
    data movement. This is a significant difference from the training XLA
    device mechanics since XLA operations are no longer required to
    be recorded after a trace. See: :ref:`pytorch-neuronx-programming-guide`

    By *default*, when multiple NeuronCores are available, every Neuron traced
    model :class:`~torch.jit.ScriptModule` within in a process
    is loaded to each available NeuronCore in round-robin order. This is
    useful at deployment to fully utilize the Neuron hardware since it means
    that multiple calls to :func:`torch.jit.load` will attempt to load to each
    available NeuronCore in linear order. The default start device is chosen
    according to the |nrt-configuration|.

    A traced Neuron module has limitations that are not present in regular
    torch modules:

    - **Fixed Control Flow**: Similar to :func:`torch.jit.trace`, tracing a
      model with Neuron statically preserves control flow (i.e.
      ``if``/``for``/``while`` statements) and will not re-evaluate the branch
      conditions upon inference. If a model result is based on data-dependent
      control flow, the traced function may produce inaccurate results.
    - **Fixed Input Shapes**: After a function has been traced, the resulting
      :class:`~torch.jit.ScriptModule` will always expect to consume tensors
      of the same shape. If the tensor shapes used at inference differs
      from the tensor shapes used in the ``example_inputs``, this will result in
      an error. See: |bucketing|.
    - **Fixed Tensor Shapes**: The intermediate tensors within the
      ``func`` must always stay the same shape for the same shaped inputs. This
      means that certain operations which produce data-dependent
      sized tensors are not supported. For example, :func:`~torch.nonzero`
      produces a different tensor shape depending on the input data.
    - **Fixed Data Types**: After a model has been traced, the input, output,
      and intermediate data types cannot be changed without recompiling.
    - **Device Compatibility**: Due to Neuron using a specialized compiled
      format (NEFF), a model traced with Neuron can no longer be executed in any
      non-Neuron environment.
    - **Operator Support**: If an operator is unsupported by `torch-xla`_, then
      this will throw an exception.

    .. rubric:: Examples

    *Function Compilation*

    .. code-block:: python

        import torch
        import torch_neuronx
        def func(x, y):
            return 2 * x + y
        example_inputs = torch.rand(3), torch.rand(3)
        # Runs `func` with the provided inputs and records the tensor operations
        trace = torch.neuronx.trace(func, example_inputs)
        # `trace` can now be run with the TorchScript interpreter or saved
        # and loaded in a Python-free environment
        torch.jit.save(trace, 'func.pt')
        # Executes on a NeuronCore
        loaded = torch.jit.load('func.pt')
        loaded(torch.rand(3), torch.rand(3))
    
    *Module Compilation*

    .. code-block:: python

        import torch
        import torch_neuronx
        import torch.nn as nn
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)
            def forward(self, x):
                return self.conv(x) + 1
        model = Model()
        model.eval()
        example_inputs = torch.rand(1, 1, 3, 3)
        # Traces the forward method and constructs a `ScriptModule`
        trace = torch_neuronx.trace(model, example_inputs)
        torch.jit.save(trace, 'model.pt')
        # Executes on a NeuronCore
        loaded = torch.jit.load('model.pt')
        loaded(torch.rand(1, 1, 3, 3))

    *Weight Separated Module*

    .. code-block:: python

        import torch
        import torch_neuronx
        import torch.nn as nn

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x) + 1

        model = Model()
        model.eval()

        example_inputs = torch.rand(1, 1, 3, 3)

        # Traces the forward method and constructs a `ScriptModule`
        trace = torch_neuronx.trace(model, example_inputs,inline_weights_to_neff=False)

        # Model can be saved like a normally traced model
        torch.jit.save(trace, 'model.pt')

        # Executes on a NeuronCore like a normally traced model
        loaded = torch.jit.load('model.pt')
        loaded(torch.rand(1, 1, 3, 3))
    
    .. note::

      Weight Separated models can have its weights replaced via the `torch_neuronx.replace_weights` API.

.. _torch-neuronx-dynamic-batching:

Dynamic Batching
~~~~~~~~~~~~~~~~

.. py:function:: torch_neuronx.dynamic_batch(neuron_script)

    Enables a compiled Neuron model to be called with variable sized batches.

    When tracing with Neuron, usually a model can only consume tensors that are the same size as the example tensor used in the :func:`torch_neuronx.trace` call. Enabling dynamic batching allows a model to consume inputs that may be either smaller or larger than the original trace-time tensor size. Internally, dynamic batching splits & pads an input batch into chunks of size equal to the original trace-time tensor size. These chunks are passed to the underlying model(s). Compared to serial inference, the expected runtime scales by ``ceil(inference_batch_size / trace_batch_size) / neuron_cores``.
    
    This function modifies the ``neuron_script`` network in-place. The returned result is a reference to the modified input.

    Dynamic batching is only supported by chunking inputs along the 0th dimension. A network that uses a non-0 batch dimension is incompatible with dynamic batching. Upon inference, inputs whose shapes differ from the compile-time shape in a non-0 dimension will raise a ValueError. For example, take a model was traced with a single example input of size ``[2, 3, 5]``. At inference time, when dynamic batching is enabled, a batch of size ``[3, 3, 5]`` is *valid* while a batch of size ``[2, 7, 5]`` is *invalid* due to changing a non-0 dimension.

    Dynamic batching is only supported when the 0th dimension is the same size for all inputs. For example, this means that dynamic batching would not be applicable to a network which consumed two inputs with shapes ``[1, 2]`` and ``[3, 2]`` since the 0th dimension is different. Similarly, at inference time, the 0th dimension batch size for all inputs must be identical otherwise a ValueError will be raised.
    
    *Required Arguments*

    :arg ~torch.jit.ScriptModule neuron_script: The neuron traced :class:`~torch.jit.ScriptModule` with the
       embedded compiled neuron graph. This is the output of :func:`torch_neuronx.trace`.

    :returns: The traced :class:`~torch.jit.ScriptModule` with the embedded
       compiled neuron graph. The same type as the input, but with dynamic_batch enabled in the neuron graph.
    :rtype: ~torch.jit.ScriptModule

.. code-block:: python

    import torch
    import torch_neuronx
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 1, 3)

        def forward(self, x):
            return self.conv(x) + 1

    n = Net()
    n.eval()

    inputs = torch.rand(1, 1, 3, 3)
    inputs_batch_8 = torch.rand(8, 1, 3, 3)

    # Trace a neural network with input batch size of 1
    neuron_net = torch_neuronx.trace(n, inputs)

    # Enable the dynamic batch size feature so the traced network
    # can consume variable sized batch inputs
    neuron_net_dynamic_batch = torch_neuronx.dynamic_batch(neuron_net)

    # Run inference on inputs with batch size of 8
    # different than the batch size used in compilation (tracing)
    ouput_batch_8 = neuron_net_dynamic_batch(inputs_batch_8)

Graph Partitioner
~~~~~~~~~~~~~~~~~

.. py:function:: torch_neuronx.PartitionerConfig(*,trace_kwargs=None,model_support_percentage_threshold=0.5,min_subgraph_size=-1,max_subgraph_count=-1,ops_to_partition=None,analyze_parameters=None)

    Allows for Neuron to trace a model with unsupported operators and partition these operators to CPU.

    This model will contain subgraphs of Neuron and CPU submodules, but it is executed like one model,
    and can be saved and loaded like one model as well.

    The graph partitioner is customized using this class, and is *only* enabled (disabled by default) from the ``torch_neuronx.trace`` API by setting ``partitioner_config``
    keyword argument to this class. Below are the various configuration options.

    :arg Dict trace_kwargs: Used if you need to pass trace kwargs to the Neuron subgraphs, such as the
      ``compiler_workdir`` and/or ``compiler_args``. The default is ``None`` corresponding to the default trace args.
    
    :arg float model_support_percentage_threshold: A number between 0 to 1 representing
      the maximum allowed percentage of operators that must be supported.
      If the max is breached, the function will throw a ValueError.
      Default is ``0.5`` (i.e 50% of operators must be supported by Neuron)
    
    :arg int min_subgraph_size: The minimum number of operators in a subgraph.
      Can be ``>= 1`` or ``== -1``. If ``-1``, minimum subgraph size is not checked (i.e no minimum).
      If ``>= 1``, each subgraph must contain at least that many operators.
      If not, the graph partitioner will throw a ``ValueError``.
    
    :arg int max_subgraph_count: The maximum number of subgraphs in the partitioned model.
      Can be ``>= 1`` or ``== -1``. If ``-1``, max subgraph count is not checked (i.e no maximum).
      If ``>= 1``, the partitioned model must contain at most that many subgraphs.
      If not, the graph partitioner will throw a ``ValueError``.
    
    :arg Set[str] ops_to_partition: This is a set of strings of this structure "aten::<operator>".
      These are operators that will be partitioned to CPU regardless of Neuron support.
      The default is ``None`` (i.e no additional operators will be partitioned).

    :arg Dict analyze_parameters: This is a dictionary of kwargs used in ``torch_neuronx.analyze()``.
      NOTE: Not all kwargs in ``torch_neuronx.analyze()`` are supported
      in the graph partitioner.
      The following kwargs in analyze are supported for use in the graph partitioenr.
          a) compiler_workdir
          b) additional_ignored_ops
          c) max_workers
      The default is ``None``, corresponding to the default analyze arguments.

    :returns: The  :class:`~torch_neuronx.PartitionerConfig` with the configuration for the graph partitioner.
    :rtype: ~torch_neuronx.PartitionerConfig

.. rubric:: Examples

.. _graph_partitioner_example_default_usage:

This example demonstrates using the graph partitioner.

The below model is a simple MLP model with sorted log softmax output.
The sort operator, ``torch.sort()`` or ``aten::sort``, is not supported
by ``neuronx-cc`` at this time, so the graph partitioner will partition
out the sort operator to CPU.

.. code-block:: python

  import torch
  import torch_neuronx
  import torch.nn as nn

  import logging
  
  # adjust logger level to see what the partitioner is doing
  logger = logging.getLogger("Neuron")

  class MLP(nn.Module):
      def __init__(
          self, input_size=28 * 28, output_size=10, layers=[4096, 2048]
      ):
          super(MLP, self).__init__()
          self.fc1 = nn.Linear(input_size, layers[0])
          self.fc2 = nn.Linear(layers[0], layers[1])
          self.fc3 = nn.Linear(layers[1], output_size)
          self.relu = nn.ReLU()

      def forward(self, x):
          f1 = self.fc1(x)
          r1 = self.relu(f1)
          f2 = self.fc2(r1)
          r2 = self.relu(f2)
          f3 = self.fc3(r2)
          out = torch.log_softmax(f3, dim=1)
          sort_out,_ = torch.sort(out)
          return sort_out

  n = MLP()
  n.eval()

  inputs = torch.rand(32,784)

  # Configure the graph partitioner with the default values
  partitioner_config = torch_neuronx.PartitionerConfig()

  # Trace a neural network with graph partitioner enabled
  neuron_net = torch_neuronx.trace(n, inputs, partitioner_config=partitioner_config)

  # Run inference on the partitioned model
  output = neuron_net(inputs)

.. note::
  Dynamic batching has a case-by-case support with partitioned
  models, because it is highly dependent on how the
  final partition scheme looks like.

.. |neuron-cc| replace:: :ref:`neuron-cc <neuron-compiler-cli-reference>`
.. |neuronx-cc| replace:: :ref:`neuronx-cc <neuron-compiler-cli-reference-guide>`
.. |NeuronCore-v1| replace:: :ref:`NeuronCore-v1 <neuroncores-v1-arch>`
.. |NeuronCore-v2| replace:: :ref:`NeuronCore-v2 <neuroncores-v2-arch>`

.. |HloModule| replace:: HloModule

.. |inf1| replace:: :ref:`inf1 <aws-inf1-arch>`
.. |trn1| replace:: :ref:`trn1 <aws-trn1-arch>`

.. |bucketing| replace:: :ref:`bucketing_app_note`
.. |nrt-configuration| replace:: :ref:`nrt-configuration`

.. _torch-xla: https://github.com/pytorch/xla
.. _torchscript: https://pytorch.org/docs/stable/jit.html