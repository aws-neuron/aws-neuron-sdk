.. _torch_neuronx_trace_api:

PyTorch Neuron (``torch-neuronx``) Tracing API for Inference
============================================================

.. py:function:: torch_neuronx.trace(func, example_inputs, *, compiler_workdir=None, compiler_args=None)

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

    :keyword str compiler_workdir: Work directory used by
       |neuronx-cc|. This can be useful for debugging and/or inspecting
       intermediary |neuronx-cc| outputs
    :keyword str,list[str] compiler_args: List of strings representing
       |neuronx-cc| compiler arguments. See :ref:`neuron-compiler-cli-reference-guide`
       for more information about compiler options.

    :returns: The traced :class:`~torch.jit.ScriptModule` with the embedded
       compiled Neuron graph. Operations in this module will execute on Neuron.
    :rtype: ~torch.jit.ScriptModule

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
