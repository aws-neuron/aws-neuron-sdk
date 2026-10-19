.. _error-code-enki003:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error ENKI003.

NCC_ENKI003
===========

**Error message**: NKI kernel <kernel> was traced for <num_funcs> NeuronCores but the graph is compiled for LNC=<lnc_count>; baking only core 0's function would silently produce a partial output.

This error occurs when a custom NKI (Neuron Kernel Interface) kernel in your model was traced for more NeuronCores than the number of Logical Neuron Cores the graph is being compiled for. Only the part of the kernel that computes the first NeuronCore's slice of the output could be included in the compiled model; the remaining parts would be dropped, leaving the kernel's output only partially computed. The compiler stops with this error instead of producing a result that would be silently incorrect.

How LNC and NKI kernels interact
--------------------------------

A Logical Neuron Core (LNC) is a software abstraction that groups one or more physical NeuronCores together. The number of Logical Neuron Cores a graph is compiled for is controlled by the ``NEURON_LOGICAL_NC_CONFIG`` environment variable, or by the ``--lnc`` flag when calling ``neuronx-cc`` directly.

When your model uses a custom NKI kernel, the framework (PyTorch or JAX) traces the kernel for a specific number of NeuronCores. A kernel traced for two NeuronCores contains two separate pieces of work, each computing a different part of the kernel's output (for example, one half each). A kernel traced for a single NeuronCore computes its whole output on its own. For the kernel to produce a complete, correct result, the number of NeuronCores it was traced for must equal the LNC the graph is compiled for.

Why the compiler stops instead of warning
-----------------------------------------

If a kernel was traced for more NeuronCores than the graph is compiled for, there is nowhere to run the extra pieces of work. Only the piece that computes the first NeuronCore's part of the output would be kept, and the rest of the output tensor would keep whatever values it held on input.

In earlier releases this happened silently: compilation reported success, produced a model, and the model ran — but the kernel's output was only partially filled in, and nothing in the compiler logs indicated a problem. Because a partially correct result is worse than a clear failure, the compiler now stops with ``NCC_ENKI003`` rather than continuing.

The compiler does not automatically run a kernel traced for more NeuronCores on fewer NeuronCores; the fix is to make the kernel's NeuronCore count and the compiled LNC agree, using one of the options below.

Common causes
-------------

- Running a model that contains an NKI kernel traced for two or more NeuronCores through Neuron's eager execution path. Eager execution always compiles for LNC=1, so any kernel traced for more than one NeuronCore triggers this error.
- Tracing a kernel at one LNC setting (for example, LNC=2) and then compiling the surrounding graph at a smaller setting (for example, LNC=1).
- Reusing a cached or previously generated kernel artifact that was traced for more NeuronCores than the current compilation is targeting.

Resolution
----------

Bring the kernel's NeuronCore count and the compiled LNC into agreement. The error message reports both values: ``<num_funcs>`` is the number of NeuronCores the kernel was traced for, and ``<lnc_count>`` is the LNC the graph is being compiled for. Choose one of the following:

1. **Re-trace the kernel for the LNC the graph is compiled for.** Set the LNC used at trace time to match the compile target — for example, ``export NEURON_LOGICAL_NC_CONFIG=<lnc_count>`` (or pass ``--lnc <lnc_count>`` when calling ``neuronx-cc`` directly) — then re-trace the kernel so it is built for ``<lnc_count>`` NeuronCores. If you hit this error on the eager execution path, the compile target is always LNC=1, so trace the kernel for a single NeuronCore.

2. **Compile the graph for the LNC the kernel was traced for.** Set ``NEURON_LOGICAL_NC_CONFIG=<num_funcs>`` (or ``--lnc <num_funcs>``) so the graph is compiled for the same number of NeuronCores the kernel already carries.

.. code-block:: bash

    # Option 2: compile the graph for the same number of NeuronCores
    # the kernel was traced for (for example, 2)
    export NEURON_LOGICAL_NC_CONFIG=2

Note that not every instance type supports every LNC value — some Neuron architectures support only LNC=1 (see ``NCC_EARG001``) — and the eager execution path always uses LNC=1. When compiling for a larger LNC is not possible, use option 1 and trace the kernel for the supported LNC instead.

For more information on Logical Neuron Cores: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/device-memory.html#logical-neuron-cores
