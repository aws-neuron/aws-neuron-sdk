.. _error-code-evrf059:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF059.

NCC_EVRF059
===========

**Error message**: Kernel file '<path>' referenced by AwsNeuronCustomNativeKernel instruction does not exist on the host.

This error occurs when the compiler cannot locate an intermediate NKI (Neuron Kernel Interface) artifact that is referenced in the HLO/StableHLO input graph.

How NKI references are stored
------------------------------

When a model uses custom NKI kernels, the framework (PyTorch or JAX) traces and compiles the kernel into intermediate NKI artifact files. References to these files are embedded into the HLO/StableHLO graph that is passed to ``neuronx-cc``. Each reference contains the absolute path to the artifact as it existed on the file system at trace time.

How the compiler resolves NKI artifacts
----------------------------------------

During verification, the compiler checks that each referenced NKI artifact can be found on the file system. It searches in two locations:

1. **The absolute path** stored in the HLO graph.
2. **A relative path** constructed from the last directory component and filename, resolved from the compiler's current working directory (the directory where ``neuronx-cc`` was launched).

If the file is not found at either location, the compiler raises NCC_EVRF059.

Example directory layout
-------------------------

Suppose the HLO graph references a kernel artifact at ``/home/user/project/71adab98b5b4ec7f/my_kernel_artifact``. The compiler will accept the file if it exists at either of these locations:

.. code-block:: text

    Option 1: Absolute path exists
    /home/user/project/71adab98b5b4ec7f/my_kernel_artifact   <-- file found here

    Option 2: Relative to compiler launch directory
    <compiler_launch_dir>/
    └── 71adab98b5b4ec7f/
        └── my_kernel_artifact   <-- file found here

For example, if you launch the compiler from ``/workspace/build/``:

.. code-block:: text

    /workspace/build/                    <-- neuronx-cc launched from here
    ├── model.hlo
    └── 71adab98b5b4ec7f/
        └── my_kernel_artifact           <-- resolved via relative path

Common causes
--------------

- The NKI artifact was generated in a temporary directory or cache that has since been cleaned up.
- The compilation is running on a different machine or container from where the model was traced.
- The NKI artifact was moved or renamed after the HLO graph was generated.

Resolution
-----------

Ensure the NKI artifact files exist at an accessible path before compilation:

1. **Clear the NKI file cache and rerun the model trace** to regenerate NKI artifacts at the expected absolute paths.
2. **Copy NKI artifacts into the compiler launch directory** using the ``<hash_dir>/<filename>`` structure shown above. This option should be used if calling ``neuronx-cc`` directly.
