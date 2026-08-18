.. _error-code-evrf065:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF065.

NCC_EVRF065
===========

**Error message**: NKI kernel '<name>' was traced with an unsupported NKI version.

This error occurs when the compiler detects a NKI (Neuron Kernel Interface) kernel that was traced with an older, incompatible version of NKI. These legacy kernels are missing required metadata that the current compiler expects.

Why this happens
-----------------

NKI kernels are traced and compiled into intermediate artifacts by the NKI toolchain. When NKI traces a kernel, it records metadata alongside the kernel binary. Newer versions of the compiler require metadata fields that were not present in older NKI versions. If the compiler encounters a kernel artifact that lacks these fields, it cannot safely compile the kernel and raises NCC_EVRF065.

Common causes
--------------

- The NKI kernel was traced with an earlier version of the NKI toolchain and is being compiled with a newer compiler release.
- A cached or serialized model graph contains stale NKI kernel references from an older environment.
- The ``neuronxcc`` package was upgraded but the NKI kernels were not retraced.

Resolution
-----------

1. **Upgrade the installed NKI version** to the latest release compatible with your compiler version.

   .. code-block:: bash

      pip install --upgrade nki neuronx-cc==2.* --extra-index-url=https://pip.repos.neuron.amazonaws.com

   You can confirm the installed version afterwards with:

   .. code-block:: bash

      pip show nki

2. **Retrace your model** to regenerate the NKI kernel artifacts with the updated toolchain. For example, re-run your model's compilation or tracing step so that all NKI kernels are recompiled with the current NKI version.

3. **Clear any cached artifacts** from previous runs to ensure stale kernel binaries are not reused.
