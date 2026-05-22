from dataclasses import dataclass
from enum import Enum
from typing import *

def jit(fn=None, **kwargs):
    r"""Just-in-time compile a top-level NKI function to run on NeuronDevices.

    The returned callable detects the current framework and compiles the function as a
    custom operator. It detects the current framework by inspecting its arguments:

    - ``torch.Tensor``: uses PyTorch integration.
    - ``jax.Array``: uses JAX integration.
    - ``np.ndarray``: compiles and executes standalone kernel, without a framework.

    You might need to explicitly set the target platform using the
    ``NEURON_PLATFORM_TARGET_OVERRIDE`` environment variable. Supported values:

    - ``trn1|inf2|gen2``
    - ``trn2|gen3``
    - ``trn3|gen4``

    The LNC (Logical NeuronCore) degree can be set at the callsite using bracket syntax:
    ``kernel[lnc](args)``. The default is LNC=1. The LNC value must match the
    ``NEURON_LOGICAL_NC_CONFIG`` environment variable set for the Neuron Runtime.
    Mismatching the two will cause a runtime error. For example, if
    ``NEURON_LOGICAL_NC_CONFIG=1``, the kernel must be launched with ``kernel[1](...)``
    or ``kernel(...)``.

    Returns a :class:`Kernel` instance wrapping the decorated function."""
    ...

def simulate(kernel):
    r"""Create a CPU-simulated version of an NKI kernel.

    .. warning::

      **This API is experimental and may change in future releases**. It has not been tested or confirmed to work on all hardware platforms and operating systems.

      Currently, Neuron confirms support for ``nki.simulate`` on these 2 operating systems:

      * Ubuntu 22.04
      * Amazon Linux 2023

    See :ref:`nki-simulator` for full documentation including target platform
    selection, precise floating-point mode, debugging, and known limitations.

    Example::

        @nki.jit
        def my_kernel(a, b): ...

        # Explicit simulation
        result = nki.simulate(my_kernel)(a_np, b_np)

        # With LNC2
        result = nki.simulate(my_kernel[2])(a_np, b_np)

    Args:
      kernel: NKI kernel function, typically decorated with ``@nki.jit``.
        If a plain function is passed, it is automatically wrapped.

    Returns:
      A callable that, when invoked with NumPy arrays or torch Tensors,
      executes the kernel on CPU and returns results in the same format
      (NumPy arrays or torch Tensors respectively)."""
    ...
