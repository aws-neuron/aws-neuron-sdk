# AUTO GENERATED FILE - DO NOT MANUALLY MODIFY.

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import *

def jit(fn=None, *, compile_config=None, specialization_config=None, **kwargs):
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

    Configure the LNC (Logical NeuronCore) degree by setting the
    ``NEURON_LOGICAL_NC_CONFIG`` environment variable to 1 or 2 before invoking
    the kernel. If the LNC degree is not explicitly set, the default is LNC=1.

    Per-kernel config is passed through two typed objects:

    - ``compile_config`` -- a :class:`nki.compiler.CompileConfig` of options that
      impact downstream compiler behavior (``address_rotation``,
      ``non_ssa_legalization``, ``dead_code_elim``, ``barrier_check``).
    - ``specialization_config`` -- a :class:`nki.compiler.SpecializationConfig` of
      options that tune the specialization phase (``allow_nested_config_override``).

    e.g.::

        from nki.compiler import CompileConfig, SpecializationConfig

        @nki.jit(compile_config=CompileConfig(non_ssa_legalization=True))
        def my_kernel(...): ...

    ``SpecializationConfig.allow_nested_config_override`` -- when this kernel is used as the
    top-level compiled kernel:

    * default (``False``): a nested sub-kernel whose config disagrees raises an error.
    * ``True``: warn instead, and inline the sub-kernel under this kernel's config.

    Useful for composing @nki.jit kernels whose config you cannot edit.

    Returns a :class:`Kernel` instance wrapping the decorated function."""
    ...

def simulate(kernel):
    r"""Create a CPU-simulated version of an NKI kernel.

    .. warning::

       This API is experimental and may change in future releases.

    See :ref:`nki-simulator` for full documentation including target platform
    selection, precise floating-point mode, debugging, and known limitations.

    Example::

        import os

        @nki.jit
        def my_kernel(a, b): ...

        # Explicit simulation
        result = nki.simulate(my_kernel)(a_np, b_np)

        # With LNC2
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
        result = nki.simulate(my_kernel)(a_np, b_np)

    Args:
      kernel: NKI kernel function, typically decorated with ``@nki.jit``.
        If a plain function is passed, it is automatically wrapped.

    Returns:
      A callable that, when invoked with NumPy arrays or torch Tensors,
      executes the kernel on CPU and returns results in the same format
      (NumPy arrays or torch Tensors respectively)."""
    ...
