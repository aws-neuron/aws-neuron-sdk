import numpy as np
import ml_dtypes

psum = ...
r"""PSUM - Only visible to each individual kernel instance in the SPMD grid, alias of ``nki.compiler.psum.auto_alloc()``"""

sbuf = ...
r"""State Buffer - Only visible to each individual kernel instance in the SPMD grid, alias of ``nki.compiler.sbuf.auto_alloc()``"""

def enable_stack_allocator(func=None, log_level=50):
  r"""
  Use stack allocator to allocate the psum and sbuf tensors in the kernel.

  Must use together with skip_middle_end_transformations.

  .. code-block:: python

    from neuronxcc import nki

    @nki.compiler.enable_stack_allocator
    @nki.compiler.skip_middle_end_transformations
    @nki.jit
    def kernel(...):
      ...

  """
  ...

def force_auto_alloc(func=None):
  r""" Force automatic allocation to be turned on in the kernel.

  This will ignore any direct allocation inside the kernel
  """
  ...

def multi_buffer(factor=2):
  r"""Create a MultiBufferDirective to enable multi-buffered allocation.

  Args:
      factor: The multi-buffer factor determining how many buffers to use.
  """
  ...

def no_reorder():
  r"""Create an OperationOrderGuard to prevent operation reordering."""
  ...

def skip_middle_end_transformations(func=None):
  r""" Skip all middle end transformations on the kernel

  """
  ...

