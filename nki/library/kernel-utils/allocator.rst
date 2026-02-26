.. meta::
    :description: API reference for the SbufManager (Allocator) utility in the NKI Library.
    :date-modified: 02/13/2026

.. currentmodule:: nkilib.core.utils.allocator

SbufManager (Allocator) API Reference
=====================================

This topic provides the API reference for the ``SbufManager`` utility. It provides stack-based SBUF memory allocation with scope management and multi-buffering support.

When to Use
-----------

Use ``SbufManager`` when you need:

* **Deterministic memory layout**: Manual control over SBUF addresses for predictable memory placement
* **Scope-based allocation**: Automatic cleanup of temporary buffers when a computation phase ends
* **Multi-buffering in loops**: Ping-pong buffers for overlapping compute and memory operations
* **Memory debugging**: Detailed logging of allocation patterns and usage statistics

``SbufManager`` is particularly useful in complex kernels with multiple computation phases where different buffers are needed at different times.

API Reference
-------------

**Source code**: https://github.com/aws-neuron/nki-library

SbufManager
^^^^^^^^^^^

.. py:class:: SbufManager(sb_lower_bound, sb_upper_bound, logger=None, use_auto_alloc=False, default_stack_alloc=True)

   Stack-based SBUF memory manager with scope support.

   :param sb_lower_bound: Lower bound of available SBUF memory region.
   :type sb_lower_bound: int
   :param sb_upper_bound: Upper bound of available SBUF memory region.
   :type sb_upper_bound: int
   :param logger: Optional logger instance for allocation tracking.
   :type logger: Logger, optional
   :param use_auto_alloc: If True, delegates address assignment to compiler. Default False.
   :type use_auto_alloc: bool
   :param default_stack_alloc: If True, ``alloc()`` uses stack; if False, uses heap. Default True.
   :type default_stack_alloc: bool

   .. py:method:: open_scope(interleave_degree=1, name="")

      Opens a new allocation scope. Allocations within this scope are freed when the scope closes.

      :param interleave_degree: Number of buffer sections for multi-buffering. Default 1.
      :type interleave_degree: int
      :param name: Optional scope name for debugging.
      :type name: str
      :rtype: None

   .. py:method:: close_scope()

      Closes the current scope and frees all stack allocations made within it.

      :rtype: None

   .. py:method:: increment_section()

      Advances to the next buffer section within a multi-buffer scope. When all sections are used, wraps back to the first section.

      :rtype: None

   .. py:method:: alloc_stack(shape, dtype, buffer=nl.sbuf, name=None, base_partition=0, align=None)

      Allocates a tensor on the stack (freed when scope closes).

      :param shape: Shape of the tensor.
      :type shape: tuple[int, ...]
      :param dtype: Data type (e.g., ``nl.bfloat16``, ``nl.float32``).
      :type dtype: dtype
      :param buffer: Buffer type. Only ``nl.sbuf`` supported.
      :type buffer: buffer
      :param name: Optional tensor name (must be unique).
      :type name: str, optional
      :param base_partition: Base partition for allocation. Default 0.
      :type base_partition: int
      :param align: Alignment requirement in bytes.
      :type align: int, optional
      :return: Allocated SBUF tensor.
      :rtype: nl.ndarray

   .. py:method:: alloc_heap(shape, dtype, buffer=nl.sbuf, name=None, base_partition=0, align=None)

      Allocates a tensor on the heap (must be manually freed with ``pop_heap()``).

      Parameters are identical to ``alloc_stack()``.

      :rtype: nl.ndarray

   .. py:method:: alloc(shape, dtype, buffer=nl.sbuf, name=None, base_partition=0, align=None)

      Allocates a tensor on the stack or heap, depending on the ``default_stack_alloc`` setting.

      Parameters are identical to ``alloc_stack()``.

      :rtype: nl.ndarray

   .. py:method:: pop_heap()

      Frees the most recently allocated heap tensor.

      :rtype: None

   .. py:method:: get_total_space()

      Returns the total number of bytes in the managed region.

      :rtype: int

   .. py:method:: get_free_space()

      Returns the number of free bytes between stack and heap.

      :rtype: int

   .. py:method:: get_used_space()

      Returns the number of bytes currently used by stack and heap allocations.

      :rtype: int

   .. py:method:: get_stack_curr_addr()

      Returns the current stack address. Not supported in auto-allocation mode.

      :rtype: int

   .. py:method:: get_heap_curr_addr()

      Returns the current heap address. Not supported in auto-allocation mode.

      :rtype: int

   .. py:method:: align_stack_curr_addr(align=32)

      Aligns the current stack address to the given alignment. Not supported in auto-allocation mode.

      :param align: Alignment in bytes. Default 32.
      :type align: int
      :rtype: None

   .. py:method:: set_name_prefix(prefix)

      Sets a prefix string prepended to all subsequent allocation names.

      :param prefix: Prefix string.
      :type prefix: str
      :rtype: None

   .. py:method:: get_name_prefix()

      Returns the current name prefix.

      :rtype: str

   .. py:method:: flush_logs()

      Prints buffered allocation logs in tree format.

      :rtype: None

create_auto_alloc_manager
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: create_auto_alloc_manager(logger=None)

   Creates an SbufManager that delegates address assignment to the compiler.

   :param logger: Optional logger instance.
   :type logger: Logger, optional
   :return: Auto-allocation SbufManager instance.
   :rtype: SbufManager

Examples
--------

Without SbufManager (Manual Allocation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.language as nl

   @nki.jit
   def kernel_without_sbm(input_hbm, output_hbm):
       addr = 0
       
       # Heap-like allocation at end of SBUF
       heap_addr = nl.tile_size.total_available_sbuf_size - 512
       weights = nl.ndarray((128, 256), dtype=nl.bfloat16, buffer=nl.sbuf,
                            address=(0, heap_addr))
       print(f"weights.address = {weights.address}")  # (0, 261632)
       
       # Outer scope
       buf1 = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf,
                         address=(0, addr))
       print(f"buf1.address = {buf1.address}")  # (0, 0)
       addr += 512 * 2  # 1024
       
       # Inner scope
       inner_start = addr
       buf2 = nl.ndarray((128, 256), dtype=nl.bfloat16, buffer=nl.sbuf,
                         address=(0, addr))
       print(f"buf2.address = {buf2.address}")  # (0, 1024)
       addr += 256 * 2  # 1536
       buf3 = nl.ndarray((128, 256), dtype=nl.bfloat16, buffer=nl.sbuf,
                         address=(0, addr))
       print(f"buf3.address = {buf3.address}")  # (0, 1536)
       # End inner scope - must manually reset
       addr = inner_start  # 1024
       
       # Back in outer - reuse inner's memory
       buf4 = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf,
                         address=(0, addr))
       print(f"buf4.address = {buf4.address}")  # (0, 1024)

With SbufManager
^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.language as nl
   from nkilib.core.utils.allocator import SbufManager

   @nki.jit
   def kernel_with_sbm(input_hbm, output_hbm):
       sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size)
       
       weights = sbm.alloc_heap((128, 256), nl.bfloat16, name="weights")
       print(f"weights.address = {weights.address}")  # (0, 261632)
       
       sbm.open_scope(name="outer")
       buf1 = sbm.alloc_stack((128, 512), nl.bfloat16, name="buf1")
       print(f"buf1.address = {buf1.address}")  # (0, 0)
       
       sbm.open_scope(name="inner")
       buf2 = sbm.alloc_stack((128, 256), nl.bfloat16, name="buf2")
       print(f"buf2.address = {buf2.address}")  # (0, 1024)
       buf3 = sbm.alloc_stack((128, 256), nl.bfloat16, name="buf3")
       print(f"buf3.address = {buf3.address}")  # (0, 1536)
       sbm.close_scope()
       
       buf4 = sbm.alloc_stack((128, 512), nl.bfloat16, name="buf4")
       print(f"buf4.address = {buf4.address}")  # (0, 1024)
       sbm.close_scope()
       
       sbm.pop_heap()

Both produce identical memory layouts:

.. code-block:: text

   weights.address = (0, 261632)  # heap at top
   buf1.address = (0, 0)          # stack grows up
   buf2.address = (0, 1024)       # inner scope
   buf3.address = (0, 1536)       # inner scope
   buf4.address = (0, 1024)       # reuses inner's memory

Multi-Buffering Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.language as nl
   from nkilib.core.utils.allocator import SbufManager

   @nki.jit
   def kernel_multibuffer(input_hbm, output_hbm, N):
       sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size)
       
       # Double-buffering: 2 sections alternate
       sbm.open_scope(interleave_degree=2, name="double_buffer")
       
       for i in nl.affine_range(N):
           # Allocates to section 0, then 1, then 0, then 1...
           buf = sbm.alloc_stack((128, 512), nl.bfloat16)
           # Load to buf[current], compute on buf[previous]
           sbm.increment_section()
       
       sbm.close_scope()

Debug output for ``N=4``:

.. code-block:: text

   [SBM] Allocations:
       ▶ SCOPE 'double_buffer' [interleave=2] @ 0
       ├── (unnamed): 1024 B @ 0 (128, 512) bfloat16
       ├── ↳ section: 1/2 @ 1024
       ├── (unnamed): 1024 B @ 1024 (128, 512) bfloat16
       ├── ↻ section: 0/2 @ 0
       ├── (unnamed): 1024 B @ 0 (128, 512) bfloat16
       ├── ↳ section: 1/2 @ 1024
       └── (unnamed): 1024 B @ 1024 (128, 512) bfloat16
       ◀ END 'double_buffer' freed=2048 B

Note how allocations alternate between addresses 0 and 1024.

See Also
--------

* :doc:`TensorView </nki/library/kernel-utils/tensor-view>` - Zero-copy tensor view operations
