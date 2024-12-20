import numpy as np
import ml_dtypes

def alloc(func):
  r"""
    Allocate SBUF memory space for each logical block in a tensor using a customized allocation method.

    This is one of the NKI direction allocation APIs.
    We recommend reading :doc:`NKI Direct Allocation Developer Guide <../../nki_direct_allocation_guide>` before
    using these APIs.

    In NKI, a SBUF tensor (declared using :ref:`NKI tensor creation APIs <nl_creation>`)
    can have three kinds of dimensions, in order: logical block(B), partition(P),
    and free(F). The partition and free dimensions directly map to the SBUF dimensions.
    Both B and F can be multi-dimensional, while P must be one-dimensional per Neuron ISA constraints.
    The block dimension describes how many (P, F) logical tiles this tensor has, but does not reflect the number
    of physical tiles being allocated.

    ``ncc.sbuf.alloc`` should be assigned to the ``buffer`` field of a NKI tensor declaration API. For example,

    .. code-block::

      nki_tensor = nl.ndarray((4, 8, nl.par_dim(128), 4, 32), dtype=nl.bfloat16, buffer=ncc.sbuf.alloc(...))

    ``ncc.sbuf.alloc`` allows programmers to specify the physical location of each logical tile in
    the tensor. The API accepts a single input ``func`` parameter, which is a callable
    object that takes in:

    1. a tuple of integers ``idx`` representing a logical block index,
    2. an integer ``pdim_size`` for the number of partitions the logical tile has, and
    3. an integer ``fdim_size`` for the number of bytes the logical tile has per partition.

    The number of integers in ``idx`` must match the number of B dimensions the SBUF tensor has. For example, for the
    above ``nki_tensor``, we expect the ``idx`` tuple to have two integers for a 2D block index.

    ``pdim_size`` should match the partition dimension size of the NKI tensor exactly. ``fdim_size`` should be the
    total size of F dimension shapes of each logical tile in the tensor, multiplied by the data type size in bytes.
    For the above ``sbuf_tensor``, ``pdim_size`` should be 128, and ``fdim_size`` should be
    ``4*32*sizeof(nl.bfloat16) = 256`` bytes.

    The ``func`` callable must return a tuple of two integers ``(start_partition, byte_addr)`` indicating
    the physical tile location for the input logical block index. ``start_partition`` indicates
    the lowest partition the physical tile allocation
    starts from and must follow the these ISA rules:

    - If ``64 < pdim_size <= 128``, ``start_partition`` must be 0
    - If ``32 < pdim_size <= 64``,  ``start_partition`` must be 0 or 64
    - If ``0  < pdim_size <= 32``,  ``start_partition`` must be one of 0/32/64/96

    The ``byte_addr`` indicates the byte offset into each partition the physical tile starts from.
    On NeuronCore-v2, a valid ``byte_addr`` can be any integer values from 0 (inclusive) to
    192KiB-16KiB=(192-16)*1024 (exclusive). 192KiB is the physical size of a SBUF partition
    (defined in :doc:`architecture guide <../../arch/trainium_inferentia2_arch>`) and 16KiB is allocated for compiler internal usage.
    In addition, the ``base_addr`` must be aligned to ``nki.language.constants.sbuf_min_align``.


    .. note::

      In current release, programmers cannot mix NKI tensor declarations using automatic allocation
      (``ncc.sbuf.auto_alloc()`` or the PSUM variant) and
      direction allocation APIs (``ncc.sbuf.alloc()``, ``ncc.sbuf.mod_alloc()`` or the PSUM variants) in the same kernel.


    :param func: a callable object to specify how to place the logical block in SBUF memory.
    """
  ...

def mod_alloc(*, base_addr, base_partition=0, num_par_tiles=(), num_free_tiles=()):
  r"""
    Allocate SBUF memory space for each logical tile in a tensor through modulo allocation.

    This is one of the NKI direction allocation APIs.
    We recommend reading :doc:`NKI Direct Allocation Developer Guide <../../nki_direct_allocation_guide>` before
    using these APIs.

    This API is equivalent to calling :doc:`nisa.compiler.alloc() <nki.compiler.sbuf.alloc>`
    with a callable ``psum_modulo_alloc_func`` as defined below.

    .. nki_example:: ../../../nki/test/test_sbuf_modulo_alloc.py
      :language: python
      :linenos:
      :marker: NKI_EXAMPLE_0

    Here's an example usage of this API:

    .. code-block:: python

      nki_tensor = nl.ndarray((4, par_dim(128), 512), dtype=nl.bfloat16,
                              buffer=nki.compiler.sbuf.mod_alloc(base_addr=0, num_free_tiles=(2, )))

      for i_block in nl.affine_range(4):
        nki_tensor[i_block, :, :] = nl.load(...)
        ...                       = nl.exp(nki_tensor[i_block, :, :])

    This produces the following allocation:

    .. list-table:: Modulo Allocation Example
      :header-rows: 1

      * - Logical Tile Index
        - Physical Tile ``start_partition``
        - Physical Tile ``byte_addr``
      * - (0, )
        - 0
        - 0 + (0 % 2) * 512 * sizeof(nl.bfloat16) = 0

      * - (1, )
        - 0
        - 0 + (1 % 2) * 512 * sizeof(nl.bfloat16) = 1024

      * - (2, )
        - 0
        - 0 + (2 % 2) * 512 * sizeof(nl.bfloat16) = 0

      * - (3, )
        - 0
        - 0 + (3 % 2) * 512 * sizeof(nl.bfloat16) = 1024

    With above scheme, we are able to implement double buffering in ``nki_tensor``, such that ``nl.load`` in one iteration
    can write to one physical tile while ``nl.exp`` of the previous iteration can read from the other physical tile
    simultaneously.


    .. note::

      In current release, programmers cannot mix NKI tensor declarations using automatic allocation
      (``ncc.sbuf.auto_alloc()`` or the PSUM variant) and
      direction allocation APIs (``ncc.sbuf.alloc()``, ``ncc.sbuf.mod_alloc()`` or the PSUM variants).

    :param base_addr: the base address in the free(F) dimension of the SBUF in bytes.
    :param base_partition: the partition where the physical tile starts from. Must be 0 in the current version.
    :param num_par_tiles: the number of physical tiles on the partition dimension of SBUF allocated for the tensor.
      The length of the tuple must be empty or equal to the length of block dimension for the tensor.
    :param num_free_tiles: the number of physical tiles on the free dimension of SBUF allocated for the tensor.
      The length of the tuple must be empty or equal to the length of block dimension for the tensor.
    """
  ...

def auto_alloc():
  r"""
    Returns a maker to indicate the tensor should be automatically allocated by compiler.
    All SBUF tensors in a kernel must either all be marked as ``auto_alloc()``, or all be allocated
    with ``alloc`` or ``mod_alloc``.

    Initialize a tensor with ``buffer=nl.sbuf`` is equivalent to ``buffer=ncc.sbuf.auto_alloc()``.
    """
  ...

