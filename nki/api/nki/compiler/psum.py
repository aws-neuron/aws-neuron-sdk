import numpy as np
import ml_dtypes

def alloc(func):
  r"""
    Allocate PSUM memory space for each logical block in a tensor using a customized allocation method.

    This is one of the NKI direction allocation APIs.
    We recommend reading :doc:`NKI Direct Allocation Developer Guide <../../nki_direct_allocation_guide>` before
    using these APIs.

    In NKI, a PSUM tensor (declared using :ref:`NKI tensor creation APIs <nl_creation>`)
    can have three kinds of dimensions, in order: logical block(B), partition(P),
    and free(F). The partition and free dimensions directly map to the PSUM dimensions.
    Both B and F can be multi-dimensional, while P must be one-dimensional per Neuron ISA constraints.
    The block dimension describes how many (P, F) logical tiles this tensor has, but does not reflect the number
    of physical tiles being allocated.

    ``ncc.psum.alloc`` should be assigned to the ``buffer`` field of a NKI tensor declaration API. For example,

    .. code-block::

      nki_tensor = nl.ndarray((2, 4, nl.par_dim(128), 512), dtype=nl.float32, buffer=ncc.psum.alloc(...))

    ``ncc.psum.alloc`` allows programmers to specify the physical location of each logical tile in
    the tensor. The API accepts a single input ``func`` parameter, which is a callable
    object that takes in:

    1. a tuple of integers ``idx`` representing a logical block index,
    2. an integer ``pdim_size`` for the number of partitions the logical tile has, and
    3. an integer ``fdim_size`` for the number of bytes the logical tile has per partition.

    The number of integers in ``idx`` must match the number of B dimensions the PSUM tensor has. For example, for the
    above ``nki_tensor``, we expect the ``idx`` tuple to have two integers for a 2D block index.

    ``pdim_size`` should match the partition dimension size of the NKI tensor exactly. ``fdim_size`` should be the
    total size of F dimension shapes of each logical tile in the tensor, multiplied by the data type size in bytes.
    For the above ``nki_tensor``, ``pdim_size`` should be 128, and ``fdim_size`` should be
    ``512*sizeof(nl.float32) = 2048`` bytes.

    .. note::

      In current release, ``fdim_size`` cannot exceed 2KiB, which is the size
      of a single PSUM bank per partition. Therefore, a physical PSUM tile cannot span multiple PSUM banks.
      Check out :ref:`trainium_inferentia2_arch` for more information on PSUM banks.

    The ``func`` returns a tuple of three integers ``(bank_id, start_partition, byte_addr)`` indicating
    the physical tile location for the input logical block index.

    ``bank_id`` indicates the PSUM bank ID of the physical tile.
    ``start_partition`` indicates the lowest partition the physical tile allocation starts from.
    The ``byte_addr`` indicates the byte offset into each PSUM bank per partition the physical tile starts from.

    .. note::

      In current release, ``start_partition`` and ``byte_addr`` must both be 0.

    .. note::

      In current release, programmers cannot mix NKI tensor declarations using automatic allocation
      (``ncc.psum.auto_alloc()`` or the SBUF variant) and
      direction allocation APIs (``ncc.psum.alloc()``, ``ncc.psum.mod_alloc()`` or the SBUF variants) in the same kernel.


    :param func: a callable object to specify how to place the logical block in PSUM memory.
    """
  ...

def mod_alloc(*, base_bank, base_addr=0, base_partition=0, num_bank_tiles=(), num_par_tiles=(), num_free_tiles=()):
  r"""
    Allocate PSUM memory space for each logical block in a tensor through modulo allocation.

    This is one of the NKI direction allocation APIs.
    We recommend reading :doc:`NKI Direct Allocation Developer Guide <../../nki_direct_allocation_guide>` before
    using these APIs.

    This API is equivalent to calling :doc:`nki.compiler.psum.alloc() <nki.compiler.psum.alloc>`
    with a callable ``psum_modulo_alloc_func`` as defined below.

    .. nki_example:: ../../../nki/test/test_psum_modulo_alloc.py
      :language: python
      :linenos:
      :marker: NKI_EXAMPLE_0

    Here's an example usage of this API:

    .. code-block:: python

      psum_tensor = nl.ndarray((4, nl.par_dim(128), 512), dtype=nl.float32,
                               buffer=ncc.psum.mod_alloc(base_bank=0,
                                                          base_addr=0,
                                                          num_bank_tiles=(2,)))

      for i_block in nl.affine_range(4):
        psum[i_block, :, :] = nisa.nc_matmul(...)
        ...                 = nl.exp(psum[i_block, :, :])

    This produces the following allocation:

    .. list-table:: Modulo Allocation Example
      :header-rows: 1

      * - Logical Tile Index
        - Physical Tile ``bank_id``
        - Physical Tile ``start_partition``
        - Physical Tile ``byte_addr``
      * - (0, )
        - 0
        - 0
        - 0

      * - (1, )
        - 1
        - 0
        - 0

      * - (2, )
        - 0
        - 0
        - 0

      * - (3, )
        - 1
        - 0
        - 0

    With above scheme, we are able to implement double buffering in ``nki_tensor``, such that ``nisa.nc_matmul``
    in one iteration can write to one physical tile while ``nl.exp`` of the previous iteration can
    read from the other physical tile simultaneously.

    .. note::

      In current release, programmers cannot mix NKI tensor declarations using automatic allocation
      (``ncc.psum.auto_alloc()`` or the SBUF variant) and
      direction allocation APIs (``ncc.psum.alloc()``, ``ncc.psum.mod_alloc()`` or the SBUF variants).


    :param base_addr: the base address in bytes along the free(F) dimension of the PSUM bank. Must be 0 in the current version.
    :param base_bank: the base bank ID that the physical tiles start from.
    :param num_bank_tiles: the number of PSUM banks allocated for the tensor.
    :param base_partition: the partition ID the physical tiles start from. Must be 0 in the current version.
    :param num_par_tiles: the number of physical tiles along the partition dimension allocated for the tensor.
      The length of the tuple must be empty or equal to the length of block dimension for the tensor.
      Currently must be an empty tuple or (1, 1, ...).
    :param num_free_tiles: the number of physical tiles on the free dimension per PSUM bank allocated for the tensor.
      The length of the tuple must be empty or equal to the length of block dimension for the tensor.
      Currently must be an empty tuple or (1, 1, ...).
    """
  ...

def auto_alloc():
  r"""
    Returns a maker to indicate the tensor should be automatically allocated by compiler.
    All PSUM tensors in a kernel must either all be marked as ``auto_alloc()``, or all be allocated
    with ``alloc`` or ``mod_alloc``.

    Initialize a tensor with ``buffer=nl.psum`` is equivalent to ``buffer=ncc.psum.auto_alloc()``.
    """
  ...

