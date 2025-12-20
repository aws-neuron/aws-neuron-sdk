.. _nki_block_dimension_migration_guide:

NKI Block Dimension Migration Guide
===================================

The SBUF/PSUM tensors in NKI used to allow block dimensions in front of the partition dimension. The block dimension support has been removed due the following reasons.

* Removing block dimensions does not hurt the expressivity of NKI.
* Block dimension is a pure software concept and does not have direct hardware mapping.
* The block dimension is unintuitive and causes confusion.
* Using block dimension has no inherit performance benefit, particularly using block dimension has no relationship with memory throughput whatsoever.
* Multi-buffering is implicit with block dimension. Removing block dimension will make multi-buffering more natural.

This document will first explain the semantics of block dimensions in detail, then it will provide information on how to migrate existing code that uses block dimensions while maintain the functional correctness and performance.

What are block dimensions?
--------------------------

Consider the following NKI tensor.

.. code-block:: python
  :linenos:

  a = nl.ndarray((4, 8, nl.par_dim(128), 2, 512), buffer=nl.sbuf)

  # - (4, 8): (B) block dimensions
  # - 128: (P) partition dimension
  # - (2, 512): (F) free dimension


A NKI tensor has three types of dimensions: `(B, P, F)` . The partition dimension maps to the partition dimension of the physical memory, and the free dimensions describe how data is organized in each SBUF/PSUM partition. The block dimensions described how many physical `(P, F)` tiles the tensor has.

The block dimension of tensors is a **logical** dimension and is a pure software concept. The compiler analyzes the memory dependency and allocates physical address to each tiles. **This means that the physical tiles may not be alive in the memory simultaneously**, and in most of the cases they don not. Consider the following code snippet that access the tensor `a`.

.. code-block:: python
  :linenos:

  @nki.jit
  def exp_func(inp):
    output = nl.ndarray((4, 8, 128, 2, 512), dtype=float32, 
      buffer=nl.shared_hbm)
    a = nl.ndarray((4, 8, nl.par_dim(128), 2, 512), dtype=float32, buffer=nl.sbuf)
    for i in range(4):
      for j in range(8):
        a[i, j] = nl.load(inp[i, j])
        a[i, j] = nl.exp(a[i, j])
        nl.store(output[i, j], value=result)


At the very minimum, only 1 physical tile of `a` needs to be alive. Then the execution is completely serialized. Essentially, all physical tiles would have the exact same memory address.

.. code-block::
  :linenos:

  Physical Address Map

  output[0, 0] --> Partition 0 - 128, Free 0 - 2048B
  output[0, 1] --> Partition 0 - 128, Free 0 - 2048B
  ...


Instead, compiler could choose to allocate 2 physical tiles to `a`, then the dma copy from HBM to SBUF can overlap with the exponential operation. In other word, **the block dimension allows compiler to perform space-time tradeoff at liberty.**

.. code-block::
  :linenos:

  Physical Address Map

  output[0, 0] --> Partition 0 - 128, Free 0    - 2048B
  output[0, 1] --> Partition 0 - 128, Free 2048 - 4096B
  output[0, 2] --> Partition 0 - 128, Free 0    - 2048B
  output[0, 3] --> Partition 0 - 128, Free 2048 - 4096B
  ...


When performing the migration, it is important to understand the dependency relationship between blocks and choose the correct migration method accordingly.

Migration for SBUF tensors
--------------------------

If blocks need to be alive at the same time, move the block dimension into free dimension
**********************************************************************************************

.. code-block:: python
  :linenos:

  a = nl.ndarray((8, par_dim(128), 512), buffer=nl.sbuf, dtype=bfloat16)

  # ----> Migrate to
  a = nl.ndarray((128, 8, 512), buffer=nl.sbuf, dtype=bfloat16)

As an example, all 8 blocks of ``add_buf`` needs to be alive at the same time when the first for loop finishes. Therefore, the block dimension need to be fold into the free dimension.

.. code-block:: python
    :linenos:

    @nki.jit
    def sb_blocks(inp):
        res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
        add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
        for i in range(8):
            add_buf[i] = nl.load(inp[i])
        for i in range(8):
            nl.store(res[i], add_buf[i])
        return res

    # should migrate to
    @nki.jit
    def sb_blocks_migrated(inp):
        res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
        add_buf = nl.ndarray(shape=(128, 8, 512), dtype=inp.dtype, buffer=nl.sbuf)
        for i in range(8):
            add_buf[0:128, i, 0:512] = nl.load(inp[i])
        for i in range(8):
            nl.store(res[i], add_buf[0:128, i, 0:512])
        return res

If blocks does not need to be alive at the same time, remove the block dimension and hoist it down 
**************************************************************************************************

.. code-block:: python
  :linenos:

  a = nl.ndarray((8, par_dim(128), 256))
  for i in nl.affine_range(8):
    <do something with a[i]>
    
  # should be transformed to ....
  for i in nl.affine_range(8):
    a = nl.ndarray((128, 256))
    <do something with a>

As an example, all 8 blocks of ``add_buf`` does not need to be alive at the same time. We can remove the block dimension and hoist down the tensor inside the loop.

.. code-block:: python
    :linenos:

    @nki.jit
    def sb_blocks(inp):
        res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
        add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
        for i in range(8):
            add_buf[i] = nl.load(inp[i])
            nl.store(res[i], add_buf[i])
        return res

    # should migrate to
    @nki.jit
    def sb_blocks_migrated(inp):
        res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
        for i in range(8):
            add_buf = nl.ndarray(shape=(128, 512), dtype=inp.dtype, buffer=nl.sbuf)
            add_buf[0:128, 0:512] = nl.load(inp[i])
            nl.store(res[i], add_buf[0:128, 0:512])
        return res

.. warning::
    To preserve performance, it is important to hoist down the tensor inside the loop.

It is important to note that the dependency relationship betweens loop iterations is different in ``sb_blocks_migrated`` and the following ``sb_blocks_migrated_incorrect``.

.. code-block:: python
    :linenos:

    @nki.jit
    def sb_blocks_migrated_incorrect(inp):
        res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
        add_buf = nl.ndarray(shape=(128, 512), dtype=inp.dtype, buffer=nl.sbuf)
        for i in range(8):
            add_buf[0:128, 0:512] = nl.load(inp[i])
            nl.store(res[i], add_buf[0:128, 0:512])
        return res

In ``sb_blocks_migrated``, compiler could unroll the loop and materialize multiple copies of the tensor ``add_buf``. However, in the ``sb_blocks_migrated_incorrect``, the execution will be serialized because the loop carries dependency on ``add_buf``.

Migration for PSUM tensors
--------------------------

.. note:: 
    To be filled, the backend support for removing blocks in PSUM tensor is still in progress.


Migration of direct allocation & multi-buffering
------------------------------------------------

When we have block dimensions, we allocate interleaved address for blocks to achieve multi-buffering.

.. code-block:: python
  :linenos:
  
  def interleave_alloc_func(idx, pdim_size, fdim_size):
    """
    This function assumes 1d block dimension, and will allocate unique
    address by modulo of 2.

    For a tensor of 4 blocks, block 0 and 2 will have the same address, while
    block 1 and 3 will have the same address that is different to that of 0 and 2.
    """
    # unpack the tuple
    idx, = idx

    # hard-code to partition 0, since each tile takes up 128 partitions
    start_partition = 0

    return (start_partition, (idx % 2) * fdim_size)
  
  @nki.jit
  def copy_func(inp):
    output = nl.ndarray((4, 128, 512), dtype=float32, buffer=nl.shared_hbm)
    a = nl.ndarray((4, nl.par_dim(128), 512), dtype=float32, buffer=ncc.sbuf.alloc(interleave_alloc_func))
    for i in range(4):
        a[i] = nl.load(inp[i])
        nl.store(output[i], value=a[i])

After removing the block dimension, we could write the following to implement the same multi-buffering, which is actually more natural and closer to that on CPU.

.. code-block:: python
  :linenos:
  
  def interleave_alloc_func(idx, pdim_size, fdim_size):
    """
    This function assumes 1d block dimension, and will allocate unique
    address by modulo of 2.

    For a tensor of 4 blocks, block 0 and 2 will have the same address, while
    block 1 and 3 will have the same address that is different to that of 0 and 2.
    """
    # unpack the tuple
    assert idx == () # We don't have any block dimension

    # hard-code to partition 0, since each tile takes up 128 partitions
    start_partition = 0

    return (start_partition, (idx % 2) * fdim_size)
  
  @nki.compiler.skip_middle_end_transformations
  @nki.jit
  def exp_func(inp):
    output = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.shared_hbm)
    a = nl.ndarray((128, 2, 512), dtype=nl.float32, buffer=ncc.sbuf.alloc(interleave_alloc_func))
    for i in range(4):
      a[0:128, i % 2, 0:512] = nl.load(inp[i])
      nl.store(output[i], value=a[0:128, i % 2, 0:512])
