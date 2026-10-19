.. meta::
    :description: API reference for the TiledRange utility in the NKI Library.
    :date-modified: 02/04/2026

.. currentmodule:: nkilib.core.utils.tiled_range

TiledRange API Reference
========================

This topic provides the API reference for the ``TiledRange`` utility. It divides a dimension into tiles and provides iterators with size, index, and offset information, automatically handling remainders.

When to Use
-----------

Use ``TiledRange`` when you need to:

* **Tile a dimension with remainders**: Process dimensions that don't divide evenly by tile size
* **Track tile metadata**: Access tile size, index, and absolute offsets during iteration
* **Nested tiling**: Subdivide tiles into smaller tiles while preserving absolute offsets

``TiledRange`` simplifies the common pattern of iterating over a dimension in fixed-size chunks while correctly handling the final partial tile.

API Reference
-------------

**Source code**: https://github.com/aws-neuron/nki-library

TiledRange
^^^^^^^^^^

.. py:function:: TiledRange(size, tile_size)

   Divides a dimension into tiles and returns a tuple of ``TiledRangeIterator`` objects.

   :param size: Total size to tile, or a ``TiledRangeIterator`` for nested tiling.
   :type size: int or TiledRangeIterator
   :param tile_size: Size of each tile.
   :type tile_size: int
   :return: Tuple of tile iterators.
   :rtype: tuple[TiledRangeIterator, ...]

TiledRangeIterator
^^^^^^^^^^^^^^^^^^

.. py:class:: TiledRangeIterator

   Represents a single tile in a tiled range.

   .. py:attribute:: size
      :type: int

      Size of this tile (may be smaller than tile_size for the last tile).

   .. py:attribute:: index
      :type: int

      Zero-based index of this tile in the range.

   .. py:attribute:: start_offset
      :type: int

      Absolute starting offset in the original dimension.

   .. py:attribute:: end_offset
      :type: int

      Absolute ending offset in the original dimension.

Examples
--------

Without TiledRange
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import math

   @nki.jit
   def kernel_without_tiled_range(data_hbm, H):
       TILE_SIZE = 128
       num_tiles = math.ceil(H / TILE_SIZE)
       
       for i in range(num_tiles):
           start = i * TILE_SIZE
           # Must manually compute remainder for last tile
           if i == num_tiles - 1:
               tile_size = H - start
           else:
               tile_size = TILE_SIZE
           end = start + tile_size
           
           # Load tile - must use computed values
           tile = nl.load(data_hbm[:, start:end])
           # ... process tile ...

With TiledRange
^^^^^^^^^^^^^^^

.. code-block:: python

   from nkilib.core.utils.tiled_range import TiledRange

   @nki.jit
   def kernel_with_tiled_range(data_hbm, H):
       TILE_SIZE = 128
       
       for tile in TiledRange(H, TILE_SIZE):
           # tile.size handles remainder automatically
           # tile.start_offset and tile.end_offset give absolute positions
           data = nl.load(data_hbm[:, tile.start_offset:tile.end_offset])
           # ... process tile of size tile.size ...

Example output for ``TiledRange(300, 128)``:

.. code-block:: text

   tile 0: size=128, start_offset=0, end_offset=128
   tile 1: size=128, start_offset=128, end_offset=256
   tile 2: size=44, start_offset=256, end_offset=300

Note how the last tile automatically has ``size=44`` to handle the remainder.

Nested Tiling
^^^^^^^^^^^^^

.. code-block:: python

   from nkilib.core.utils.tiled_range import TiledRange

   @nki.jit
   def kernel_nested_tiling(data_hbm, H):
       OUTER_TILE = 512
       INNER_TILE = 128
       
       for outer in TiledRange(H, OUTER_TILE):
           # Nested tiling: subdivide outer tile
           for inner in TiledRange(outer, INNER_TILE):
               # inner.start_offset is absolute (not relative to outer)
               # inner.size handles remainder within outer tile
               data = nl.load(data_hbm[:, inner.start_offset:inner.end_offset])

Example output for ``H=1024``, ``OUTER_TILE=512``, ``INNER_TILE=128``:

.. code-block:: text

   outer 0, inner 0: size=128, start=0, end=128
   outer 0, inner 1: size=128, start=128, end=256
   outer 0, inner 2: size=128, start=256, end=384
   outer 0, inner 3: size=128, start=384, end=512
   outer 1, inner 0: size=128, start=512, end=640
   outer 1, inner 1: size=128, start=640, end=768
   outer 1, inner 2: size=128, start=768, end=896
   outer 1, inner 3: size=128, start=896, end=1024

Note how ``inner.start_offset`` gives absolute positions, not relative to the outer tile.

Practical Example: Cumsum Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from nkilib.core.utils.tiled_range import TiledRange

   @nki.jit
   def cumsum_tiled(x, H):
       TILE_SIZE = 2048
       carry = 0
       
       for tile in TiledRange(H, TILE_SIZE):
           # Load current tile
           x_tile = nl.load(x[:, tile.start_offset:tile.end_offset])
           
           # Compute cumsum with carry from previous tile
           result = compute_cumsum(x_tile, carry)
           
           # Store result
           nl.store(y[:, tile.start_offset:tile.end_offset], result)
           
           # Update carry for next tile
           carry = result[:, tile.size - 1]

See Also
--------

* :doc:`SbufManager </nki/library/kernel-utils/allocator>` - Memory allocation with scope management
