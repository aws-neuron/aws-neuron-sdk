.. meta::
    :description: API reference for kernel utility modules in the NKI Library.
    :date-modified: 02/13/2026

.. _nkl_kernel_utils_home:

NKI Library Kernel Utilities Reference
======================================

The NKI Library provides utility modules to simplify common patterns in NKI kernel development. These utilities help manage memory allocation, tensor views, dimension tiling, and data broadcasting.

**Source code for these utilities can be found at**: https://github.com/aws-neuron/nki-library

Memory Management
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`SbufManager (Allocator) </nki/library/kernel-utils/allocator>`
     - Stack-based SBUF memory allocator with scope management and multi-buffering support.

Tensor Operations
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`TensorView </nki/library/kernel-utils/tensor-view>`
     - Zero-copy tensor view operations including slicing, permuting, reshaping, and broadcasting.
   * - :doc:`stream_shuffle_broadcast </nki/library/kernel-utils/stream-shuffle-broadcast>`
     - Broadcasts a single partition across the partition dimension using hardware shuffle.

Iteration Helpers
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60

   * - :doc:`TiledRange </nki/library/kernel-utils/tiled-range>`
     - Divides dimensions into tiles with automatic remainder handling.

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Memory Management

    SbufManager (Allocator) <allocator>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Tensor Operations

    TensorView <tensor-view>
    stream_shuffle_broadcast <stream-shuffle-broadcast>

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Iteration Helpers

    TiledRange <tiled-range>
