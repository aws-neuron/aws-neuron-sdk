.. meta::
    :description: Elementwise addition with dynamic partition dimension tiling.
    :date-modified: 04/09/2026

.. currentmodule:: nkilib.experimental.dynamic_shapes

Dynamic Elementwise Add Kernel API Reference
============================================

Elementwise addition with dynamic partition dimension tiling.

Computes output = input_a + input_b for 2D bf16 tensors where the number of M-dimension tiles to process is determined at runtime via num_m_tiles. Optimized for M dimensions up to 2048 and H dimensions up to 8192.

Background
-----------

The ``dynamic_elementwise_add`` kernel computes elementwise addition where the number of M-dimension tiles to process is determined at runtime. This demonstrates NKI's support for dynamic loop bounds using ``sequential_range`` with runtime-variable trip counts.

API Reference
--------------

**Source code for this kernel API can be found at**: `dynamic_elementwise_add.py <https://github.com/aws-neuron/nki-library/blob/2.32/src/nkilib_src/nkilib/experimental/dynamic_shapes/dynamic_elementwise_add.py>`_

dynamic_elementwise_add
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: dynamic_elementwise_add(input_a: nl.ndarray, input_b: nl.ndarray, num_m_tiles: nl.ndarray) -> nl.ndarray

   Elementwise addition with dynamic partition dimension tiling.

   :param input_a: [M, H], First input tensor, bf16, on HBM.
   :type input_a: ``nl.ndarray``
   :param input_b: [M, H], Second input tensor, bf16, on HBM. Must match input_a shape.
   :type input_b: ``nl.ndarray``
   :param num_m_tiles: [1, 1], int32 scalar tensor on HBM. Value = number of M-tiles to process (0 <= num_m_tiles <= M // P_MAX).
   :type num_m_tiles: ``nl.ndarray``
   :return: [M, H], bf16 output tensor on HBM. Elements in the first (num_m_tiles * P_MAX) rows contain input_a + input_b; remaining rows are unmodified.
   :rtype: ``nl.ndarray``

   **Notes**:

   * M must be divisible by P_MAX (128)
   * H must be divisible by H_TILE_SIZE (512)
   * input_a and input_b must have identical shapes

   **Dimensions**:

   * M: Row dimension, tiled at P_MAX (128). Dynamic at runtime via num_m_tiles.
   * H: Hidden/column dimension, tiled at H_TILE_SIZE (512). Static.

