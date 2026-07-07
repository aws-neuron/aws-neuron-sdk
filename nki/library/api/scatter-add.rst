.. meta::
    :description: Scatter-add from src into input based on indices using gather-accumulate-scatter pattern.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.misc

Scatter-Add Kernel API Reference
================================

Scatter-add from src into input based on indices using gather-accumulate-scatter pattern.

Equivalent to PyTorch's ``input.scatter_add(dim=0, index=index, src=src)``. Performs: ``input[index[i], j] += src[i, j]`` for all i, j.

Background
-----------

The ``scatter_add`` kernel scatter-adds rows from ``src`` into a 2D input tensor based on a 1D index tensor, using a gather-accumulate-scatter pattern. Indices within a tile of 128 rows should be unique for correctness.

API Reference
--------------

**Source code for this kernel API can be found at**: `scatter_add.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/misc/scatter_add.py>`_

scatter_add
^^^^^^^^^^^

.. py:function:: scatter_add(input: nl.ndarray, dim: int, index: nl.ndarray, src: nl.ndarray) -> nl.ndarray

   Scatter-add from src into input based on indices using gather-accumulate-scatter pattern.

   :param input: [N, D], Destination tensor to accumulate into (modified in-place)
   :type input: ``nl.ndarray``
   :param dim: Dimension along which to scatter (must be 0)
   :type dim: ``int``
   :param index: [K], 1D tensor of row indices into input
   :type index: ``nl.ndarray``
   :param src: [K, D], Source values to scatter-add
   :type src: ``nl.ndarray``
   :return: [N, D], The input tensor with scattered values added
   :rtype: ``nl.ndarray``

   **Notes**:

   * Input and src tensors must be 2D
   * Index tensor must be 1D
   * dim must be 0
   * Indices within a tile of 128 rows should be unique for correctness

   **Dimensions**:

   * N: Number of rows in input tensor
   * D: Feature dimension size
   * K: Number of source rows / indices

