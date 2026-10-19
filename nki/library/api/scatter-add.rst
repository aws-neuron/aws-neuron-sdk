.. meta::
    :description: Scatter-add from src into input based on indices using gather-accumulate-scatter pattern.
    :date-modified: 10/19/2026

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

**Source code for this kernel API can be found at**: `scatter_add.py <https://github.com/aws-neuron/nki-library/blob/2.33/src/nkilib_src/nkilib/experimental/misc/scatter_add.py>`_

scatter_add
^^^^^^^^^^^

.. py:function:: scatter_add(input: nl.NkiTensor, dim: int, index: nl.NkiTensor, src: nl.NkiTensor, unique_indices: bool = True) -> nl.NkiTensor

   Scatter-add from src into input based on indices using gather-accumulate-scatter pattern.

   :param input: [N, D], Destination tensor to accumulate into (modified in-place)
   :type input: ``nl.NkiTensor``
   :param dim: Dimension along which to scatter (must be 0)
   :type dim: ``int``
   :param index: [K], 1D tensor of row indices into input
   :type index: ``nl.NkiTensor``
   :param src: [K, D], Source values to scatter-add
   :type src: ``nl.NkiTensor``
   :param unique_indices: If True (default), assume destination indices are unique within each 128-row tile and use the plain tile-wide gather/scatter path. If False, duplicate rows within a tile are pre-combined on the Tensor Engine so repeated indices accumulate correctly, which embedding-gradient scatters need. Costs one extra 128x128 matmul per PSUM-wide column chunk.
   :type unique_indices: ``bool``
   :return: [N, D], The input tensor with scattered values added
   :rtype: ``nl.NkiTensor``

   **Notes**:

   * Input and src tensors must be 2D
   * Index tensor must be 1D
   * dim must be 0
   * Indices within a tile of 128 rows should be unique for correctness

   **Dimensions**:

   * N: Number of rows in input tensor
   * D: Feature dimension size
   * K: Number of source rows / indices

