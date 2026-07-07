.. meta::
    :description: Gather rows from input based on indices using indirect DMA load.
    :date-modified: 06/11/2026

.. currentmodule:: nkilib.experimental.misc

Gather Kernel API Reference
===========================

Gather rows from input based on indices using indirect DMA load.

Equivalent to PyTorch's ``input[index]`` for dim=0 with 2D input and 1D index.

Background
-----------

The ``gather`` kernel gathers rows from a 2D input tensor based on a 1D index tensor using an indirect DMA load, producing ``output[i, :] = input[index[i], :]``.

API Reference
--------------

**Source code for this kernel API can be found at**: `gather.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/misc/gather.py>`_

gather
^^^^^^

.. py:function:: gather(input: nl.ndarray, dim: int, index: nl.ndarray) -> nl.ndarray

   Gather rows from input based on indices using indirect DMA load.

   :param input: [N, D], Source tensor to gather from
   :type input: ``nl.ndarray``
   :param dim: Dimension along which to gather (must be 0)
   :type dim: ``int``
   :param index: [K], 1D tensor of row indices into input
   :type index: ``nl.ndarray``
   :return: [K, D], Gathered result where output[i, :] = input[index[i], :]
   :rtype: ``nl.ndarray``

   **Notes**:

   * Input tensor must be 2D
   * Index tensor must be 1D
   * dim must be 0
   * Under LNC sharding (``num_shards > 1``), the index size ``K`` must be divisible by ``num_shards``

   **Dimensions**:

   * N: Number of rows in input tensor
   * D: Feature dimension size
   * K: Number of indices (output rows)

