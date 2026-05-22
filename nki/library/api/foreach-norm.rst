.. meta::
    :description: Compute L2 norm (Euclidean norm) of input tensor.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.foreach

Foreach Norm Kernel API Reference
==================================

Compute L2 norm (Euclidean norm) of input tensor.

Computes sqrt(sum(x^2)) using SPMD parallelization across 2 cores with fused activation-reduce and sendrecv-based cross-core reduction.

Background
-----------

The ``l2_norm_kernel`` kernel computes the L2 norm (Euclidean norm) of an input tensor using SPMD parallelization with fused activation-reduce and cross-core reduction.

API Reference
--------------

**Source code for this kernel API can be found at**: `foreach_norm.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/foreach/foreach_norm.py>`_

l2_norm_kernel
^^^^^^^^^^^^^^

.. py:function:: l2_norm_kernel(data: nl.ndarray, numel: int) -> nl.ndarray

   Compute L2 norm (Euclidean norm) of input tensor.

   :param data: [N], Input tensor on HBM.
   :type data: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [1, 1], L2 norm scalar on HBM.
   :rtype: ``nl.ndarray``

l1_norm_kernel
^^^^^^^^^^^^^^

.. py:function:: l1_norm_kernel(data: nl.ndarray, numel: int) -> nl.ndarray

   Compute L1 norm (Manhattan norm) of input tensor.

   :param data: [N], Input tensor on HBM.
   :type data: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [1, 1], L1 norm scalar on HBM.
   :rtype: ``nl.ndarray``

linf_norm_kernel
^^^^^^^^^^^^^^^^

.. py:function:: linf_norm_kernel(data: nl.ndarray, numel: int) -> nl.ndarray

   Compute Linf norm (max norm) of input tensor.

   :param data: [N], Input tensor on HBM.
   :type data: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [1, 1], Linf norm scalar on HBM.
   :rtype: ``nl.ndarray``

