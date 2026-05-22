.. meta::
    :description: Elementwise add scalar to tensor.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.foreach

Foreach Elementwise Kernel API Reference
=========================================

Elementwise add scalar to tensor.

Computes out = data + scalar using SPMD parallelization across cores.

Background
-----------

The ``add_scalar_kernel`` kernel performs elementwise arithmetic operations (add, subtract, multiply, divide) between tensors and scalars or between pairs of tensors, using SPMD parallelization across cores.

API Reference
--------------

**Source code for this kernel API can be found at**: `foreach_elementwise.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/foreach/foreach_elementwise.py>`_

add_scalar_kernel
^^^^^^^^^^^^^^^^^

.. py:function:: add_scalar_kernel(data: nl.ndarray, scalar_tensor: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise add scalar to tensor.

   :param data: [N], Input tensor on HBM. Must have ndim >= 1.
   :type data: ``nl.ndarray``
   :param scalar_tensor: [P_MAX, 1], Scalar broadcast tensor on HBM.
   :type scalar_tensor: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

sub_scalar_kernel
^^^^^^^^^^^^^^^^^

.. py:function:: sub_scalar_kernel(data: nl.ndarray, scalar_tensor: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise subtract scalar from tensor.

   :param data: [N], Input tensor on HBM. Must have ndim >= 1.
   :type data: ``nl.ndarray``
   :param scalar_tensor: [P_MAX, 1], Scalar broadcast tensor on HBM.
   :type scalar_tensor: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

mul_scalar_kernel
^^^^^^^^^^^^^^^^^

.. py:function:: mul_scalar_kernel(data: nl.ndarray, scalar_tensor: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise multiply tensor by scalar.

   :param data: [N], Input tensor on HBM. Must have ndim >= 1.
   :type data: ``nl.ndarray``
   :param scalar_tensor: [P_MAX, 1], Scalar broadcast tensor on HBM.
   :type scalar_tensor: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

div_scalar_kernel
^^^^^^^^^^^^^^^^^

.. py:function:: div_scalar_kernel(data: nl.ndarray, scalar_tensor: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise divide tensor by scalar.

   :param data: [N], Input tensor on HBM. Must have ndim >= 1.
   :type data: ``nl.ndarray``
   :param scalar_tensor: [P_MAX, 1], Scalar broadcast tensor on HBM.
   :type scalar_tensor: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

add_tensor_kernel
^^^^^^^^^^^^^^^^^

.. py:function:: add_tensor_kernel(data1: nl.ndarray, data2: nl.ndarray, alpha_tensor: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise add tensors with alpha scaling.

   :param data1: [N], First input tensor on HBM. Must have ndim >= 1.
   :type data1: ``nl.ndarray``
   :param data2: [N], Second input tensor on HBM.
   :type data2: ``nl.ndarray``
   :param alpha_tensor: [P_MAX, 1], Alpha scalar broadcast tensor on HBM.
   :type alpha_tensor: ``nl.ndarray``
   :param numel: Number of elements in data1.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

sub_tensor_kernel
^^^^^^^^^^^^^^^^^

.. py:function:: sub_tensor_kernel(data1: nl.ndarray, data2: nl.ndarray, alpha_tensor: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise subtract tensors with alpha scaling.

   :param data1: [N], First input tensor on HBM. Must have ndim >= 1.
   :type data1: ``nl.ndarray``
   :param data2: [N], Second input tensor on HBM.
   :type data2: ``nl.ndarray``
   :param alpha_tensor: [P_MAX, 1], Alpha scalar broadcast tensor on HBM.
   :type alpha_tensor: ``nl.ndarray``
   :param numel: Number of elements in data1.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

mul_tensor_kernel
^^^^^^^^^^^^^^^^^

.. py:function:: mul_tensor_kernel(data1: nl.ndarray, data2: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise multiply tensors.

   :param data1: [N], First input tensor on HBM. Must have ndim >= 1.
   :type data1: ``nl.ndarray``
   :param data2: [N], Second input tensor on HBM.
   :type data2: ``nl.ndarray``
   :param numel: Number of elements in data1.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

div_tensor_kernel
^^^^^^^^^^^^^^^^^

.. py:function:: div_tensor_kernel(data1: nl.ndarray, data2: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise divide tensors.

   :param data1: [N], First input tensor on HBM. Must have ndim >= 1.
   :type data1: ``nl.ndarray``
   :param data2: [N], Second input tensor on HBM.
   :type data2: ``nl.ndarray``
   :param numel: Number of elements in data1.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

addcdiv_kernel
^^^^^^^^^^^^^^

.. py:function:: addcdiv_kernel(data: nl.ndarray, data1: nl.ndarray, data2: nl.ndarray, value_tensor: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise addcdiv: data + value * (data1 / data2).

   :param data: [N], Base input tensor on HBM.
   :type data: ``nl.ndarray``
   :param data1: [N], Numerator tensor on HBM.
   :type data1: ``nl.ndarray``
   :param data2: [N], Denominator tensor on HBM.
   :type data2: ``nl.ndarray``
   :param value_tensor: [P_MAX, 1], Scalar value broadcast tensor on HBM.
   :type value_tensor: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

addcmul_kernel
^^^^^^^^^^^^^^

.. py:function:: addcmul_kernel(data: nl.ndarray, data1: nl.ndarray, data2: nl.ndarray, value_tensor: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise addcmul: data + value * (data1 * data2).

   :param data: [N], Base input tensor on HBM.
   :type data: ``nl.ndarray``
   :param data1: [N], First multiplicand tensor on HBM.
   :type data1: ``nl.ndarray``
   :param data2: [N], Second multiplicand tensor on HBM.
   :type data2: ``nl.ndarray``
   :param value_tensor: [P_MAX, 1], Scalar value broadcast tensor on HBM.
   :type value_tensor: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

lerp_kernel
^^^^^^^^^^^

.. py:function:: lerp_kernel(data: nl.ndarray, end: nl.ndarray, weight_tensor: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise linear interpolation: data + weight * (end - data).

   :param data: [N], Start tensor on HBM.
   :type data: ``nl.ndarray``
   :param end: [N], End tensor on HBM.
   :type end: ``nl.ndarray``
   :param weight_tensor: [P_MAX, 1], Interpolation weight broadcast tensor on HBM.
   :type weight_tensor: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

sqrt_kernel
^^^^^^^^^^^

.. py:function:: sqrt_kernel(data: nl.ndarray, numel: int) -> nl.ndarray

   Elementwise square root.

   :param data: [N], Input tensor on HBM. Elements must be non-negative.
   :type data: ``nl.ndarray``
   :param numel: Number of elements in data.
   :type numel: ``int``
   :return: [N], Output tensor on HBM.
   :rtype: ``nl.ndarray``

