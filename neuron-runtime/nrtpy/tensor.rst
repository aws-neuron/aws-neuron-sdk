.. meta::
   :description: nrtpy tensor API — allocate device tensors and move data between AWS Neuron NeuronCores and NumPy from Python.
   :date-modified: 2026-07-30
   :keywords: Neuron, nrtpy, NrtpyTensor, NumPy, device tensor, NeuronCore, HBM

.. _nrtpy-tensor-ref:

====================
nrtpy tensor API
====================

This page documents :py:class:`~nrtpy.NrtpyTensor` — the device-resident
tensor with NumPy integration and automatic cleanup.

NrtpyTensor
-----------

.. py:class:: nrtpy.NrtpyTensor(tensor_ref, shape, dtype, name=None)

   A tensor resident in high bandwidth memory (HBM) on the device. Device
   memory is freed
   automatically when the tensor is garbage collected by Python. For explicit
   control, call :py:func:`nrtpy.get_nrtpy_singleton().free_tensor(tensor)`.

   :param tensor_ref: Reference to the underlying device tensor.
   :type tensor_ref: nrtpy.NrtTensor
   :param shape: Shape of the tensor. An integer is treated as a
      single-dimension shape.
   :type shape: tuple[int, ...] or int
   :param dtype: Data type of the tensor, in NumPy format.
   :type dtype: numpy.dtype
   :param name: Optional name for the tensor.
   :type name: str or None

   .. rubric:: Attributes

   .. py:attribute:: shape
      :type: tuple[int, ...]

      Shape of the tensor.

   .. py:attribute:: dtype
      :type: numpy.dtype

      Data type of the tensor.

   .. py:attribute:: name
      :type: str

      Name of the tensor.

   .. py:classmethod:: from_numpy(array, name=None, core_id=0)

      Allocate a device tensor and copy a NumPy array into it.

      :param array: Source array. It is made contiguous before the copy.
      :type array: numpy.ndarray
      :param name: Optional name for the tensor.
      :type name: str or None
      :param int core_id: Target NeuronCore for the allocation (default 0).
      :returns: An ``NrtpyTensor`` backed by the newly allocated device memory.
      :rtype: nrtpy.NrtpyTensor

   .. py:method:: write_from_numpy(array)

      Write new data from a NumPy array into this existing device tensor
      without reallocating.

      :param array: Source array. It is made contiguous before the copy and
         must match the tensor's byte size.
      :type array: numpy.ndarray
      :raises ValueError: If the source array's byte size does not match the
         tensor's byte size.

   .. py:method:: numpy()

      Read the tensor data back from the device as a NumPy array.

      :returns: A NumPy array with the tensor's shape and dtype.
      :rtype: numpy.ndarray

.. note::

   The ``float8_e4m3`` and ``float8_e5m2`` dtypes are reported as ``int8`` by
   ``libnrt``. ``nrtpy`` includes workarounds, but be aware of this when
   inspecting dtypes on FP8 tensors.

Examples
--------

Create, write, and read
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from nrtpy import NrtpyTensor

   # Allocate a device tensor from a NumPy array
   data = np.random.randn(1, 10).astype(np.float32)
   tensor = NrtpyTensor.from_numpy(data, name="input")

   # Overwrite in place (no reallocation)
   new_data = np.zeros((1, 10), dtype=np.float32)
   tensor.write_from_numpy(new_data)

   # Read back to host
   host_array = tensor.numpy()

Multiple cores
~~~~~~~~~~~~~~~

.. code-block:: python

   import nrtpy
   from nrtpy import NrtpyTensor

   nrtpy.configure(visible_cores=[0, 1])

   # Allocate tensors on different cores
   t0 = NrtpyTensor.from_numpy(data, name="core0_input", core_id=0)
   t1 = NrtpyTensor.from_numpy(data, name="core1_input", core_id=1)

Related reference
-----------------

- :ref:`nrtpy-model-ref` — pass tensors to ``NrtpyModel`` for execution and
  benchmarking.
- :ref:`nrtpy-configuration-ref` — choose which NeuronCores tensors are
  allocated on.
