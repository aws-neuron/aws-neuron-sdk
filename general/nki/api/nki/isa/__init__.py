import numpy as np
import ml_dtypes

def activation(op, data, *, bias=None, scale=1.0, mask=None, dtype=None, **kwargs):
  r"""
  Apply an activation function on every element of the input tile using Scalar Engine. The activation
  function is specified in the ``op`` input field (see :ref:`nki-act-func` for a list of
  supported activation functions).

  The activation instruction can optionally multiply the input ``data`` by a scalar or vector ``scale``
  and then add another vector ``bias`` before the activation function is applied,
  at no additional performance cost:

  .. math::
        output = f_{act}(data * scale + bias)

  When the scale is a scalar, it must be a compile-time constant. In this case, the scale
  is broadcasted to all the elements in the input ``data`` tile.
  When the scale/bias is a vector, it must have the same partition axis size as the input ``data`` tile
  and only one element per partition.
  In this case, the element of scale/bias within each partition is broadcasted to
  elements of the input ``data`` tile in the same partition.

  Note, the Scalar Engine always performs the math operations in float32 precision.
  Therefore, the engine automatically casts the input ``data`` tile to float32 before
  performing multiply/add/activate specified in the activation instruction.
  The engine is also capable of casting the float32 math results into another
  output data type specified by the ``dtype`` field at no additional performance cost.
  If ``dtype`` field is not specified, Neuron Compiler will set output data type of the instruction
  to be the same as input data type of ``data``. On the other hand, the ``scale`` parameter must
  have a float32 data type, while the ``bias`` parameter can be float32/float16/bfloat16.

  The input ``data`` tile can be an SBUF or PSUM tile. Similarly, the instruction
  can write the output tile into either SBUF or PSUM, which is specified
  using the ``buffer`` field. If not specified, ``nki.language.sbuf`` is selected by default.

  **Estimated instruction cost:**

  ``max(MIN_II, N)`` Scalar Engine cycles, where

  - ``N`` is the number of elements per partition in ``data``.
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles. 
    ``MIN_II`` is roughly 64 engine cycles. 

  :param op: an activation function (see :ref:`nki-act-func` for supported functions)
  :param data: the input tile; layout: (partition axis <= 128, free axis)
  :param bias: a vector with the same partition axis size as ``data``
               for broadcast add (after broadcast multiply with ``scale``)
  :param scale: a scalar or a vector with the same partition axis size as ``data``
                for broadcast multiply
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :return: output tile of the activation instruction; layout: same as input ``data`` tile

  Example:

  .. nki_example:: ../../test/test_nki_isa_activation.py
   :language: python

  """
  ...

def activation_reduce(op, data, *, reduce_op, reduce_res, bias=None, scale=1.0, mask=None, dtype=None, **kwargs):
  r"""
  Perform the same computation as ``nisa.activation`` and also a reduction along the free dimension of the
  ``nisa.activation`` result using Scalar Engine.

  Refer to :doc:`nisa.activation <nki.isa.activation>` for semantics of ``op/data/bias/scale``.

  In addition to :doc:`nisa.activation <nki.isa.activation>` computation, this API also performs a reduction
  along the free dimension(s) of the :doc:`nisa.activation <nki.isa.activation>` result, at a small additional
  performance cost. The reduction result is returned in ``reduce_res`` in-place, which must be a
  SBUF/PSUM tile with the same partition axis size as the input tile ``data`` and one element per partition.
  On NeuronCore-v2, the ``reduce_op`` can only be an addition, ``np.add`` or ``nl.add``.

  Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will
  reduce across all of them.

  Mathematically, this API performs the following computation:

  .. math::
        output = f_{act}(data * scale + bias) \\
        reduce\_res = reduce\_op(output, axis=<FreeAxis>)

  **Estimated instruction cost:**

  ``max(MIN_II, N) + MIN_II`` Scalar Engine cycles, where

  - ``N`` is the number of elements per partition in ``data``, and
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
    ``MIN_II`` is roughly 64 engine cycles.

  :param op: an activation function (see :ref:`nki-act-func` for supported functions)
  :param data: the input tile; layout: (partition axis <= 128, free axis)
  :param reduce_op: the reduce operation to perform on the free dimension of the activation result
  :param reduce_res: a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                  is the partition axis size of the input ``data`` tile. The result of ``sum(ReductionResult)``
                  is written in-place into the tensor.
  :param bias: a vector with the same partition axis size as ``data``
               for broadcast add (after broadcast multiply with ``scale``)
  :param scale: a scalar or a vector with the same partition axis size as ``data``
                for broadcast multiply
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :return: output tile of the activation instruction; layout: same as input ``data`` tile
  """
  ...

def affine_select(pred, on_true_tile, on_false_value, *, mask=None, dtype=None, **kwargs):
  r"""
  Select elements between an input tile ``on_true_tile`` and a scalar value ``on_false_value``
  according to a boolean predicate tile using GpSimd Engine. The predicate tile is
  calculated on-the-fly in the engine using the input affine expression ``pred``.
  The input tile ``on_true_tile``, the calculated boolean predicate tile expressed by ``pred``,
  and the returned output tile of this instruction
  must have the same shape. If the predicate value of a given position is ``True``,
  the corresponding output element will take the element from ``on_true_tile`` in the same position.
  If the predicate value of a given position is ``False``,
  the corresponding output element will take the value of ``on_false_value``.

  A common use case for ``affine_select`` is to apply a causal mask on the attention
  scores for transformer decoder models.

  This instruction allows any float or 8-bit/16-bit integer data types
  for both the input data tile and output tile (see :ref:`nki-dtype` for more information).
  The output tile data type is specified using
  the ``dtype`` field. If ``dtype`` is not specified, the output data type will be the same as
  the input data type of ``data``. However, the data type of ``on_false_value`` must be float32,
  regardless of the input/output tile data types.

  **Estimated instruction cost:**

  ``150 + N`` GpSimd Engine cycles, where ``N`` is the number of elements per partition in ``on_true_tile``.

  :param pred: an affine expression that defines the boolean predicate
  :param on_true_tile: an input tile for selection with a ``True`` predicate value
  :param on_false_value: a scalar value for selection with a ``False`` predicate value
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
  :return: an output tile with values selected from either ``on_true_tile`` or
           ``on_false_value`` according to the following equation:
           output[x] = (pred[x] > 0) ? on_true_tile[x] : on_false_value

  Example:

  .. nki_example:: ../../test/test_nki_isa_affine_select.py
   :language: python

  """
  ...

def bn_aggr(data, *, mask=None, dtype=None, **kwargs):
  r"""
  Aggregate one or multiple ``bn_stats`` outputs to generate
  a mean and variance per partition using Vector Engine.

  The input ``data`` tile
  effectively has an array of ``(count, mean, variance*count)`` tuples per partition
  produced by  :doc:`bn_stats <nki.isa.bn_stats>` instructions. Therefore, the number of elements per partition
  of ``data`` must be a modulo of three.

  Note, if you need to aggregate multiple ``bn_stats`` instruction outputs,
  it is recommended to declare a SBUF tensor
  and then make each ``bn_stats`` instruction write its output into the
  SBUF tensor at different offsets (see example implementation
  in Example 2 in :doc:`bn_stats <nki.isa.bn_stats>`).

  Vector Engine performs the statistics aggregation in float32 precision.
  Therefore, the engine automatically casts the input ``data`` tile to float32 before
  performing float32 computation and is capable of casting
  the float32 computation results into another data type specified by the ``dtype`` field,
  at no additional performance cost. If ``dtype`` field is not specified, the instruction
  will cast the float32 results back to the same data type as the input ``data`` tile.


  **Estimated instruction cost:**

  ``max(MIN_II, 13*(N/3))`` Vector Engine cycles, where ``N`` is the number of elements per partition in ``data`` and
  ``MIN_II`` is the minimum instruction initiation interval for small input tiles. ``MIN_II`` is roughly 64 engine cycles.

  :param data: an input tile with results of one or more :doc:`bn_stats <nki.isa.bn_stats>`
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :return: an output tile with two elements per partition: a mean followed by a variance
  """
  ...

def bn_stats(data, *, mask=None, dtype=None, **kwargs):
  r"""
  Compute mean- and variance-related statistics for each partition of an input tile ``data``
  in parallel using Vector Engine.

  The output tile of the instruction has 6 elements per partition:

  - the ``count`` of the even elements (of the input tile elements from the same partition)
  - the ``mean`` of the even elements
  - ``variance * count`` of the even elements
  - the ``count`` of the odd elements
  - the ``mean`` of the odd elements
  - ``variance * count`` of the odd elements

  To get the final mean and variance of the input tile,
  we need to pass the above ``bn_stats`` instruction output
  into the :doc:`bn_aggr <nki.isa.bn_aggr>`
  instruction, which will output two elements per partition:

  - mean (of the original input tile elements from the same partition)
  - variance

  Due to hardware limitation, the number of elements per partition
  (i.e., free dimension size) of the input ``data`` must not exceed 512 (nl.tile_size.bn_stats_fmax).
  To calculate per-partition mean/variance of a tensor with more than
  512 elements in free dimension, we can invoke ``bn_stats`` instructions
  on each 512-element tile and use a single ``bn_aggr`` instruction to
  aggregate ``bn_stats`` outputs from all the tiles. Refer to Example 2
  for an example implementation.

  Vector Engine performs the above statistics calculation in float32 precision.
  Therefore, the engine automatically casts the input ``data`` tile to float32 before
  performing float32 computation and is capable of casting
  the float32 computation results into another data type specified by the ``dtype`` field,
  at no additional performance cost. If ``dtype`` field is not specified, the instruction
  will cast the float32 results back to the same data type as the input ``data`` tile.

  **Estimated instruction cost:**

  ``max(MIN_II, N)`` Vector Engine cycles, where ``N`` is the number of elements per partition in ``data`` and
  ``MIN_II`` is the minimum instruction initiation interval for small input tiles. ``MIN_II`` is roughly 64 engine cycles.

  :param data: the input tile (up to 512 elements per partition)
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :return: an output tile with 6-element statistics per partition

  Example:

  .. nki_example:: ../../test/test_nki_isa_bn_stats.py
   :language: python

  """
  ...

def builtin_custom_op(function_name, lib_file_name, ulib_to_ucode_version, ulib_to_isa_version, srcs, dsts, **kwargs):
  ...

def dma_copy(src, dst):
  r"""
  Copy data from `src` to `dst` using DMA engine. Both `src` and `dst` tiles can be in device memory (HBM) or SBUF.
  However, if both `src` and `dst` tiles are in SBUF, consider using
  :doc:`nisa.tensor_copy <nki.isa.tensor_copy>` instead for better performance.

  :param src: the source of copy.
  :param dst: the dst of copy

  A cast will happen if the `src` and `dst` have different dtype.

  Example:

  .. nki_example:: ../../test/test_nki_isa_dma_copy.py
   :language: python
   :marker: NKI_EXAMPLE_7

  """
  ...

dma_engine = ...
r"""DMA Engine"""

def dma_transpose(src, *, mask=None, dtype=None, **kwargs):
  r"""
  Perform a transpose on input ``src`` using DMA Engine.

  The permutation of transpose follow the rules described below:
  For 2-d input tile, the permutation will be [1, 0]
  For 3-d input tile, the permutation will be [2, 1, 0]
  For 4-d input tile, the permutation will be [3, 1, 2, 0]

  **Estimated instruction cost:**

  TBD

  :param src: the source of transpose, must be a tile in HBM or SBUF.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :return: a tile with transposed content

  Example:

  .. nki_example:: ../../test/test_nki_nl_load_transpose2d.py
   :language: python
   :marker: NKI_EXAMPLE_20

  """
  ...

def dropout(data, prob, *, mask=None, dtype=None, **kwargs):
  r"""
  Randomly replace some elements of the input tile ``data`` with zeros
  based on input probabilities using Vector Engine.
  The probability of replacing input elements with zeros (i.e., drop probability)
  is specified using the ``prob`` field:
  - If the probability is 1.0, all elements are replaced with zeros.
  - If the probability is 0.0, all elements are kept with their original values.

  The ``prob`` field can be a scalar constant or a tile of shape ``(data.shape[0], 1)``,
  where each partition contains one drop probability value.
  The drop probability value in each partition is applicable to the input
  ``data`` elements from the same partition only.

  Data type of the input ``data`` tile can be any valid NKI data types
  (see :ref:`nki-dtype` for more information).
  However, data type of ``prob`` has restrictions based on the data type of ``data``:

  - If data type of ``data`` is any of the integer types (e.g., int32, int16),
    ``prob`` data type must be float32
  - If data type of data is any of the float types (e.g., float32, bfloat16),
    ``prob`` data can be any valid float type

  The output data type of this instruction is specified by the ``dtype`` field. The output data type
  must match the input data type of ``data`` if input data type is any of the integer types.
  Otherwise, output data type can be any valid NKI data types. If output data type is not specified,
  it is default to be the same as input data type.

  **Estimated instruction cost:**

  ``max(MIN_II, N)`` Vector Engine cycles, where ``N`` is the number of elements per partition in ``data``,
  and ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
  ``MIN_II`` is roughly 64 engine cycles.

  :param data: the input tile
  :param prob: a scalar or a tile of shape ``(data.shape[0], 1)`` to indicate the
               probability of replacing elements with zeros
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.

  :return: an output tile of the dropout result

  Example:

  .. nki_example:: ../../test/test_nki_isa_dropout.py
   :language: python

  """
  ...

def get_nc_version():
  r""" Returns the ``nc_version`` of the current target context. """
  ...

gpsimd_engine = ...
r"""GpSimd Engine"""

def iota(expr, dtype, *, mask=None, **kwargs):
  r"""
  Build a constant literal in SBUF using GpSimd Engine,
  rather than transferring the constant literal values from the host to device.

  The iota instruction takes an affine expression of ``nki.language.arange()``
  indices as the input pattern to generate constant index values
  (see examples below for more explanation). The index values are computed in
  32-bit integer math. The GpSimd Engine is capable of casting the integer results
  into any desirable data type (specified by ``dtype``) before writing
  them back to SBUF, at no additional performance cost.

  **Estimated instruction cost:**

  ``150 + N`` GpSimd Engine cycles, where ``N`` is the number of elements per partition in the output tile.

  :param expr: an input affine expression of ``nki.language.arange()``
  :param dtype: output data type of the generated constant literal (see :ref:`nki-dtype` for more information)
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :return: an output tile in SBUF

  Example:

  .. nki_example:: ../../test/test_nki_isa_iota.py
   :language: python

  """
  ...

def local_gather(src_buffer, index, num_elem_per_idx=1, num_valid_indices=None, *, mask=None):
  r"""
  Gather SBUF data in ``src_buffer`` using ``index`` on GpSimd Engine.

  Each of the eight GpSimd cores in GpSimd Engine connects to 16 contiguous SBUF partitions
  (e.g., core[0] connected to partition[0:16]) and performs gather from the connected 16
  SBUF partitions *independently* in parallel. The indices used for gather on each core should also
  come from the same 16 connected SBUF partitions.

  During execution of the instruction, each GpSimd core reads a 16-partition slice from ``index``, flattens
  all indices into a 1D array ``indices_1d`` (along the partition dimension first).
  By default with no ``num_valid_indices`` specified, each GpSimd core
  will treat all indices from its corresponding 16-partition ``index`` slice as valid indices.
  However, when the number of valid indices per core
  is not a multiple of 16, users can explicitly specify the valid index count per core in ``num_valid_indices``.
  Note, ``num_valid_indices`` must not exceed the total element count in each 16-partition ``index`` slice
  (i.e., ``num_valid_indices <= index.size / (index.shape[0] / 16)``).

  Next, each GpSimd core uses the flattened ``indices_1d`` indices as *partition offsets* to gather from
  the connected 16-partition slice of ``src_buffer``. Optionally, this API also allows gathering of multiple
  contiguous elements starting at each index to improve gather throughput, as indicated by ``num_elem_per_idx``.
  Behavior of out-of-bound index access is undefined.

  Even though all eight GpSimd cores can gather with completely different indices, a common use case for
  this API is to make all cores gather with the same set of indices (i.e., partition offsets). In this case,
  users can generate indices into 16 partitions, replicate them eight times to 128 partitions and then feed them into
  ``local_gather``.

  As an example, if ``src_buffer`` is (128, 512) in shape and ``index`` is (128, 4) in shape, where the partition
  dimension size is 128, ``local_gather`` effectively performs the following operation:

  .. nki_example:: ../../test/test_nki_isa_local_gather.py
   :language: python
   :marker:   NUMPY_SEMANTICS

  ``local_gather`` preserves the input data types from ``src_buffer`` in the gather output.
  Therefore, no data type casting is allowed in this API. The indices in ``index`` tile must be uint16 types.

  This API has three tile size constraints [subject to future relaxation]:

  #. The partition axis size of ``src_buffer`` must match that of ``index`` and must
     be a multiple of 16. In other words, ``src_buffer.shape[0] == index.shape[0] and src_buffer.shape[0] % 16 == 0``.
  #. The number of contiguous elements to gather per index per partition ``num_elem_per_idx``
     must be one of the following values: ``[1, 2, 4, 8, 16, 32]``.
  #. The number of indices for gather per core must be less than or equal to 4096.

  **Estimated instruction cost:**

  ``150 + (num_valid_indices * num_elem_per_idx)/C`` GpSimd Engine cycles, where ``C`` can be calculated
  using
  ``((28 + t * num_elem_per_idx)/(t * num_elem_per_idx)) / min(4/dtype_size, num_elem_per_idx)``.
  ``dtype_size`` is the size of ``src_buffer.dtype`` in bytes.
  Currently, ``t`` is a constant 4, but subject to change in future software implementation.

  :param src_buffer: an input tile for gathering.
  :param index: an input tile with indices used for gathering.
  :param num_elem_per_idx: an optional integer value to read multiple contiguous elements per index per partition; default is 1.
  :param num_valid_indices: an optional integer value to specify the number of valid indices per GpSimd core; default is
                            ``index.size / (index.shape[0] / 16)``.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :return: an output tile of the gathered data

  **Example:**

  .. nki_example:: ../../test/test_nki_isa_local_gather.py
   :language: python


  Click :download:`here <../../test/test_nki_isa_local_gather.py>` to download the
  full NKI code example with equivalent numpy implementation.
  """
  ...

def memset(shape, value, dtype, *, mask=None, engine=0, **kwargs):
  r"""
  Initialize a tile filled with a compile-time constant value using Vector Engine.
  The shape of the tile is specified in the ``shape`` field and the
  initialized value in the ``value`` field.
  The memset instruction supports all valid NKI dtypes
  (see :ref:`nki-dtype`).

  :param shape: the shape of the output tile; layout: (partition axis, free axis)
  :param value: the constant value to initialize with
  :param dtype: data type of the output tile (see :ref:`nki-dtype` for more information)
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param engine: specify which engine to use for reciprocal: ``nki.isa.vector_engine`` or ``nki.isa.gpsimd_engine`` ;
                 ``nki.isa.unknown_engine`` by default, lets compiler select the best engine for the given
                 input tile shape
  :return: a tile with shape `shape` whose elements are initialized to `value`.

  **Estimated instruction cost:**

  Given ``N`` is the number of elements per partition in the output tile, and ``MIN_II`` is the minimum
  instruction initiation interval for small input tiles. ``MIN_II`` is roughly 64 engine cycles.

  - If the initialized value is zero and output data type is bfloat16/float16, ``max(MIN_II, N/2)`` Vector Engine cycles;
  - Otherwise, ``max(MIN_II, N)`` Vector Engine cycles


  Example:

  .. nki_example:: ../../test/test_nki_isa_memset.py
   :language: python
   :marker: NKI_EXAMPLE_7

  """
  ...

def nc_matmul(stationary, moving, *, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, mask=None, **kwargs):
  r"""
  Compute ``stationary.T @ moving`` matrix multiplication using Tensor Engine.

  The ``nc_matmul`` instruction *must* read inputs from SBUF and
  write outputs to PSUM. Therefore, the ``stationary`` and ``moving`` must be SBUF tiles, and the result
  tile is a PSUM tile.

  The nc_matmul instruction currently supports ``float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32/float32``
  input data types as listed in :ref:`nki-dtype`.
  The matmul accumulation and results are always in float32.

  The Tensor Engine imposes special layout constraints on the input tiles.
  First, the partition axis sizes of the ``stationary`` and ``moving`` tiles must be identical and ``<=128``,
  which corresponds to the contraction dimension of the matrix multiplication. Second, the free axis
  sizes of ``stationary`` and ``moving`` tiles must be ``<= 128`` and ``<=512``, respectively,
  For example, ``stationary.shape = (128, 126)``; ``moving.shape = (128, 512)`` and ``nc_matmul(stationary,moving)``
  returns a tile of ``shape = (126, 512)``. For more information about the matmul layout, see :ref:`arch_guide_tensor_engine`.


  .. figure:: ../../img/arch_images/matmul.png
    :align: center
    :width: 100%

    MxKxN Matrix Multiplication Visualization.

  If the contraction dimension of the matrix multiplication
  exceeds ``128``, you may accumulate multiple ``nc_matmul`` instruction output tiles into the same PSUM tile.
  See example code snippet below.

  **Estimated instruction cost:**

  The Tensor Engine has complex performance characteristics given its data flow and pipeline design. The below formula
  is the *average* nc_matmul cost assuming many ``nc_matmul`` instructions of the same shapes running back-to-back
  on the engine:

  .. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Cost `(Tensor Engine Cycles)`
      - Condition
    * - ``max(min(64, N_stationary), N_moving)``
      - input data type is one of ``float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32``
    * - ``4 * max(min(64, N_stationary), N_moving)`` 
      - input data type is ``float32``

  where,

  - ``N_stationary`` is the number of elements per partition in ``stationary`` tile.
  - ``N_moving`` is the number of elements per partition in ``moving`` tile.

  :param stationary: the stationary operand on SBUF; layout: (partition axis ``<= 128``, free axis ``<= 128``)
  :param moving: the moving operand on SBUF; layout: (partition axis ``<= 128``, free axis ``<= 512``)
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param is_stationary_onezero: hints to the compiler whether the ``stationary`` operand is a tile with ones/zeros only;
                         setting this field explicitly could lead to 2x better performance
                         if ``stationary`` tile is in float32; the field has no impact for non-float32 ``stationary``.
  :param is_moving_onezero: hints to the compiler if the ``moving`` operand is a tile with ones/zeros only;
                         setting this field explicitly could lead to 2x better performance
                         if ``moving`` tile is in float32; the field has no impact for non-float32 ``moving``.
  :param is_transpose: hints to the compiler that this is a transpose operation  with ``moving`` as an identity matrix.
  :return: a tile on PSUM that has the result of matrix multiplication of ``stationary`` and ``moving`` tiles;
           layout: partition axis comes from free axis of ``stationary``, while free axis comes from free axis of ``moving``.

  Example:

  .. nki_example:: ../../test/test_nki_isa_nc_matmul.py
   :language: python
   :marker: NKI_EXAMPLE_0

  """
  ...

def nc_transpose(data, *, mask=None, dtype=None, engine=0, **kwargs):
  r"""
  Perform a 2D transpose between the partition axis and the free axis of input ``data``, i.e., a PF-transpose,
  using Tensor or Vector Engine. If the ``data`` tile has more than one free axes,
  this API implicitly collapses all free axes into one axis and then performs a 2D PF-transpose.

  In NeuronCore, both Tensor and Vector Engine can perform a PF-transpose, but they support different input shapes.
  Tensor Engine ``nc_transpose`` can handle an input tile of shape (128, 128) or smaller, while Vector
  Engine can handle shape (32, 32) or smaller.
  Therefore, when the input tile shape is (32, 32) or smaller,
  we have an option to run it on either engine, which is controlled by the
  ``engine`` field. If no ``engine`` is specified, Neuron Compiler will automatically select an engine
  based on the input shape. Note, similar to other Tensor Engine instructions, the Tensor Engine
  ``nc_transpose`` must read the input tile from SBUF and write the transposed result to PSUM. On the other hand,
  Vector Engine ``nc_transpose`` can read/write from/to either SBUF or PSUM.

  Note, PF-transpose on Tensor Engine is done by performing a matrix multiplication between ``data`` as the
  stationary tensor and an identity matrix as the moving tensor.
  See :ref:`architecture guide <arch_sec_tensor_engine_alternative_use>` for more information. On NeuronCore-v2,
  such matmul-style transpose is not bit-accurate if the input ``data`` contains NaN/Inf. You may consider replacing
  NaN/Inf with regular floats (float_max/float_min/zeros) in the input matrix before calling
  ``nc_transpose(engine=nki.isa.tensor_engine)``.


  **Estimated instruction cost:**

  .. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Cost `(Engine Cycles)`
      - Condition
    * - ``max(MIN_II, N)`` 
      - ``engine`` set to ``nki.isa.vector_engine`` 
    * - ``max(P, min(64, F))``
      - ``engine`` set to ``nki.isa.tensor_engine`` and assuming many back-to-back ``nc_transpose`` of the same shape on Tensor Engine

  where,

  - ``N`` is the number of elements per partition in ``data``.
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
    ``MIN_II`` is roughly 64 engine cycles.
  - ``P`` is partition axis size of ``data``.
  - ``F`` is the number of elements per partition in ``data``.
      
          
  :param data: the input tile to be transposed
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: if specified and it's different from the data type of input tile ``data``, an additional
                nki.isa.cast instruction will be inserted to cast the transposed data into the target ``dtype``
                (see :ref:`nki-dtype` for more information)
  :param engine: specify which engine to use for transpose: ``nki.isa.tensor_engine`` or ``nki.isa.vector_engine`` ;
                 by default, the best engine will be selected for the given input tile shape
  :return: a tile with transposed result of input ``data`` tile

  Example:

  .. nki_example:: ../../test/test_nki_isa_nc_transpose.py
   :language: python
   :marker: NKI_EXAMPLE_1

  """
  ...

from enum import IntEnum
class nc_version(IntEnum): 
  r""" NeuronCore version """

  gen2 = 2
  r"""Trn1/Inf2 target"""

  gen3 = 3
  r"""Trn2 target"""


def reciprocal(data, *, dtype=None, mask=None, engine=0, **kwargs):
  r"""
  Compute reciprocal of each element in the input ``data`` tile using Scalar Engine or Vector Engine.

  The target compute engine can be specified with the ``engine`` parameter. Vector Engine performs reciprocal
  at a higher precision compared to Scalar Engine; however, the computation throughput of reciprocal on Vector Engine
  is about 8x lower than Scalar Engine for large input tiles. For input tiles with a small number of elements
  per partition (less than 64), we suggest using Vector Engine for the better precision and comparable performance
  between Scalar and Vector Engines due to instruction initiation intervals.

  **Estimated instruction cost:**

  .. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Cost `(Engine Cycles)`
      - Condition
    * - ``max(MIN_II, N)``
      - ``engine`` set to ``nki.isa.scalar_engine`` 
    * - ``max(MIN_II, 8*N)``
      - ``engine`` set to ``nki.isa.vector_engine`` 

  where,

  - ``N`` is the number of elements per partition in ``data``.
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
    ``MIN_II`` is roughly 64 engine cycles.

  :param data: the input tile
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param engine: (optional) the engine to use for the operation: ``nki.isa.vector_engine``, ``nki.isa.scalar_engine``,
                 or ``nki.isa.unknown_engine`` (default, compiler selects best engine based on the input tile shape).
  :return: an output tile of reciprocal computation

  Example:

  .. nki_example:: ../../test/test_nki_isa_reciprocal.py
   :language: python
   :marker: NKI_EXAMPLE_6

  """
  ...

scalar_engine = ...
r"""Scalar Engine"""

def scalar_tensor_tensor(*, data, op0, operand0, op1, operand1, reverse0=False, reverse1=False, dtype=None, mask=None, **kwargs):
  r"""
  Apply up to two math operators using Vector Engine: ``(data <op0> operand0) <op1> operand1``.

  ``data`` input can be an SBUF or PSUM tile of 2D shape.
  ``operand0`` can be SBUF or PSUM tile of shape ``(data.shape[0], 1)``, i.e., vector, or a compile-time constant scalar.
  ``operand1`` can be SBUF or PSUM tile of shape ``(data.shape[0], data.shape[1])`` (i.e., has to match ``data`` shape), 
  note that ``operand1`` and ``data`` can't both be on PSUM. 

  **Estimated instruction cost:**

  .. list-table::
    :widths: 30 70
    :header-rows: 1

    * - Cost `(Vector Engine Cycles)`
      - Condition
    * - ``N``
      - ``data`` and ``operand1`` are both ``bfloat16``, ``op0=nl.subtract`` and ``op1=nl.multiply``, and ``N`` is even
    * - ``2*N`` 
      - otherwise

  where,

  - ``N`` is the number of elements per partition in ``data``.

  :param data: the input tile
  :param op0: the first math operator used with operand0 (see :ref:`nki-aluop` for supported operators)
  :param operand0: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                  is the partition axis size of the input ``data`` tile.
  :param reverse0: reverse ordering of inputs to ``op0``; if false, ``operand0`` is the rhs of ``op0``;
                   if true, ``operand0`` is the lhs of ``op0``.
  :param op1: the second math operator used with operand1 (see :ref:`nki-aluop` for supported operators).
  :param operand1: a tile of shape with the same partition and free dimension as ``data`` input.
  :param reverse1: reverse ordering of inputs to ``op1``; if false, ``operand1`` is the rhs of ``op1``;
                   if true, ``operand1`` is the lhs of ``op1``.
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :return: an output tile of ``(data <op0> operand0) <op1> operand1`` computation

  """
  ...

def tensor_copy(src, *, mask=None, dtype=None, engine=0, **kwargs):
  r"""
  Create a copy of the source tile within SBUF/PSUM using Vector, Scalar or GpSimd Engine.

  The output tile has the same partition axis size and also the same number of elements per partition
  as the input tile ``src``.

  All three compute engines, Vector, Scalar and GpSimd Engine can perform tensor copy. However, their copy behavior
  is slightly different across engines:

  - Scalar Engine on NeuronCore-v2 performs copy by first casting the input tile to FP32 internally and then casting from
    FP32 to the output dtype (``dtype``, or src.dtype if ``dtype`` is not specified). Therefore, users should be
    cautious with assigning this instruction to Scalar Engine when the input data type cannot be precisely cast to FP32
    (e.g., INT32).
  - Both GpSimd and Vector Engine can operate in two modes: (1) bit-accurate copy when input and output data types are
    the same or (2) intermediate FP32 cast when input and output data types differ, similar to Scalar Engine.

  In addition, since GpSimd Engine cannot access PSUM in NeuronCore, Scalar or Vector Engine must be chosen when the input or
  output tile is in PSUM (see :ref:`arch_sec_neuron_core_engines` for details). By default, this API returns
  a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.

  **Estimated instruction cost:**

  ``max(MIN_II, N)`` engine cycles, where ``N`` is the number of elements per partition in the input tile,
  and ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
  ``MIN_II`` is roughly 64 engine cycles.

  :param src: the source of copy, must be a tile in SBUF or PSUM.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param engine: (optional) the engine to use for the operation: `nki.isa.vector_engine`, `nki.isa.scalar_engine`,
                `nki.isa.gpsimd_engine` or `nki.isa.unknown_engine` (default, compiler selects best engine based on engine workload).
  :return: a tile with the same content and partition axis size as the ``src`` tile.

  Example:

  .. nki_example:: ../../test/test_nki_isa_tensor_copy.py
   :language: python
   :marker: NKI_EXAMPLE_7

  """
  ...

def tensor_copy_dynamic_src(src, *, mask=None, dtype=None, **kwargs):
  r"""
  Copy a source tile in SBUF, with a dynamic offset along the free 
  dimension (except for the partition dimension) using Vector, Scalar or GpSimd
  Engine.

  Consider combining with ``nl.mgrid`` to specify a static grid and use an
  index tensor (containing offsets) to specify dynamic offsets. The source tile 
  ``src`` must be in SBUF / PSUM. Index tensor which specifies the offsets must 
  be in SBUF memory.

  In addition, since GpSimd Engine cannot access PSUM in NeuronCore, Scalar or 
  Vector Engine must be chosen when the input or output tile is in PSUM 
  (see :ref:`arch_sec_neuron_core_engines` for details). By default, this API returns
  a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.

  **Estimated instruction cost:**

  Each index element specifying the offset along the free dimension will trigger 
  a tensor_copy instruction. In addition, since the index is dynamic, an overhead
  of approximately 140 cycles is incurred to read index elements from SBUF / PSUM 
  into the engine.

  :param src: the source of copy, must be a tile in SBUF or PSUM.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.

  Example:

  .. nki_example:: ../../test/test_nki_isa_tensor_copy_dynamic.py
   :language: python
   :marker: NKI_EXAMPLE_0

  .. nki_example:: ../../test/test_nki_isa_tensor_copy_dynamic.py
   :language: python
   :marker: NKI_EXAMPLE_1

  """
  ...

tensor_engine = ...
r"""Tensor Engine"""

def tensor_partition_reduce(op, data, *, mask=None, dtype=None, **kwargs):
  r"""
  Apply a reduction operation across partitions of an input ``data`` tile using GpSimd Engine.

  :param op: the reduction operator (add, max, bitwise_or, bitwise_and)
  :param data: the input tile to be reduced
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :return: output tile with reduced result

  Example:

  .. nki_example:: ../../test/test_nki_isa_partition_reduce.py
   :language: python
   :marker: NKI_EXAMPLE_1

  """
  ...

def tensor_reduce(op, data, axis, *, mask=None, dtype=None, negate=False, keepdims=False, **kwargs):
  r"""
  Apply a reduction operation to the free axes of an input ``data`` tile using Vector Engine.

  The reduction operator is specified in the ``op`` input field
  (see :ref:`nki-aluop` for a list of supported reduction operators).
  There are two types of reduction operators: 1) bitvec operators (e.g., bitwise_and, bitwise_or)
  and 2) arithmetic operators (e.g., add, subtract, multiply). For bitvec
  operators, the input/output data types must be integer types and Vector Engine treats
  all input elements as bit patterns without any data type casting. For arithmetic operators, there is no
  restriction on the input/output data types, but the engine automatically casts input data types to float32
  and performs the reduction operation in float32 math. The float32 reduction results are cast to the target
  data type specified in the ``dtype`` field before written into the output tile. If the ``dtype`` field is not
  specified, it is default to be the same as input tile data type.

  When the reduction ``op`` is an arithmetic operator, the instruction can also multiply the output reduction
  results by ``-1.0`` before writing into the output tile, at no additional performance cost. This behavior is
  controlled by the ``negate`` input field.

  The reduction axes are specified in the ``axis`` field using a list of integer(s) to indicate axis indices.
  The reduction axes can contain up to four free axes and must start at the most minor free axis.
  Since axis 0 is the partition axis in a tile, the reduction axes must contain axis 1 (most-minor). In addition,
  the reduction axes must be consecutive: e.g., [1, 2, 3, 4] is a legal ``axis`` field, but [1, 3, 4] is not.

  Since this instruction only supports free axes reduction, the output tile must have the same partition
  axis size as the input ``data`` tile. To perform a partition axis reduction, we can either:

  1. invoke a ``nki.isa.nc_transpose`` instruction on the input tile and then this ``reduce`` instruction
     to the transposed tile, or
  2. invoke ``nki.isa.nc_matmul`` instructions to multiply a ``nki.language.ones([128, 1], dtype=data.dtype)``
     vector with the input tile.

  **Estimated instruction cost:**

  .. list-table::
    :widths: 30 70
    :header-rows: 1

    * - Cost `(Vector Engine Cycles)`
      - Condition
    * - ``N/2`` 
      - both input and output data types are ``bfloat16`` *and* the reduction operator is add or maximum
    * - ``N``
      - otherwise

  where,

  - ``N`` is the number of elements per partition in ``data``.
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
    ``MIN_II`` is roughly 64 engine cycles.


  :param op: the reduction operator (see :ref:`nki-aluop` for supported reduction operators)
  :param data: the input tile to be reduced
  :param axis: int or tuple/list of ints. The axis (or axes) along which to operate; must be free dimensions, not partition dimension (0); can only be the last contiguous dim(s) of the tile: ``[1], [1,2], [1,2,3], [1,2,3,4]``
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param negate: if True, reduction result is multiplied by ``-1.0``;
                 only applicable when op is an arithmetic operator
  :param keepdims: If this is set to True, the axes which are reduced are left in the result as dimensions with size one.
                   With this option, the result will broadcast correctly against the input array.
  :return: output tile of the reduction result

  Example:

  .. nki_example:: ../../test/test_nki_isa_reduce.py
   :language: python
   :marker: NKI_EXAMPLE_2

  """
  ...

def tensor_scalar(data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, *, dtype=None, mask=None, engine=0, **kwargs):
  r"""
  Apply up to two math operators to the input ``data`` tile by broadcasting scalar/vector operands
  in the free dimension using Vector or Scalar Engine: ``(data <op0> operand0) <op1> operand1``.

  The input ``data`` tile can be an SBUF or PSUM tile. Both ``operand0`` and ``operand1`` can be
  SBUF or PSUM tiles of shape ``(data.shape[0], 1)``, i.e., vectors,
  or compile-time constant scalars.

  ``op1`` and ``operand1`` are optional, but must be ``None`` (default values) when unused.
  Note, performing one operator has the same performance cost as performing two operators in the instruction.

  When the operators are non-commutative (e.g., subtract), we can reverse ordering of the inputs for each operator through:

    - ``reverse0 = True``: ``tmp_res = operand0 <op0> data``
    - ``reverse1 = True``: ``operand1 <op1> tmp_res``

  The ``tensor_scalar`` instruction supports two types of operators: 1) bitvec
  operators (e.g., bitwise_and) and 2) arithmetic operators (e.g., add).
  See :ref:`nki-aluop` for the full list of supported operators.
  The two operators, ``op0`` and ``op1``, in a ``tensor_scalar`` instruction must be of the same type
  (both bitvec or both arithmetic).
  If bitvec operators are used, the ``tensor_scalar`` instruction must run on Vector Engine. Also, the input/output
  data types must be integer types, and input elements are treated as bit patterns without any data type casting.

  If arithmetic operators are used, the ``tensor_scalar`` instruction can run on Vector or Scalar Engine.
  However, the Scalar Engine only supports a subset of the operator combination:

    - ``op0=np.multiply`` and ``op1=np.add``
    - ``op0=np.multiply`` and ``op1=None``
    - ``op0=add`` and ``op1=None``

  Also, arithmetic operators impose no restriction on the input/output data types,
  but the engine automatically casts input data types to float32
  and performs the operators in float32 math. The float32 computation results are cast to the target
  data type specified in the ``dtype`` field before written into the output tile, at no additional performance cost.
  If the ``dtype`` field is not specified, it is default to be the same as input tile data type.

  **Estimated instruction cost:**

  ``max(MIN_II, N)`` Vector or Scalar Engine cycles, where

  - ``N`` is the number of elements per partition in ``data``.
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles. 
    ``MIN_II`` is roughly 64 engine cycles.

  :param data: the input tile
  :param op0: the first math operator used with operand0 (see :ref:`nki-aluop` for supported operators)
  :param operand0: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                  is the partition axis size of the input ``data`` tile
  :param reverse0: reverse ordering of inputs to ``op0``; if false, ``operand0`` is the rhs of ``op0``;
                   if true, ``operand0`` is the lhs of ``op0``
  :param op1: the second math operator used with operand1 (see :ref:`nki-aluop` for supported operators);
              this operator is optional
  :param operand1: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                  is the partition axis size of the input ``data`` tile
  :param reverse1: reverse ordering of inputs to ``op1``; if false, ``operand1`` is the rhs of ``op1``;
                   if true, ``operand1`` is the lhs of ``op1``
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param engine: (optional) the engine to use for the operation: `nki.isa.vector_engine`, `nki.isa.scalar_engine`,
                 or `nki.isa.unknown_engine` (default, lets compiler select best engine based on the input tile shape).
  :return: an output tile of ``(data <op0> operand0) <op1> operand1`` computation

  Example:

  .. nki_example:: ../../test/test_nki_isa_tensor_scalar.py
   :language: python
   :marker: NKI_EXAMPLE_5


  """
  ...

def tensor_scalar_reduce(*, data, op0, operand0, reduce_op, reduce_res, reverse0=False, dtype=None, mask=None, **kwargs):
  r"""
  Perform the same computation as ``nisa.tensor_scalar`` with one math operator
  and also a reduction along the free dimension of the ``nisa.tensor_scalar`` result using Vector Engine.

  Refer to :doc:`nisa.tensor_scalar <nki.isa.tensor_scalar>` for semantics of ``data/op0/operand0``.
  Unlike regular ``nisa.tensor_scalar`` where two operators are supported, only one
  operator is supported in this API. Also, ``op0`` can only be arithmetic operation in :ref:`nki-aluop`.
  Bitvec operators are not supported in this API.

  In addition to :doc:`nisa.tensor_scalar <nki.isa.activation>` computation, this API also performs a reduction
  along the free dimension(s) of the :doc:`nisa.tensor_scalar <nki.isa.activation>` result, at a small additional
  performance cost. The reduction result is returned in ``reduce_res`` in-place, which must be a
  SBUF/PSUM tile with the same partition axis size as the input tile ``data`` and one element per partition.
  The ``reduce_op`` can be any of ``nl.add``, ``nl.subtract``, ``nl.multiply``, ``nl.max`` or ``nl.min``.

  Reduction axis is not configurable in this API. If the input tile has multiple free axis, the API will
  reduce across all of them.

  .. math::
    result = data <op0> operand0 \\
    reduce\_res = reduce\_op(dst, axis=<FreeAxis>)

  **Estimated instruction cost:**

  ``max(MIN_II, N) + MIN_II`` Vector Engine cycles, where

  - ``N`` is the number of elements per partition in ``data``, and
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
    ``MIN_II`` is roughly 64 engine cycles.

  :param data: the input tile
  :param op0: the math operator used with operand0 (any arithmetic operator in :ref:`nki-aluop` is allowed)
  :param operand0: a scalar constant or a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                  is the partition axis size of the input ``data`` tile
  :param reverse0: `(coming soon)` reverse ordering of inputs to ``op0``; if false, ``operand0`` is the rhs of ``op0``;
                   if true, ``operand0`` is the lhs of ``op0``. `<-- currently not supported yet.`
  :param reduce_op: the reduce operation to perform on the free dimension of ``data <op0> operand0``
  :param reduce_res: a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                  is the partition axis size of the input ``data`` tile. The result of ``reduce_op(data <op0> operand0)``
                  is written in-place into the tile.
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :return: an output tile of ``(data <op0> operand0)`` computation
  """
  ...

def tensor_tensor(data1, data2, op, *, dtype=None, mask=None, **kwargs):
  r"""
  Perform an element-wise operation of input two tiles using Vector Engine or GpSimd Engine. 
  The two tiles must have the same partition axis size and the same number of elements per partition.

  The two input tiles, ``data1`` and ``data2`` cannot both reside in PSUM. The three legal cases are:

  1. Both ``data1`` and ``data2`` are in SBUF.
  2. ``data1`` is in SBUF, while ``data2`` is in PSUM.
  3. ``data1`` is in PSUM, while ``data2`` is in SBUF.

  The output tile can be in either SBUF or PSUM.

  The element-wise operator is specified using the ``op`` field and can be any *binary* operator
  supported by NKI (see :ref:`nki-aluop` for details) that runs on the Vector Engine, 
  or can be ``np.power``/``nl.power`` (a special case that runs on the GpSimd Engine).
  For bitvec operators, the input/output data types must be integer types and Vector Engine treats
  all input elements as bit patterns without any data type casting. For arithmetic operators, there is no
  restriction on the input/output data types, but the engine automatically casts input data types to float32
  and performs the element-wise operation in float32 math. The float32 results are cast to the target
  data type specified in the ``dtype`` field before written into the
  output tile. If the ``dtype`` field is not specified, it is default to be the same as the data type of ``data1``
  or ``data2``, whichever has the highest precision.

  Note, if you need broadcasting capability in the free dimension for either input tile, you should consider
  using :doc:`nki.isa.tensor_scalar <nki.isa.tensor_scalar>` API instead,
  which has better performance than ``nki.isa.tensor_tensor`` in general.

  **Estimated instruction cost:**

  .. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Cost `(Vector Engine Cycles)`
      - Condition
    * - ``max(MIN_II, N)`` 
      - one input tile is in PSUM and the other is in SBUF
    * - ``max(MIN_II, N)`` 
      - all of the below:

        * both input tiles are in SBUF,
        * input/output data types are all ``bfloat16``,
        * the operator is add, multiply or subtract,
        * Input tensor data is contiguous along the free dimension (that is, stride in each partition is 1 element)
    * - ``max(MIN_II, 2N)`` 
      - otherwise

  where,

  - ``N`` is the number of elements per partition in ``data1``/``data2``.
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles. 
    ``MIN_II`` is roughly 64 engine cycles.


  :param data1: lhs input operand of the element-wise operation
  :param data2: rhs input operand of the element-wise operation
  :param op: a binary math operator (see :ref:`nki-aluop` for supported operators)
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
  :return: an output tile of the element-wise operation

  Example:

  .. nki_example:: ../../test/test_nki_isa_tensor_tensor.py
   :language: python
   :marker: NKI_EXAMPLE_3

  """
  ...

def tensor_tensor_scan(data0, data1, initial, op0, op1, reverse0=False, reverse1=False, *, dtype=None, mask=None, **kwargs):
  r"""
  Perform a scan operation of two input tiles using Vector Engine.

  Mathematically, the tensor_tensor_scan instruction on Vector Engine performs
  the following computation per partition:

  .. code-block:: python

      # Let's assume we work with numpy, and data0 and data1 are 2D (with shape[0] being the partition axis)
      import numpy as np

      result = np.ndarray(data0.shape, dtype=data0.dtype)
      result[:, 0] = op1(op0(data0[:. 0], initial), data1[:, 0])

      for i in range(1, data0.shape[1]):
          result[:, i] = op1(op0(data0[:, i], result[:, i-1]), data1[:, i])

  The two input tiles (``data0`` and ``data1``) must have the same
  partition axis size and the same number of elements per partition.
  The third input ``initial`` can either be a float32 compile-time scalar constant
  that will be broadcasted in the partition axis of ``data0``/``data1``, or a tile
  with the same partition axis size as ``data0``/``data1`` and one element per partition.

  The two input tiles, ``data0`` and ``data1`` cannot both reside in PSUM. The three legal cases are:

  1. Both ``data1`` and ``data2`` are in SBUF.
  2. ``data1`` is in SBUF, while ``data2`` is in PSUM.
  3. ``data1`` is in PSUM, while ``data2`` is in SBUF.

  The scan operation supported by this API has two programmable
  math operators in ``op0`` and ``op1`` fields.
  Both ``op0`` and ``op1`` can be any binary arithmetic operator
  supported by NKI (see :ref:`nki-aluop` for details).
  We can optionally reverse the input operands of ``op0`` by setting ``reverse0`` to True
  (or ``op1`` by setting ``reverse1``). Reversing operands is useful for non-commutative
  operators, such as subtract.

  Input/output data types can be any supported NKI data type (see :ref:`nki-dtype`),
  but the engine automatically casts input data types to float32
  and performs the computation in float32 math. The float32 results are cast to the target
  data type specified in the ``dtype`` field before written into the
  output tile. If the ``dtype`` field is not specified, it is default to be the
  same as the data type of ``data0``
  or ``data1``, whichever has the highest precision.

  **Estimated instruction cost:**
  
  ``max(MIN_II, 2N)`` Vector Engine cycles, where

  - ``N`` is the number of elements per partition in ``data0``/``data1``.
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles. 
    ``MIN_II`` is roughly 64 engine cycles.
  
  :param data0: lhs input operand of the scan operation
  :param data1: rhs input operand of the scan operation
  :param initial: starting state of the scan; can be a SBUF/PSUM tile with 1 element/partition or a scalar
                      compile-time constant
  :param op0: a binary arithmetic math operator (see :ref:`nki-aluop` for supported operators)
  :param op1: a binary arithmetic math operator (see :ref:`nki-aluop` for supported operators)
  :param reverse0: reverse ordering of inputs to ``op0``; if false, ``data0`` is the lhs of ``op0``;
                 if true, ``data0`` is the rhs of ``op0``
  :param reverse1: reverse ordering of inputs to ``op1``; if false, ``data1`` is the rhs of ``op1``;
                 if true, ``data1`` is the lhs of ``op1``
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tiles, or whichever input type has the highest precision (see :ref:`nki-type-promotion` for more information);
  :return: an output tile of the scan operation

  Example:

  .. nki_example:: ../../test/test_nki_isa_tensor_tensor_scan.py
   :language: python
   :marker: NKI_EXAMPLE_4

  """
  ...

unknown_engine = ...
r"""Unknown Engine"""

vector_engine = ...
r"""Vector Engine"""

