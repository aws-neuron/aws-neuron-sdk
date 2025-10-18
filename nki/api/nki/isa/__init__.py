import numpy as np
import ml_dtypes

from enum import IntEnum
class dge_mode(IntEnum): 
  r""" Neuron Descriptor Generation Engine Mode """

  none = 0
  r"""Not using DGE"""

  swdge = 1
  r"""Software DGE"""

  hwdge = 2
  r"""Hardware DGE"""

  unknown = 3
  r"""Unknown DGE mode, i.e., let compiler decide the DGE mode"""


dma_engine = ...
r"""DMA Engine"""

from enum import IntEnum
class engine(IntEnum): 
  r""" Neuron Device engines """

  tensor = 1
  r"""Tensor Engine"""

  vector = 5
  r"""Vector Engine"""

  scalar = 2
  r"""Scalar Engine"""

  gpsimd = 3
  r"""GpSIMD Engine"""

  sync = 6
  r"""Sync Engine"""

  unknown = 0
  r"""Unknown Engine"""


class fp32: 
  r""" FP32 Constants"""

  @property
  def min(self):
    r"""FP32 Bit pattern (0xff7fffff) representing the minimum (or maximum negative) FP32 value"""
    ...

gpsimd_engine = ...
r"""GpSIMD Engine"""

from enum import IntEnum
class nc_version(IntEnum): 
  r""" NeuronCore version """

  gen2 = 2
  r"""Trn1/Inf2 target"""

  gen3 = 3
  r"""Trn2 target"""


from enum import IntEnum
class oob_mode(IntEnum): 
  r""" Neuron OOB Access Mode """

  error = 0

  skip = 1


from enum import IntEnum
class reduce_cmd(IntEnum): 
  r"""Engine Register Reduce commands """

  idle = 0
  r"""Not using the accumulator registers"""

  reset = 1
  r"""Resets the accumulator registers to its initial state"""

  reset_reduce = 3
  r""" Resets the accumulator registers then immediately accumulate the results of the current instruction into the accumulators"""

  reduce = 2
  r"""keeps accumulating over the current value of the accumulator registers"""


scalar_engine = ...
r"""Scalar Engine"""

tensor_engine = ...
r"""Tensor Engine"""

unknown_engine = ...
r"""Unknown Engine"""

vector_engine = ...
r"""Vector Engine"""

def activation(op, data, *, bias=None, scale=1.0, reduce_op=None, reduce_res=None, reduce_cmd=reduce_cmd.idle, mask=None, dtype=None, **kwargs):
  r"""
  Apply an activation function on every element of the input tile using Scalar Engine. The activation
  function is specified in the ``op`` input field (see :ref:`nki-act-func` for a list of
  supported activation functions and their valid input ranges).

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

  There are 128 registers on the scalar engine for storing reduction results, corresponding
  to the 128 partitions of the input.
  The scalar engine can reduce along free dimensions without extra performance penalty,
  and store the result of reduction into these registers. The reduction is done after the activation function
  is applied.

  .. math::
        output = f_{act}(data * scale + bias)
        accu\_registers = reduce\_op(accu\_registers, reduce\_op(output, axis=<FreeAxis>))

  These registers are shared between ``activation`` and ``activation_accu`` calls, and the state of them can be
  controlled via the ``reduce_cmd`` parameter.

  - ``nisa.reduce_cmd.reset``: Reset the accumulators to zero
  - ``nisa.reduce_cmd.idle``: Do not use the accumulators
  - ``nisa.reduce_cmd.reduce``: keeps accumulating over the current value of the accumulator
  - ``nisa.reduce_cmd.reset_reduce``: Resets the accumulators then immediately accumulate the results of the current instruction into the accumulators

  We can choose to read out the current values stored in the
  register by passing in a tensor in the ``reduce_res`` arguments. Reading out the accumulator will
  incur a small overhead.

  Note that ``activation_accu`` can also change the state of the registers. It's user's responsibility
  to ensure correct ordering. It's recommended to not mixing the use of ``activation_accu`` and ``activation``,
  when ``reduce_cmd`` is not set to idle.

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
  :param reduce_op: the reduce operation to perform on the free dimension of the activation result
  :param reduce_res: a tile of shape ``(data.shape[0], 1)``, where data.shape[0]
                  is the partition axis size of the input ``data`` tile. The result of ``sum(ReductionResult)``
                  is written in-place into the tensor.
  :param reduce_cmd: an enum member from ``nisa.reduce_cmd`` to control the state of reduction registers
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
  ``nisa.activation`` result using Scalar Engine. The results for the reduction is stored
  in the reduce_res.

  This API is equivalent to calling ``nisa.activation`` with
  ``reduce_cmd=nisa.reduce_cmd.reset_reduce`` and passing in reduce_res. This API is kept for
  backward compatibility, we recommend using ``nisa.activation`` moving forward.

  Refer to :doc:`nisa.activation <nki.isa.activation>` for semantics of ``op/data/bias/scale``.

  In addition to :doc:`nisa.activation <nki.isa.activation>` computation, this API also performs a reduction
  along the free dimension(s) of the :doc:`nisa.activation <nki.isa.activation>` result, at a small additional
  performance cost. The reduction result is returned in ``reduce_res`` in-place, which must be a
  SBUF/PSUM tile with the same partition axis size as the input tile ``data`` and one element per partition.
  On NeuronCore-v2, the ``reduce_op`` can only be an addition, ``np.add`` or ``nl.add``.

  There are 128 registers on the scalar engine for storing reduction results, corresponding
  to the 128 partitions of the input. These registers are shared between ``activation`` and ``activation_accu`` calls.
  This instruction first resets those
  registers to zero, performs the reduction on the value after activation function is applied,
  stores the results into the registers,
  then reads out the reduction results from the register, eventually store them into ``reduce_res``.

  Note that ``nisa.activation`` can also change the state of the register. It's user's
  responsibility to ensure correct ordering. It's the best practice to not mixing
  the use of ``activation_reduce`` and ``activation``.

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
  calculated on-the-fly in the engine by evaluating an affine expression element-by-element as indicated in ``pred``.

  ``pred`` must meet the following requirements:

    - It must not depend on any runtime variables that can't be resolved at compile-time.
    - It can't be multiple masks combined using logical operators such as ``&`` and ``|``.

  For a complex predicate that doesn't meet the above requirements, consider using :doc:`nl.where <nki.language.where>`.

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

  ``GPSIMD_START + N`` GpSimd Engine cycles, where ``N`` is the number of elements per partition in ``on_true_tile`` and
  ``GPSIMD_START`` is the instruction startup overhead on GpSimdE, roughly 150 engine cycles.

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

def dma_copy(*, dst, src, mask=None, dst_rmw_op=None, oob_mode=oob_mode.error, dge_mode=dge_mode.unknown):
  r"""
  Copy data from ``src`` to ``dst`` using DMA engine. Both ``src`` and ``dst`` tiles can be in device memory (HBM) or SBUF.
  However, if both ``src`` and ``dst`` tiles are in SBUF, consider using
  :doc:`nisa.tensor_copy <nki.isa.tensor_copy>` instead for better performance.

  :param src: the source of copy.
  :param dst: the dst of copy.
  :param dst_rmw_op: the read-modify-write operation to be performed at the destination.
                     Currently only ``np.add`` is supported, which adds the source data to the existing destination data.
                     If ``None``, the source data directly overwrites the destination.
                     If ``dst_rmw_op`` is specified, only ``oob_mode=oob_mode.error`` is allowed.
                     For best performance with Descriptor Generation Engine (DGE), unique dynamic offsets
                     must be used to access ``dst``. Multiple accesses to the same offset will cause a data hazard.
                     If duplicated offsets are present, the compiler automatically adds synchronization to avoid
                     hazards, which slows down computation.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param mode: (optional) Specifies how to handle out-of-bounds (oob) array indices during indirect access operations. Valid modes are:

        - ``oob_mode.error``: (Default) Raises an error when encountering out-of-bounds indices.
        - ``oob_mode.skip``: Silently skips any operations involving out-of-bounds indices.

        For example, when using indirect gather/scatter operations, out-of-bounds indices can occur if the index array contains values that exceed the dimensions of the target array.
  :param dge_mode: (optional) specify which Descriptor Generation Engine (DGE) mode to use for copy: ``nki.isa.dge_mode.none`` (turn off DGE) or ``nki.isa.dge_mode.swdge`` (software DGE) or ``nki.isa.dge_mode.hwdge`` (hardware DGE)  or ``nki.isa.dge_mode.unknown`` (by default, let compiler select the best DGE mode). HWDGE is only supported for NeuronCore-v3+.

  A cast will happen if the ``src`` and ``dst`` have different dtype.

  Example:

  .. nki_example:: ../../test/test_nki_isa_dma_copy.py
   :language: python
   :marker: NKI_EXAMPLE_0

  .. nki_example:: ../../test/test_nki_isa_dma_copy.py
   :language: python
   :marker: NKI_EXAMPLE_1

  .. nki_example:: ../../test/test_nki_isa_dma_copy.py
   :language: python
   :marker: NKI_EXAMPLE_2

  .. nki_example:: ../../test/test_nki_isa_dma_copy.py
   :language: python
   :marker: NKI_EXAMPLE_3

  .. nki_example:: ../../test/test_nki_isa_dma_copy.py
   :language: python
   :marker: NKI_EXAMPLE_4

  .. nki_example:: ../../test/test_nki_isa_dma_copy.py
   :language: python
   :marker: NKI_EXAMPLE_5

  .. nki_example:: ../../test/test_nki_isa_dma_copy.py
   :language: python
   :marker: NKI_EXAMPLE_6

  .. nki_example:: ../../test/test_nki_isa_dma_copy.py
   :language: python
   :marker: NKI_EXAMPLE_7
  """
  ...

def dma_transpose(src, *, axes=None, mask=None, dtype=None, **kwargs):
  r"""
  Perform a transpose on input ``src`` using DMA Engine.

  The permutation of transpose follow the rules described below:

  1. For 2-d input tile, the permutation will be [1, 0]
  2. For 3-d input tile, the permutation will be [2, 1, 0]
  3. For 4-d input tile, the permutation will be [3, 1, 2, 0]

  :param src: the source of transpose, must be a tile in HBM or SBUF.
  :param axes: transpose axes where the i-th axis of the transposed tile will correspond to the axes[i] of the source.
               Supported axes are ``(1, 0)``, ``(2, 1, 0)``, and ``(3, 1, 2, 0)``.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param dge_mode: (optional) specify which Descriptor Generation Engine (DGE) mode to use for copy: ``nki.isa.dge_mode.none`` (turn off DGE) or ``nki.isa.dge_mode.swdge`` (software DGE) or ``nki.isa.dge_mode.hwdge`` (hardware DGE)  or ``nki.isa.dge_mode.unknown`` (by default, let compiler select the best DGE mode). HWDGE is only supported for NeuronCore-v3+.
  :return: a tile with transposed content

  Example:

  .. nki_example:: ../../test/test_nki_isa_dma_transpose.py
   :language: python
   :marker: NKI_EXAMPLE_0

  .. nki_example:: ../../test/test_nki_isa_dma_transpose.py
   :language: python
   :marker: NKI_EXAMPLE_1

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

def max8(*, src, mask=None, dtype=None, **kwargs):
  r"""
  Find the 8 largest values in each partition of the source tile.

  This instruction reads the input elements, converts them to fp32 internally, and outputs
  the 8 largest values in descending order for each partition. By default, returns the
  same dtype as the input tensor.

  The source tile can be up to 5-dimensional, while the output tile is always 2-dimensional.
  The number of elements read per partition must be between 8 and 16,384 inclusive.
  The output will always contain exactly 8 elements per partition.
  The source and output must have the same partition dimension size:

  - source: [par_dim, ...]
  - output: [par_dim, 8]

  **Estimated instruction cost:**

  ``N`` engine cycles, where:

  - ``N`` is the number of elements per partition in the source tile

  :param src: the source tile to find maximum values from
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :return: a 2D tile containing the 8 largest values per partition in descending order with shape [par_dim, 8]

  Example:

  .. nki_example:: ../../test/test_nki_isa_max8.py
   :language: python
   :marker: NKI_EXAMPLE_0

  """
  ...

def memset(shape, value, dtype, *, mask=None, engine=engine.unknown, **kwargs):
  r"""
  Initialize a tile filled with a compile-time constant value using Vector or GpSimd Engine.
  The shape of the tile is specified in the ``shape`` field and the
  initialized value in the ``value`` field.
  The memset instruction supports all valid NKI dtypes
  (see :ref:`nki-dtype`).

  :param shape: the shape of the output tile; layout: (partition axis, free axis). Note that memset
                ignores nl.par_dim() and always treats the first dimension as the partition dimension.
  :param value: the constant value to initialize with
  :param dtype: data type of the output tile (see :ref:`nki-dtype` for more information)
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param engine: specify which engine to use for memset: ``nki.isa.vector_engine`` or ``nki.isa.gpsimd_engine`` ;
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

def nc_find_index8(*, data, vals, mask=None, dtype=None, **kwargs):
  r"""
  Find indices of the 8 given vals in each partition of the data tensor.

  This instruction first loads the 8 values,
  then loads the data tensor and outputs the indices (starting at 0) of the first
  occurrence of each value in the data tensor, for each partition.

  The data tensor can be up to 5-dimensional, while the vals tensor must be up
  to 3-dimensional. The data tensor must have between 8 and 16,384 elements per
  partition. The vals tensor must have exactly 8 elements per partition.
  The output will contain exactly 8 elements per partition and will be uint16 or
  uint32 type. Default output type is uint32.

  Behavior is undefined if vals tensor contains values that are not in
  the data tensor.

  If provided, a mask is applied only to the data tensor.

  **Estimated instruction cost:**

  ``N`` engine cycles, where:

  - ``N`` is the number of elements per partition in the data tensor

  :param data: the data tensor to find indices from
  :param vals: tensor containing the 8 values per partition whose indices will be found
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: uint16 or uint32
  :return: a 2D tile containing indices (uint16 or uint32) of the 8 values in each partition with shape [par_dim, 8]

  Example:

  .. nki_example:: ../../test/test_nki_isa_nc_find_index8.py
     :language: python
     :marker: NKI_EXAMPLE_0

  """
  ...

def nc_match_replace8(*, data, vals, imm, dst_idx=None, mask=None, dtype=None, **kwargs):
  r"""
  Replace first occurrence of each value in ``vals`` with ``imm`` in ``data``
  using the Vector engine and return the replaced tensor. If ``dst_idx``
  tile is provided, the indices of the matched values are written to ``dst_idx``.

  This instruction reads the input ``data``, replaces the first occurrence of each
  of the given values (from ``vals`` tensor) with the specified immediate constant and,
  optionally, output indices of matched values to ``dst_idx``. When performing the operation,
  the free dimensions of both ``data`` and ``vals`` are flattened. However, these dimensions
  are preserved in the replaced output tensor and in ``dst_idx`` respectively. The partition
  dimension defines the parallelization boundary. Match, replace, and index
  generation operations execute independently within each partition.

  The ``data`` tensor can be up to 5-dimensional, while the ``vals`` tensor can be up
  to 3-dimensional. The ``vals`` tensor must have exactly 8 elements per partition.
  The data tensor must have no more than 16,384 elements per partition.
  The replaced output will have the same shape as the input data tensor. ``data`` and ``vals``
  must have the same number of partitions. Both input tensors can come from SBUF
  or PSUM.

  Behavior is undefined if vals tensor contains values that are not in the data
  tensor.

  If provided, a mask is applied to the data tensor.

  **Estimated instruction cost:**

  ``min(MIN_II, N)`` engine cycles, where:

  - ``N`` is the number of elements per partition in the data tensor
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
    ``MIN_II`` is roughly 64 engine cycles.

  **NumPy equivalent:**

  .. code-block:: python

      # Let's assume we work with NumPy, and ``data``, ``vals`` are 2-dimensional arrays
      # (with shape[0] being the partition axis) and imm is a constant float32 value.

      import numpy as np

      # Get original shapes
      data_shape = data.shape
      vals_shape = vals.shape

      # Reshape to 2D while preserving first dimension
      data_2d = data.reshape(data_shape[0], -1)
      vals_2d = vals.reshape(vals_shape[0], -1)

      # Initialize output array for indices
      indices = np.zeros(vals_2d.shape, dtype=np.uint32)

      for i in range(data_2d.shape[0]):
        for j in range(vals_2d.shape[1]):
          val = vals_2d[i, j]
          # Find first occurrence of val in data_2d[i, :]
          matches = np.where(data_2d[i, :] == val)[0]
          if matches.size > 0:
            indices[i, j] = matches[0]  # Take first match
            data_2d[i, matches[0]] = imm

      output = data_2d.reshape(data.shape)
      indices = indices.reshape(vals.shape) # Computed only if ``dst_idx`` is specified

  :param data: the data tensor to modify
  :param dst_idx: (optional) the destination tile to write flattened indices of matched values
  :param vals: tensor containing the 8 values per partition to replace
  :param imm: float32 constant to replace matched values with
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :return: the modified data tensor

  Example:

  .. nki_example:: ../../test/test_nki_isa_nc_match_replace8.py
     :language: python
     :marker: NKI_EXAMPLE_0

  .. nki_example:: ../../test/test_nki_isa_nc_match_replace8.py
     :language: python
     :marker: NKI_EXAMPLE_1

  .. nki_example:: ../../test/test_nki_isa_nc_match_replace8.py
     :language: python
     :marker: NKI_EXAMPLE_2

  .. nki_example:: ../../test/test_nki_isa_nc_match_replace8.py
     :language: python
     :marker: NKI_EXAMPLE_3

  .. nki_example:: ../../test/test_nki_isa_nc_match_replace8.py
     :language: python
     :marker: NKI_EXAMPLE_4

  """
  ...

def nc_matmul(stationary, moving, *, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False, tile_position=(), tile_size=(), mask=None, **kwargs):
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

  The Tensor Engine, as a systolic array with 128 rows and 128 columns of processing elements (PEs), could be underutilized
  for small ``nc_matmul`` instructions, i.e., the ``stationary`` tile has small free axis size or small partition axis size
  (e.g. 32, 64). In such a case, the Tensor Engine allows PE tiling, i.e., multiple small ``nc_matmul`` instructions to execute
  in parallel on the PE array, to improve compute throughput. PE tiling is enabled by setting ``tile_position`` and ``tile_size``.
  ``tile_position`` indicates the PE tile starting position (row position, column position) for a ``nc_matmul`` instruction in
  the PE array. ``tile_size`` indicates the PE tile size (row size, column size) to hold by a ``nc_matmul`` instruction starting
  from the ``tile_position``. For example, setting ``tile_position`` to (0, 0) and ``tile_size`` to (128, 128) means using full
  PE array.

  Requirements on ``tile_position`` and ``tile_size`` are:

  #. ``tile_position`` and ``tile_size`` must be both set to enable PE tiling.
  #. The type of values in ``tile_position`` and ``tile_size`` must be integer or affine expression.
  #. Values in ``tile_position`` and ``tile_size`` must be multiple of 32.
  #. ``tile_size`` must be larger than or equal to accessed ``stationary`` tile size.
  #. Both the row and column sizes in ``tile_size`` cannot be 32 for NeuronCore-v2.

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
  :param tile_position: a 2D tuple (row, column) for the start PE tile position to run ``nc_matmul``.
  :param tile_size: a 2D tuple (row, column) for the PE tile size to hold by ``nc_matmul`` starting from ``tile_position``.
  :return: a tile on PSUM that has the result of matrix multiplication of ``stationary`` and ``moving`` tiles;
           layout: partition axis comes from free axis of ``stationary``, while free axis comes from free axis of ``moving``.

  Example:

  .. nki_example:: ../../test/test_nki_isa_nc_matmul.py
   :language: python
   :marker: NKI_EXAMPLE_0

  """
  ...

def nc_stream_shuffle(src, dst, shuffle_mask, *, dtype=None, mask=None, **kwargs):
  r"""
  Apply cross-partition data movement within a quadrant of 32 partitions from source tile
  ``src`` to destination tile ``dst`` using Vector Engine.

  Both source and destination tiles can be in either SBUF or PSUM, and passed in by reference as arguments.
  In-place shuffle is allowed, i.e., ``dst`` same as ``src``. ``shuffle_mask`` is a 32-element list. Each mask
  element must be in data type int or affine expression. ``shuffle_mask[i]`` indicates which input partition the
  output partition [i] copies from within each 32-partition quadrant. The special value ``shuffle_mask[i]=255``
  means the output tensor in partition [i] will be unmodified. ``nc_stream_shuffle`` can be applied to multiple
  of quadrants. In the case with more than one quadrant, the shuffle is applied to each quadrant independently,
  and the same ``shuffle_mask`` is used for each quadrant. ``mask`` applies to ``dst``, meaning that locations
  masked out by ``mask`` will be unmodified. For more information about the cross-partition data movement,
  see :ref:`arch_guide_cross_partition_data_movement`.

  This API has 3 constraints on ``src`` and ``dst``:

  #. ``dst`` must have same data type as ``src``.
  #. ``dst`` must have the same number of elements per partition as ``src``.
  #. The access start partition of ``src`` (``src_start_partition``), does not have to match or be in the same quadrant
     as that of ``dst`` (``dst_start_partition``). However, ``src_start_partition``/``dst_start_partition`` needs to follow
     some special hardware rules with the number of active partitions ``num_active_partitions``.
     ``num_active_partitions = ceil(max(src_num_partitions, dst_num_partitions)/32) * 32``, where ``src_num_partitions`` and
     ``dst_num_partitions`` refer to the number of partitions the ``src`` and ``dst`` tensors access respectively.
     ``src_start_partition``/``dst_start_partition`` is constrained based on the value of ``num_active_partitions``:

    * If ``num_active_partitions`` is 96/128, ``src_start_partition``/``dst_start_partition`` must be 0.

    * If ``num_active_partitions`` is 64, ``src_start_partition``/``dst_start_partition`` must be 0/64.

    * If ``num_active_partitions`` is 32, ``src_start_partition``/``dst_start_partition`` must be 0/32/64/96.

  **Estimated instruction cost:**

  ``max(MIN_II, N)`` Vector Engine cycles, where ``N`` is the number of elements per
  partition in ``src``, and ``MIN_II`` is the minimum instruction initiation interval
  for small input tiles. ``MIN_II`` is roughly 64 engine cycles.

  :param src: the source tile
  :param dst: the destination tile
  :param shuffle_mask: a 32-element list that specifies the shuffle source and destination partition
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)

  **Example:**

  .. nki_example:: ../../test/test_nki_isa_nc_stream_shuffle.py
   :language: python
   :marker: NKI_EXAMPLE_0

  .. nki_example:: ../../test/test_nki_isa_nc_stream_shuffle.py
   :language: python
   :marker: NKI_EXAMPLE_1

  .. nki_example:: ../../test/test_nki_isa_nc_stream_shuffle.py
   :language: python
   :marker: NKI_EXAMPLE_2

  """
  ...

def nc_transpose(data, *, mask=None, dtype=None, engine=engine.unknown, **kwargs):
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
  ``nc_transpose(engine=nki.isa.constants.engine.tensor)``.


  **Estimated instruction cost:**

  .. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Cost `(Engine Cycles)`
      - Condition
    * - ``max(MIN_II, N)``
      - ``engine`` set to ``nki.isa.constants.engine.vector``
    * - ``max(P, min(64, F))``
      - ``engine`` set to ``nki.isa.constants.engine.tensor`` and assuming many back-to-back ``nc_transpose`` of the same shape on Tensor Engine

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

def range_select(*, on_true_tile, comp_op0, comp_op1, bound0, bound1, reduce_cmd=reduce_cmd.idle, reduce_res=None, reduce_op=np.amax, range_start=0, on_false_value=fp32.min, mask=None, dtype=None, **kwargs):
  r"""

    Select elements from ``on_true_tile`` based on comparison with bounds using Vector Engine.

    .. note::

      Available only on NeuronCore-v3 and beyond.

    For each element in ``on_true_tile``, compares its free dimension index + ``range_start`` against ``bound0`` and ``bound1``
    using the specified comparison operators (``comp_op0`` and ``comp_op1``). If both comparisons
    evaluate to True, copies the element to the output; otherwise uses  ``on_false_value``.

    Additionally performs a reduction operation specified by ``reduce_op`` on the results,
    storing the reduction result in ``reduce_res``.

    **Note on numerical stability:**

    In self-attention, we often have this instruction sequence: ``range_select`` (VectorE) -> ``reduce_res`` -> ``activation`` (ScalarE).
    When ``range_select`` outputs a full row of ``fill_value``, caution is needed to avoid NaN in the
    activation instruction that subtracts the output of ``range_select`` by ``reduce_res`` (max value):

    - If ``dtype`` and ``reduce_res`` are both FP32, we should not hit any NaN issue
      since ``FP32_MIN - FP32_MIN = 0``. Exponentiation on 0 is stable (1.0 exactly).

    - If ``dtype`` is FP16/BF16/FP8, the fill_value in the output tile will become ``-INF``
      since HW performs a downcast from FP32_MIN to a smaller dtype.
      In this case, you must make sure reduce_res uses FP32 ``dtype`` to avoid NaN in ``activation``.
      NaN can be avoided because ``activation`` always upcasts input tiles to FP32 to perform math operations: ``-INF - FP32_MIN = -INF``.
      Exponentiation on ``-INF`` is stable (0.0 exactly).

    **Constraints:**

    The comparison operators must be one of:

    - np.equal
    - np.less
    - np.less_equal
    - np.greater
    - np.greater_equal

    Partition dim sizes must match across ``on_true_tile``, ``bound0``, and ``bound1``:

    - ``bound0`` and ``bound1`` must have one element per partition
    - ``on_true_tile`` must be one of the FP dtypes, and ``bound0/bound1`` must be FP32 types.

    The comparison with ``bound0``, ``bound1``, and free dimension index is done in FP32.
    Make sure ``range_start`` + free dimension index is within 2^24 range.

    **Estimated instruction cost:**

    ``max(MIN_II, N)`` Vector Engine cycles, where:

    - ``N`` is the number of elements per partition in ``on_true_tile``, and
    - ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
    - ``MIN_II`` is roughly 64 engine cycles.

    **Numpy equivalent:**

    .. code-block:: python

        indices = np.zeros(on_true_tile.shape)
        indices[:] = range_start + np.arange(on_true_tile[0].size)

        mask = comp_op0(indices, bound0) & comp_op1(indices, bound1)
        select_out_tile = np.where(mask, on_true_tile, on_false_value)
        reduce_tile = reduce_op(select_out_tile, axis=1, keepdims=True)

    :param on_true_tile: input tile containing elements to select from
    :param on_false_value: constant value to use when selection condition is False.
      Due to HW constraints, this must be FP32_MIN FP32 bit pattern
    :param comp_op0: first comparison operator
    :param comp_op1: second comparison operator
    :param bound0: tile with one element per partition for first comparison
    :param bound1: tile with one element per partition for second comparison
    :param reduce_op: reduction operator to apply on across the selected output. Currently only ``np.max`` is supported.
    :param reduce_res: optional tile to store reduction results.
    :param range_start: starting base offset for index array for the free dimension of ``on_true_tile``
      Defaults to 0, and must be a compiler time integer.
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
    :return: output tile with selected elements

    Example:

    .. nki_example:: ../../test/test_nki_isa_range_select.py
       :language: python
       :marker: NKI_EXAMPLE_0

    Alternatively, ``reduce_cmd`` can be used to chain multiple calls to the same accumulation
    register to accumulate across multiple range_select calls. For example:

    .. nki_example:: ../../test/test_nki_isa_range_select.py
       :language: python
       :marker: NKI_EXAMPLE_1

    """
  ...

def reciprocal(data, *, dtype=None, mask=None, **kwargs):
  r"""
  Compute reciprocal of each element in the input ``data`` tile using Vector Engine.

  **Estimated instruction cost:**

  ``max(MIN_II, 8*N)`` Vector Engine cycles, where ``N`` is the number of elements per partition in ``data``, and
  ``MIN_II`` is the minimum instruction initiation interval for small input tiles. ``MIN_II`` is roughly 64 engine cycles.

  :param data: the input tile
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :return: an output tile of reciprocal computation

  Example:

  .. nki_example:: ../../test/test_nki_isa_reciprocal.py
   :language: python
   :marker: NKI_EXAMPLE_6

  """
  ...

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

def select_reduce(*, dst, predicate, on_true, on_false, reduce_res=None, reduce_cmd=reduce_cmd.idle, reduce_op=np.amax, reverse_pred=False, mask=None, dtype=None, **kwargs):
  r"""
    Selectively copy elements from either ``on_true`` or ``on_false`` to the destination tile
    based on a ``predicate`` using Vector Engine, with optional reduction (max).

    The operation can be expressed in NumPy as:

    .. code-block:: python

        # Select:
        predicate = ~predicate if reverse_pred else predicate
        result = np.where(predicate, on_true, on_false)

        # With Reduce:
        reduction_result = np.max(result, axis=1, keepdims=True)

    **Memory constraints:**

    - Both ``on_true`` and ``predicate`` are permitted to be in SBUF
    - Either ``on_true`` or ``predicate`` may be in PSUM, but not both simultaneously
    - The destination ``dst`` can be in either SBUF or PSUM

    **Shape and data type constraints:**

    - ``on_true``, ``dst``, and ``predicate`` must have identical shapes (same number of partitions and elements per partition)
    - ``on_true`` can be any supported dtype except ``tfloat32``, ``int32``, ``uint32``
    - ``on_false`` dtype must be ``float32`` if ``on_false`` is a scalar.
    - ``on_false`` has to be either scalar or vector of shape ``(on_true.shape[0], 1)``
    - ``predicate`` dtype can be any supported integer type ``int8``, ``uint8``, ``int16``, ``uint16``
    - ``reduce_res`` must be a vector of shape ``(on_true.shape[0], 1)``
    - ``reduce_res`` dtype must of float type
    - ``reduce_op`` only supports ``max``

    **Behavior:**

    - Where predicate is True: The corresponding elements from ``on_true`` are copied to ``dst``
    - Where predicate is False: The corresponding elements from ``on_false`` are copied to ``dst``
    - When reduction is enabled, the max value from each partition of the ``result`` is computed and stored in ``reduce_res``

    **Accumulator behavior:**

    The Vector Engine maintains internal accumulator registers that can be controlled via the ``reduce_cmd`` parameter:

    - ``nisa.reduce_cmd.reset_reduce``: Reset accumulators to -inf, then accumulate the current results
    - ``nisa.reduce_cmd.reduce``: Continue accumulating without resetting (useful for multi-step reductions)
    - ``nisa.reduce_cmd.idle``: No accumulation performed (default)

    .. note::
      Even when ``reduce_cmd`` is set to ``idle``, the accumulator state may still be modified.
      Always use ``reset_reduce`` after any operations that ran with ``idle`` mode to ensure
      consistent behavior.

    .. note::
      The accumulator registers are shared for other Vector Engine accumulation instructions such :doc:`nki.isa.range_select <nki.isa.range_select>`

    :param dst: The destination tile to write the selected values to
    :param predicate: Tile that determines which value to select (on_true or on_false)
    :param on_true: Tile to select from when predicate is True
    :param on_false: Value to use when predicate is False, can be a scalar value or a vector tile of ``(on_true.shape[0], 1)``
    :param reduce_res: (optional) Tile to store reduction results, must have shape ``(on_true.shape[0], 1)``
    :param reduce_cmd: (optional) Control accumulator behavior using ``nisa.reduce_cmd`` values, defaults to idle
    :param reduce_op: (optional) Reduction operator to apply (only ``np.max`` is supported)
    :param reverse_pred: (optional) Reverse the meaning of the predicate condition, defaults to False
    :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
    :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.

    **Example 1: Basic selection**

    .. nki_example:: ../../test/test_nki_isa_select_reduce.py
       :language: python
       :marker: NKI_EXAMPLE_1

    **Example 2: Selection with reduction**

    .. nki_example:: ../../test/test_nki_isa_select_reduce.py
       :language: python
       :marker: NKI_EXAMPLE_2

    **Example 3: Selection with reversed predicate**

    .. nki_example:: ../../test/test_nki_isa_select_reduce.py
       :language: python
       :marker: NKI_EXAMPLE_3
    """
  ...

def sequence_bounds(*, segment_ids, dtype=None):
  r"""
  Compute the sequence bounds for a given set of segment IDs using GpSIMD Engine.

  Given a tile of segment IDs, this function identifies where each segment begins and ends.
  For each element, it returns a pair of values: [start_index, end_index] indicating
  the boundaries of the segment that element belongs to. All segment IDs must be non-negative
  integers. Padding elements (with segment ID of zero) receive special boundary
  values: a start index of n and an end index of (-1), where n is the length
  of ``segment_ids``.

  The output tile contains two values per input element: the start index (first column)
  and end index (second column) of each segment. The partition dimension must always be 1.
  For example, with input shape (1, 512), the output shape becomes (1, 2, 512), where
  the additional dimension holds the start and end indices for each element.

  The input tile (``segment_ids``) must have data type np.float32 or np.int32.
  The output tile data type is specified using the ``dtype`` field (must be np.float32 or np.int32).
  If ``dtype`` is not specified, the output data type will be the same as the input
  data type of ``segment_ids``.

  **NumPy equivalent:**

  .. nki_example:: ../../test/test_nki_isa_sequence_bounds.py
   :language: python
   :marker: NKI_EXAMPLE_1

  :param segment_ids: tile containing the segment IDs. Elements with ID=0 are treated as padding.
  :param dtype: data type of the output (must be np.float32 or np.int32)
  :return: tile containing the sequence bounds.

  Example:

  .. nki_example:: ../../test/test_nki_isa_sequence_bounds.py
   :language: python
   :marker: NKI_EXAMPLE_0

  """
  ...

def tensor_copy(src, *, mask=None, dtype=None, engine=engine.unknown, **kwargs):
  r"""
  Create a copy of ``src`` tile within NeuronCore on-chip SRAMs using Vector, Scalar or GpSimd Engine.

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

def tensor_copy_dynamic_dst(*, dst, src, mask=None, dtype=None, engine=engine.unknown, **kwargs):
  r"""
  Create a copy of ``src`` tile within NeuronCore on-chip SRAMs using Vector or Scalar or GpSimd Engine,
  with ``dst`` located at a dynamic offset within each partition.

  Both source and destination tiles can be in either SBUF or PSUM.

  The source and destination tiles must also have the same number of partitions and the same number of elements
  per partition.

  The dynamic offset must be a scalar value resided in SBUF. If you have a list of dynamic offsets
  for scattering tiles in SBUF/PSUM, you may loop over each offset and call ``tensor_copy_dynamic_dst``
  once per offset.

  **Estimated instruction cost:**

  ``max(MIN_II_DYNAMIC, N)`` engine cycles, where:

  -  ``N`` is the number of elements per partition in the ``src`` tile,
  -  ``MIN_II_DYNAMIC`` is the minimum instruction initiation interval for instructions with dynamic destination location.
     ``MIN_II_DYNAMIC`` is roughly 600 engine cycles.

  :param dst: the destination of copy, must be a tile in SBUF of PSUM that is dynamically indexed within each dimension.
  :param src: the source of copy, must be a tile in SBUF or PSUM.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param engine: (optional) the engine to use for the operation: `nki.isa.vector_engine`, `nki.isa.gpsimd_engine`,
                 `nki.isa.scalar_engine` or `nki.isa.unknown_engine` (default, let compiler select best engine).

  """
  ...

def tensor_copy_dynamic_src(src, *, mask=None, dtype=None, engine=engine.unknown, **kwargs):
  r"""
  Create a copy of ``src`` tile within NeuronCore on-chip SRAMs using Vector or Scalar or GpSimd Engine,
  with ``src`` located at a dynamic offset within each partition.

  Both source and destination tiles can be in either SBUF or PSUM. By default, this API returns
  a tile in SBUF, unless the returned value is assigned to a pre-declared PSUM tile.

  The source and destination tiles must also have the same number of partitions and the same number of elements
  per partition.

  The dynamic offset must be a scalar value resided in SBUF. If you have a list of dynamic offsets
  for gathering tiles in SBUF/PSUM, you may loop over each offset and call ``tensor_copy_dynamic_src``
  once per offset.

  **Estimated instruction cost:**

  ``max(MIN_II_DYNAMIC, N)`` engine cycles, where:

  -  ``N`` is the number of elements per partition in the ``src`` tile,
  -  ``MIN_II_DYNAMIC`` is the minimum instruction initiation interval for instructions with dynamic source location.
     ``MIN_II_DYNAMIC`` is roughly 600 engine cycles.

  :param src: the source of copy, must be a tile in SBUF or PSUM that is dynamically indexed within each partition.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.
  :param engine: (optional) the engine to use for the operation: `nki.isa.vector_engine`, `nki.isa.gpsimd_engine`,
                 `nki.isa.scalar_engine` or `nki.isa.unknown_engine` (default, let compiler select best engine).
  :param return: the modified destination of copy.

  Example:

  .. nki_example:: ../../test/test_nki_isa_tensor_copy_dynamic.py
   :language: python
   :marker: NKI_EXAMPLE_0

  .. nki_example:: ../../test/test_nki_isa_tensor_copy_dynamic.py
   :language: python
   :marker: NKI_EXAMPLE_1

  """
  ...

def tensor_copy_predicated(*, src, dst, predicate, reverse_pred=False, mask=None, dtype=None, **kwargs):
  r"""
  Conditionally copy elements from the ``src`` tile to the destination tile on SBUF / PSUM
  based on a ``predicate`` using Vector Engine.

  This instruction provides low-level control over conditional data movement on NeuronCores,
  optimized for scenarios where only selective copying of elements is needed. Either ``src`` or
  ``predicate`` may be in PSUM, but not both simultaneously. Both ``src`` and ``predicate`` are permitted to be in SBUF.

  Shape and data type constraints:

  1. ``src`` (if it is a tensor), ``dst``, and ``predicate`` must occupy the same number of partitions and same number of elements per partition.
  2. ``predicate`` must be of type ``uint8``, ``uint16``, or ``uint32``.
  3. ``src`` and ``dst`` must share the same data type.

  **Behavior:**

  - Where predicate is True: The corresponding elements from `src` are copied to `dst` tile. If `src` is a scalar, the scalar is copied to the `dst` tile.
  - Where predicate is False: The corresponding values in `dst` tile are unmodified

  **Estimated instruction cost:**

  .. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Cost ``(Vector Engine Cycles)``
      - Condition
    * - ``max(MIN_II, N)``
      - If ``src`` is from SBUF and ``predicate`` is from PSUM or the other way around
    * - ``max(MIN_II, 2N)``
      - If both ``src`` and ``dst`` are in SBUF

  - ``N`` is the number of elements per partition in ``src`` tile
  - ``MIN_II`` is the minimum instruction initiation interval for small input tiles.
    ``MIN_II`` is roughly 64 engine cycles.

  :param ``src``: The source tile or number to copy elements from when ``predicate`` is True
  :param ``dst``: The destination tile to copy elements to
  :param ``predicate``: A tile that determines which elements to copy
  :param reverse_pred: A boolean that reverses the effect of ``predicate``.
  :param mask: (optional) a compile-time constant predicate that controls whether/how this instruction is executed (see :ref:`nki-mask` for details)
  :param dtype: (optional) data type to cast the output type to (see :ref:`nki-dtype` for more information); if not specified, it will default to be the same as the data type of the input tile.

  Example:

  .. nki_example:: ../../test/test_nki_isa_copypredicated.py
    :language: python
    :marker: NKI_EXAMPLE_21

  """
  ...

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

def tensor_scalar(data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, *, dtype=None, mask=None, engine=engine.unknown, **kwargs):
  r"""
  Apply up to two math operators to the input ``data`` tile by broadcasting scalar/vector operands
  in the free dimension using Vector or Scalar or GpSimd Engine: ``(data <op0> operand0) <op1> operand1``.

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

  If arithmetic operators are used, the ``tensor_scalar`` instruction can run on Vector or Scalar or GpSimd Engine.
  However, each engine supports limited arithmetic operators (see :ref:``tbl-aluop``). The Scalar Engine on trn2 only
  supports a subset of the operator combination:

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
                 `nki.isa.gpsimd_engine` (only allowed for rsqrt) or `nki.isa.unknown_engine` (default, let
                 compiler select best engine based on the input tile shape).
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
  :param reverse0: `(not supported yet)` reverse ordering of inputs to ``op0``; if false, ``operand0`` is the rhs of ``op0``;
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

def tensor_tensor(data1, data2, op, *, dtype=None, mask=None, engine=engine.unknown, **kwargs):
  r"""
  Perform an element-wise operation of input two tiles using Vector Engine or GpSimd Engine.
  The two tiles must have the same partition axis size and the same number of elements per partition.

  The element-wise operator is specified using the ``op`` field and can be any *binary* operator
  supported by NKI (see :ref:`nki-aluop` for details) that runs on the Vector Engine,
  or it can be ``power`` or integer ``add``, ``multiply```, or ``subtract`` which run on the GpSimd Engine.
  For bitvec operators, the input/output data types must be integer types and Vector Engine treats
  all input elements as bit patterns without any data type casting. 
  For arithmetic operators, there is no
  restriction on the input/output data types, but the engine automatically casts input data types to float32
  and performs the element-wise operation in float32 math (unless it is one of the supported integer ops mentioned above). 
  The float32 results are cast to the target
  data type specified in the ``dtype`` field before written into the
  output tile. If the ``dtype`` field is not specified, it is default to be the same as the data type of ``data1``
  or ``data2``, whichever has the higher precision.

  Since GpSimd Engine cannot access PSUM, the input or output tiles cannot be in PSUM
  if ``op`` is one of the GpSimd operations mentioned above.
  (see :ref:`arch_sec_neuron_core_engines` for details).
  Otherwise, the output tile can be in either SBUF or PSUM.
  However, the two input tiles, ``data1`` and ``data2`` cannot both reside in PSUM.
  The three legal cases are:

  1. Both ``data1`` and ``data2`` are in SBUF.
  2. ``data1`` is in SBUF, while ``data2`` is in PSUM.
  3. ``data1`` is in PSUM, while ``data2`` is in SBUF.

  Note, if you need broadcasting capability in the free dimension for either input tile, you should consider
  using :doc:`nki.isa.tensor_scalar <nki.isa.tensor_scalar>` API instead,
  which has better performance than ``nki.isa.tensor_tensor`` in general.

  **Estimated instruction cost:**

  See below table for tensor_tensor performance when it runs on Vector Engine.

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
  :param engine: (optional) the engine to use for the operation: `nki.isa.vector_engine`, `nki.isa.gpsimd_engine`
                 or `nki.isa.unknown_engine` (default, let compiler select best engine based on the input tile shape).
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

