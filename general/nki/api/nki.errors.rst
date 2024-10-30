NKI API Errors
==============

.. currentmodule:: nki.errors

.. _nki-errors-err_1d_arange_not_supported:

err_1d_arange_not_supported
---------------------------

Indexing a NKI tensor with 1D arange is not supported.

NKI expects tile indices to have at least two dimensions to match the underlying
memory (SBUF or PSUM)

.. code-block:: python

  tmp = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
  i = nl.arange(64)
  c = nl.exp(tmp[i, 0]) # Error: indexing tensor `tmp` with 1d arange is not supported,

You can workaround the problem by introducing new axes like the following code:

.. code-block:: python

  tmp = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
  i = nl.arange(64)[:, None]
  c = nl.exp(tmp[i, 0])

Or using simple slicing:

.. code-block:: python

  tmp = nl.zeros((128, 1), dtype=nl.float32, buffer=nl.sbuf)
  c = nl.exp(tmp[0:64, 0])

.. _nki-errors-err_annotation_shape_mismatch:

err_annotation_shape_mismatch
-----------------------------

Tensor shape and the annotated shape mismatch

NKI check the object shape based on python type annotation in the `target: type = value` syntax,
NKI will throw an error if the expected shape and the object shape mismatch.

For example:

.. code-block:: python

  import neuronxcc.nki.typing as nt
  data: nt.tensor[128, 512] = nl.zeros((par_dim(128), 128), dtype=np.float32) # Error: shape of `data[128, 128]` does not match the expected shape of [128, 512]

.. _nki-errors-err_cannot_assign_to_index:

err_cannot_assign_to_index
--------------------------

An `index` tensor does not support item assignment. You may explicitly call
`iota` to convert an `index` tensor to a normal `tile` before any assignments.

.. code-block:: python

    x = nl.arange(8)[None, :]
    x[0, 5] = 1024   # Error: 'index' tensor does not support item assignment
    y = nisa.iota(x, dtype=nl.uint32)
    y[0, 5] = 1024   # works

.. _nki-errors-err_control_flow_condition_depending_on_arange:

err_control_flow_condition_depending_on_arange
----------------------------------------------

Control-flow depending on `nl.arange` or `nl.mgrid` is not supported.

.. code-block:: python

  for j0 in nl.affine_range(4096):
    i1 = nl.arange(512)[None, :]
    j = j0 * 512 + i1
    if j > 2048: # Error: Control-flow depending on `nl.arange` or `nl.mgrid` is not supported
      y = nl.add(x[0, j], x[0, j - 2048])

In the above example, j depends on the value of `i1`, which is `nl.arange(512)[None, :]`.
NKI does not support using `nl.arange` or `nl.mgrid` in control-flow condition.
To workaround this error, you can use the `mask` parameter:

.. code-block:: python

  for j0 in nl.affine_range(4096):
    i1 = nl.arange(512)[None, :]
    j = j0 * 512 + i1
    y = nl.add(x[0, j], x[0, j - 2048], mask=j > 2048)

.. _nki-errors-err_dynamic_control_flow_not_supported:

err_dynamic_control_flow_not_supported
--------------------------------------

Dynamic control-flow depending on tensor value is currently not supported by NKI.

.. code-block:: python

  cnd = nl.load(a) # a have shape of [1, 1]
  if cnd:          # Error: dynamic control-flow depending on tensor value is not supported.
    nl.store(b, 1)

.. _nki-errors-err_exceed_max_supported_dimension:

err_exceed_max_supported_dimension
----------------------------------

NKI API tensor parameter exceeds max supported number of dimensions.

Certain NKI APIs have restrictions on how many dimensions the tensor parameter can have:

.. code-block:: python

  x = nl.zeros(shape=[64, 32, 2], dtype=np.float32, buffer=nl.sbuf)
  b = nl.transpose(x) # Error: parameter 'x[64, 32, 2]' of 'transpose' exceed max supported number of dimensions of 2.

  x = nl.zeros(shape=[64, 64], dtype=np.float32, buffer=nl.sbuf)
  b = nl.transpose(x) # Works if input `x` only have 2 dimensions (i.e. rank=2)

.. _nki-errors-err_failed_to_infer_tile_from_local_tensor:

err_failed_to_infer_tile_from_local_tensor
------------------------------------------

NKI requires inputs of all compute APIs to be valid tiles with the first dimension
being the partition dimension.

.. code-block:: python

  # We mark the second dimension as the partition dimension
  a = nl.zeros((4, nl.par_dim(8), 8), dtype=nl.float32, buffer=nl.sbuf)
  c = nl.add(a, 32) # Error: Failed to infer tile from tensor 'a',

To fix the problem you can use index tensor `a` to generate a tile whose first dimension is the partition dimension

.. code-block:: python

  # We mark the second dimension of tensor a as the partition dimension
  a = nl.zeros((4, nl.par_dim(8), 8), dtype=nl.float32, buffer=nl.sbuf)
  c = nl.ndarray((4, nl.par_dim(8), 8), dtype=nl.float32, buffer=nl.sbuf)
  for i in range(4):
    # result of `a[i]` is a tile with shape (8, 8) and the first dimension is the partition dimension
    c[i] = nl.add(a[i], 32) # works
    # Or explicitly generate a tile with `nl.arange`
    ix = nl.arange(8)[:, None]
    iy = nl.arange(8)[None, :]
    # result of `a[i, ix, iy]` is a tile with shape (8, 8) and the first dimension is the partition dimension
    c[i, ix, iy] = nl.add(a[i, ix, iy], 32) # also works

.. _nki-errors-err_indirect_indices_free_dim:

err_indirect_indices_free_dim
-----------------------------

Dynamic indexing for load/store only supports the indirect indexing 
to be on the partition or block dimension. Refer to the code examples in 
:doc:`nl.load <generated/nki.language.load>` and :doc:`nl.store <generated/nki.language.store>`.

Also, if you're using ``nl.mgrid`` you may get this error even though your indirect indexing 
was on the partition dimension, use ``nl.arange`` instead.

.. code-block:: python

  i_p, i_f = nl.mgrid[0:64, 0:512] # this won't work for dynamic access

  i_p = nl.arange(64)[:, None]  # this works for dynamic access
  i_f = nl.arange(512)[None, :]  
  
  data_tile = nl.load(data_tensor[idx_tile[i_p, 0], i_f]) 

.. _nki-errors-err_local_variable_used_out_of_scope:

err_local_variable_used_out_of_scope
------------------------------------

Tensors in NKI are not allowed to be used outside of their parent scope.

Tensors in NKI have a stricter scope rules than Python. In NKI, control blocks
in if/else/for statements will introduce their own scope for tensors. A tensor
defined in if/else/for control blocks are not allowed to be used outside of the
scope.

.. code-block:: python

  for i in range(4):
    if i < 2:
      tmp = nl.load(a)
    else:
      tmp = nl.load(b)
  
    nl.store(c, tmp) # Error: Local variable 'tmp' is referenced outside of its parent scope ...

To fix the problem, you can rewrite the above code as:

.. code-block:: python

  for i in range(4):
    tmp = nl.ndarray(shape=a.shape, dtype=a.dtype)
    if i < 2:
      tmp[...] = nl.load(a)
    else:
      tmp[...] = nl.load(b)
  
    nl.store(c, tmp)

This stricter scope rules may also introduce unexpected error like the following:

.. code-block:: python

  data = nl.zeros((par_dim(128), 128), dtype=np.float32)
  
  for i in nl.sequential_range(4):
    i_tile = nisa.iota(i, dtype=nl.uint32).broadcast_to(data.shape)
    data = data + i_tile # Warning: shadowing local tensor 'float32 data[128, 128]' with a new object, use 'data[...] =' if you want to update the existing object
  
  nl.store(ptr, value=data) # # Error: Local variable 'tmp' is referenced outside of its parent scope ...

To fix the problem you can follow the suggestion from the warning

.. code-block:: python

  data = nl.zeros((par_dim(128), 128), dtype=np.float32)
  
  for i in nl.sequential_range(4):
    i_tile = nisa.iota(i, dtype=nl.uint32).broadcast_to(data.shape)
    data[...] = data + i_tile
  
  nl.store(ptr, value=data)

.. _nki-errors-err_nested_kernel_with_spmd_grid:

err_nested_kernel_with_spmd_grid
--------------------------------

Calling a NKI kernel with a SPMD grid from another NKI kernel is not supported.

.. code-block:: python

  @nki.trace
  def kernel0(...):
    ...
  
  @nki.trace
  def kernel1(...):
    ...
  
  @nki_jit
  def kernel_top():
    kernel0(...)        # works
    kernel1[4, 4](...)  # Error: Calling kernel with spmd grid (kernel1[4,4]) inside another kernel is not supported

.. _nki-errors-err_nki_api_outside_of_nki_kernel:

err_nki_api_outside_of_nki_kernel
---------------------------------

Calling NKI API outside of NKI kernels is not supported.

Make sure the NKI kernel function is wrapped with the respective framework decorators
or nki.baremetal.

.. _nki-errors-err_num_partition_exceed_arch_limit:

err_num_partition_exceed_arch_limit
-----------------------------------

Number of partitions exceeds architecture limitation.

NKI requires the number of partitions of a tile to not exceed the architecture limitation of 128

For example in Trainium:

.. code-block:: python

  x = nl.zeros(shape=[256, 1024], dtype=np.float32, buffer=nl.sbuf) # Error: number of partitions 256 exceed architecture limitation of 128.
  x = nl.zeros(shape=[128, 1024], dtype=np.float32, buffer=nl.sbuf) # Works

.. _nki-errors-err_num_partition_mismatch:

err_num_partition_mismatch
--------------------------

Number of partitions mismatch.

Most of the APIs in the nki.isa module require all operands to have the same number of partitions.
For example, the nki.isa.tensor_tensor() requires all operands to have the same number of partitions.

.. code-block:: python

  x = nl.zeros(shape=[128, 512], dtype=np.float32, buffer=nl.sbuf)
  y0 = nl.zeros(shape=[1, 512], dtype=np.float32, buffer=nl.sbuf)
  z = nisa.tensor_tensor(x, y0, op=nl.add) # Error: number of partitions (dimension 0 size of a tile) mismatch in parameters (data1[128, 512], data2[1, 512]) of 'tensor_tensor'
  
  y1 = y0.broadcast_to([128, 512])         # Call `broadcast_to` to explicitly broadcast on the partition dimension
  z = nisa.tensor_tensor(x, y0, op=nl.add) # works because x and y1 has the same number of partitions

.. _nki-errors-err_read_modify_write_on_kernel_parameter:

err_read_modify_write_on_kernel_parameter
-----------------------------------------

Read-modify-write on kernel parameter is not supported.

.. code-block:: python

  def kernel(tensor_ref):
    a = nl.load(tensor_ref)
    b = a + 1
    nl.store(tensor_ref, a)  # Error: read-modify-write on kernel parameter `tensor_ref` is not supported

Consider doing the following:

1. introduce distinct input and output parameters, since NKI doesn't support inplace update on parameters
2. copy the kernel input parameter to a local variable,
3. modify the local variable
4. copy the local variable back to the output kernel parameter.

Specifically, the workaround looks like this:

.. code-block:: python

  def kernel(tensor_in_ref, tensor_out_ref):
    a = nl.load(tensor_in_ref)
    b = a + 1
    nl.store(tensor_out_ref, a)

.. _nki-errors-err_size_of_dimension_exceed_arch_limit:

err_size_of_dimension_exceed_arch_limit
---------------------------------------

Size of dimension exceeds architecture limitation.

Certain NKI APIs have restrictions on dimension sizes of the parameter tensor:

.. code-block:: python

  x = nl.zeros(shape=[128, 512], dtype=np.float32, buffer=nl.sbuf)
  b = nl.transpose(x) # Error: size of dimension 1 in 'x[128, 512]' of 'transpose' exceed architecture limitation of 128.

  x = nl.zeros(shape=[128, 128], dtype=np.float32, buffer=nl.sbuf)
  b = nl.transpose(x) # Works size of dimension 1 < 128

.. _nki-errors-err_store_dst_shape_smaller_than_other_shape:

err_store_dst_shape_smaller_than_other_shape
--------------------------------------------

Illegal shape in assignment destination.

The destination of assignment must have the same or bigger shape than the source
of assignment. Assigning multiple values to the same element in the assignment
destination from a single NKI API is not supported

.. code-block:: python

  x = nl.zeros(shape=(128, 512), dtype=nl.float32, buffer=nl.sbuf)
  y = nl.zeros(shape=(128, 1), dtype=nl.float32, buffer=nl.sbuf)

  y[...] = x # Error: Illegal assignment destination shape in 'a = b': shape [128, 1] of parameter 'a' is smaller than other parameter shapes b[128, 512].
  x[...] = y # ok, if we are broadcasting from source to the destination of the assignment

.. _nki-errors-err_tensor_access_out_of_bound:

err_tensor_access_out_of_bound
------------------------------

Tensor access out-of-bound.

Out-of-bound access is considered illegal in NKI. When the indices are calculated
from nki indexing APIs, out-of-bound access results in a compile-time error.
When the indices are calculated dynamically at run-time, such as indirect
memory accesses, out-of-bound access results in run-time exceptions during
execution of the kernel.

.. code-block:: python

  x = nl.ndarray([128, 4000], dtype=np.float32, buffer=nl.hbm)
  for i in nl.affine_range((4000 + 512 - 1) // 512):
    tile = nl.mgrid[0:128, 0:512]
    nl.store(x[tile.p, i * 512 + tile.x], value=0)  # Error: Out-of-bound access for tensor `x` on dimension 1: index range [0, 4095] exceed dimension size of 4000

You could carefully check the corresponding indices and make necessary correction.
If the indices are correct and intentional, out-of-bound access can be avoided by
providing a proper mask:

.. code-block:: python

  x = nl.ndarray([128, 4000], dtype=np.float32, buffer=nl.hbm)
  for i in nl.affine_range((4000 + 512 - 1) // 512):
    tile = nl.mgrid[0:128, 0:512]
    nl.store(x[tile.p, i * 512 + tile.x], value=0,
              mask=i * 512 + tile.x < 4000)  # Ok

.. _nki-errors-err_tensor_output_not_written_to:

err_tensor_output_not_written_to
--------------------------------

A tensor was either passed as an output parameter to kernel but never written to, or
no output parameter was passed to the kernel at all. At least one output parameter 
must be provided to kernels.

If you did pass an output parameter to your kernel, and this still occurred, this means the tensor
was never written to. The most common cause for this is a dead-loop, such as when a range expression 
evaluates to 0 and the loop performing the store operation is not actually being entered. But this can occur
in any situation in which a loop is never entered, regardless of flow-control construct (for, if, while, etc..)

.. code-block:: python

  def incorrect(tensor_in, tensor_out):
    M = 128
    N = M + 1
  
    for i in nl.affine_range( M // N ): # This is the cause of the error, as N > M, M // N will evaluate to 0
      a = nl.load(tensor_in)
      nl.store(tensor_out, value=a) # This store will never be called.
  
  def also_incorrect_in_the_same_way(tensor_in, tensor_out, cnd):
    # This will cause the error if the value of `cnd` is False
    while cnd: 
      a = nl.load(tensor_in)
      nl.store(tensor_out, value=a) # This store will never be called.


Consider doing the following:

1. Evaluate your range expressions and conditionals to make sure they're what you intended. If you were trying to perform
   a computation on tiles smaller than your numerator (M in this case), use math.ceil() around your
   range expression. e.g. nl.affine_range(math.ceil(M / N)). You will likely need to pass a mask to your
   load and store operations as well to account for this.

2. If the possible dead-loop is intentional, you need to issue a store that writes to the entire tensor
   somewhere in the kernel outside of the dead loop. One good way to do this is to invoke
   :func:`~neuronxcc.nki.language.store` on your output tensor with a default value. 
   
   For example:

.. code-block:: python

  def memset_output(input, output, cnd):
    # Initialize the output if we cannot guarantee the output are always written later
    nl.store(output[i_p, i_f], value=0)
   
    while cnd: # Ok even if the value of `cnd` is False
      a = nl.load(tensor_in)
      nl.store(tensor_out, value=a)

.. _nki-errors-err_unexpected_output_dependencies:

err_unexpected_output_dependencies
----------------------------------

Unexpected output dependencies.

NKI assume kernel instances in the spmd grid and iteration between affine_range
can be executed in parallel require synchronization on the output. As a result,
each iteration of the loop will write to a different memory location.

.. code-block:: python

  a = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.sbuf)
  
  for i in nl.affine_range(4):
    a[0] = 0 # Unexpected output dependencies, different iterations of i loop write to `a[0]`

To fix the problem, you could either index the destination with the missing indices:

.. code-block:: python

  a = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.sbuf)
  
  for i in nl.affine_range(4):
    a[i] = 0 # Ok

Or if you want to write to the same memory location, you could use `sequential_range`
which allows writing to the same memory location:

.. code-block:: python

  a = nl.ndarray((4, 128, 512), dtype=nl.float32, buffer=nl.sbuf)

  for i in nl.sequential_range(4):
    a[0] = 0 # Also ok, we dont expect the sequential_range to execute in parallel

.. _nki-errors-err_unsupported_memory:

err_unsupported_memory
----------------------

NKI API parameters are in the wrong memory.

NKI enforces API-specific requirements on which memory the parameters are allocated,
that is, HBM, SBUF or PSUM. NKI will throw this error when the operands of a
NKI API call are not placed in the correct memory.

.. code-block:: python

  tmp = nl.ndarray((4, 4), dtype=nl.float32, buffer=nl.sbuf)
  x = nl.load(tmp) # Error: Expected operand 'src' of 'load' to be in address space 'hbm', but got a tile in 'sbuf' instead.
  
  tmp = nl.ndarray((4, 4), dtype=nl.float32, buffer=nl.hbm)
  x = nl.exp(tmp) # Error: Expected operand 'x' of 'exp' to be in address space 'psum|sbuf', but got a tile in 'hbm' instead.

.. _nki-errors-err_unsupported_mixing_basic_advanced_tensor_indexing:

err_unsupported_mixing_basic_advanced_tensor_indexing
-----------------------------------------------------

Mixing basic tensor indexing and advanced tensor indexing is not supported

.. code-block:: python

  a = nl.zeros((4, 4), dtype=nl.float32, buffer=nl.sbuf)
  i = nl.arange(4)[:, None]
  c = nl.exp(a[i, :]) # Error: Mixing basic tensor indexing and advanced tensor indexing is not supported.

You could avoid the error by either use basic indexing or advanced indexing but not both:

.. code-block:: python

  c = nl.exp(a[:, :]) # ok
  
  i = nl.arange(4)[:, None]
  j = nl.arange(4)[None. :]
  c = nl.exp(a[i, j]) # also ok

