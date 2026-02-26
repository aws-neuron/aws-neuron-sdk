.. meta::
   :description: Best practices for migrating NKI kernels from Beta 1 to the Beta 2 NKI Compiler
   :keywords: NKI kernels, Neuron Kernel Interface, AWS Neuron SDK, kernel compilation, Trainium, Inferentia, machine learning acceleration

.. _nki-migration-guide:

=========================================
NKI Migration Guide from Beta 1 to Beta 2
=========================================

This topic covers best practices for migrating NKI kernels from the legacy 
``neuronxcc.nki.*`` namespace to the new ``nki.*`` namespace which uses the 
new NKI Compiler. See :ref:`nki-beta-versions` and :ref:`nki_compiler_about` 
for more in-depth information.

Background: NKI has a Compiler!
==================================

As of Release 2.27, NKI now has a new standalone compiler. The syntax of NKI 
remains a subset of Python. This means you can largely use Python syntax when 
writing NKI kernels. However, it is important to remember that your NKI 
functions are compiled by the NKI Compiler and not evaluated by the Python 
interpreter. The goal is to offer a better programming experience with more 
precise error messages.

With the NKI Compiler, we have chosen to define the NKI language as a subset 
of Python. This means that all NKI programs are valid Python programs, but not 
all Python programs are valid NKI programs. The delineation is the ``nki.jit`` 
decorator. Just as before, you mark your NKI kernels with the ``nki.jit`` 
decorator. However, unlike before, the functions under this decorator will be 
passed to the NKI Compiler and not be evaluated by the Python interpreter.

.. code-block:: python

   def a_function(x,y,z):
     # this is Python code

   @nki.jit
   def kernel(x,y,z):
     # this is NKI code

If you use Python features within a NKI kernel that are not supported, the NKI
Compiler will give an error. The goal is that programming in NKI is intuitive 
and convenient and all of the features you need are available and behave as 
expected. However, if you find some curious errors or confusing behavior, 
reach out to us on the NKI Samples repository on AWS Neuron GitHub.

This document is intended for experienced NKI developers who are looking to 
migrate their existing kernels to the Beta 2 NKI compiler. Most code snippets 
below are assumed to be executed within a valid NKI kernel.

Key Migration Items
===================

These are the key items to migrate existing kernel to the Beta 2 NKI Compiler.

What new features are available in NKI Beta 2?
----------------------------------------------

* A new namespace for NKI Beta 2, ``nki.*``
* ``device_print`` is available to inspect tensor values
* The behavior of loops and branching is consistent with regular Python
* Lists and dictionaries are available and their behavior in loops is consistent with regular Python
* Direct allocation APIs have been reworked

What features in ``neuronxcc.nki.*`` are not available in ``nki.*``?
--------------------------------------------------------------------

* ``arange`` has been removed, use slicing or :ref:`nki-aps`
* The ``mask`` parameter is no longer supported
* Block dimensions of tensors have been removed
* Explicit ``dst`` parameter is now required for ``nki.isa`` instructions and is always the first argument
* ``nl.load`` and ``nl.store`` have been removed, use ``nisa.dma_copy``
* Nested slicing is not available
* Dynamic Access syntax has changed
* Decorators on sub-kernels need to be removed
* Dictionaries support only string keys

New Features in NKI Beta 2
===========================

New namespace, new APIs
-----------------------

NKI Beta 2 introduces a number of changes to the language and to the 
compilation process. While we are deprecating NKI Beta 1, the Beta 2 release 
supports both versions of the language via namespaces. The Beta 1 APIs can 
be used via the ``neuronxcc.nki.*`` namespace, while Beta 2 has moved to the 
``nki.*`` namespace.

.. code-block:: python

   # Legacy Beta 1 APIs
   import neuronxcc.nki as nki
   import neuronxcc.nki.isa as nisa

   # New Beta 2 APIs
   import nki
   import nki.isa as nisa

We have made improvements to the APIs, like consistent naming, order of 
arguments, and matching more closely the hardware ISA so that what developers 
write in NKI and what they see in the profiler are the same. There is one 
change that developers should be aware of: all ISA functions now require a 
destination parameter.

All ISA functions require a destination parameter
-------------------------------------------------

In Beta 2, all of the ISA functions now require a ``dst`` parameter instead 
of returning a result. So, instead of writing:

.. code-block:: python

   result[...] = nisa.reciprocal(src)

Developers must write:

.. code-block:: python

   nisa.reciprocal(dst=result[...], src)

This change makes the behavior of the APIs more consistent and matches cases
where APIs may perform accumulation or return multiple results. It also helps 
avoid scenarios where developers might inadvertently write to the wrong buffer 
or inadvertently introduce additional copy operations.

Dynamic control flow
--------------------

NKI Beta 2 includes support for dynamic (on-chip) control flow. All of the 
dynamic control flow uses on-chip registers to hold the conditional values. 
See :ref:`trainium_inferentia2_arch` for more information. If a control flow 
construct uses a register as a conditional, then the loop will be an on-chip, 
dynamic (or runtime) loop. This is very common in scenarios like Mixture of 
Experts (MoE), where the index space for the expert is known at runtime, but 
not at compile time. Dynamic control flow with the new NKI APIs unlock this 
use case.

To support dynamic control flow, NKI has a new set of ``nki.isa`` APIs for 
reading and writing to hardware registers. See :doc:`/nki/api/index` for 
more information.

.. code-block:: python

   # Define a register
   def register_alloc(x: Optional[int]) -> register: ...

   # Fill the register with an immediate value
   def register_move(dst: imm: int): ...

   # Load SRAM tensor element into the dst register
   def register_load(dst: register, src: tensor): ...

   # Store the value of the register into SRAM
   def register_store(dst: tensor, src: register): ...

The most basic dynamic loop is a ``for`` loop that uses a register value for 
the iteration value and another register for the upper bound. Developers can 
write this kind of loop using ``dynamic_range``:

.. code-block:: python

   # dynamic loop with dynamically computed upper bounds
   # upper_bound is a hardware register
   # the loop index, i, is also a hardware register
   upper_bound = register_alloc()
   register_load(upper_bound, tensor)
   for i in dynamic_range(5, upper_bound, 2):
     ...

Developers can also write dynamic while loops. When using a dynamic while loop, 
the developer should update the register within the body of the loop.

.. code-block:: python

   # initialize a conditional tensor which will be updated in the loop
   cond = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=np.int32)

   # create register with initial value
   reg = register_alloc(5)

   while reg: # loop will terminate when the value reaches 0
     ...
     # store the register value into SBUF for computation
     nisa.register_store(cond, reg)
     # Decrement the condition variable by 1
     nisa.tensor_scalar(cond, cond, nl.add, -1)
     # load (updated) value from cond tensor into register
     nisa.register_load(reg, cond)

Update indexing syntax for ``mgrid`` and ``arange``
---------------------------------------------------

If using ``nl.mgrid/arange`` to access continuous elements in an existing NKI 
kernel, this should be replaced with integer slicing. Take a look at the 
following example.

.. code-block:: python

   # Example 1
   t = nl.ndarray(shape=(128, 16, 64), ...)
   # Old Approach: use mgrid to access continuous elements
   i_p, if0, if1 = nl.mgrid[0:128, 0:8, 0:64]
   t[i_p, if0, if1] 
   # Updated: should just use integers to create the slice
   t[0:128, 0:8, 0:64]

   # Example 2
   t = nl.ndarray(shape=(128, 16*64))
   # Old Approach: using mgrid
   i_p, if0, if1 = nl.mgrid[0:128, 0:8, 0:64]
   t[i_p, if0*64+i_f1]
   # should just use integer slicing
   t[0:128, 0:8*64]

If your use case cannot be represented with the slicing syntax above, see
:ref:`nki-aps`.

Changes in NKI Beta 2 from Beta 1
==================================

Consistent control flow behavior
--------------------------------

In NKI Beta 1, range iterators were converted into special objects that allowed 
the eDSL to capture the loop body. Because of this, loops were only executed once 
by the Python evaluator, which could lead to some surprising results. For example, 
in the code below, the normal Python variable ``var`` ends up with a value of 1 
rather than the expected value of 8. This has been solved in the new NKI Compiler.

.. code-block:: python

   val = 0
   for i in range(8):
     val += 1
   print(val) # will print 1 in Beta 1, prints 8 in Beta 2

For similar reasons, sometimes Python control flow constructs, such as ``if`` 
statements, could not be handled properly when nested within a ``for`` loop. 
For example, in Beta 1 the code below produces an undefined result. In Beta 2, 
this code produces the expected result.

.. code-block:: python

   val = 0
   for i in range(8):
     if i == 0:
       val = 1
     else:
       val = 2
   print(val) # undefined behaviour in Beta 1, prints 2 in Beta 2

Many other examples of troublesome control flow have been fixed, which should 
make using NKI easier and more intuitive.

.. _nki-mask:

Deprecation of masking
----------------------

Follow this section if you are using the ``mask`` parameter in your kernel.

In NKI Beta 1, the concept of masking was introduced to order modify the 
behavior of tensor indexing expressions. The use of masking was almost always 
used to avoid out-of-bounds access. For example, suppose a developer is tiling 
a tensor of size 129 x 513, and you want to use tiles of size 128 x 512. A 
typical way to write a tiling loop in Beta 1 is shown below.

.. code-block:: python

   t = nl.ndarray(shape=(129, 513), ...)
   result = nl.ndarray(shape=(129, 513), ...)
   for i in range(2):
     for j in range(2):
       i_p, i_f = nl.mgrid[0:128, 0:512]
       result[i_p+128*i, i_f+512*i] = nisa.tensor_copy(t[i_p+128*i, i_f+512*i],
        mask=(i_p+128*i<129) & (i_f+512*i<513))

Note, when ``i`` (or ``j``) is equal to 1, then the index expression 
``result[i_p+128*i, i_f+512*i]`` would overflow the tensor dimension. The mask 
expression ``mask=(i_p+128*i<129) & (i_f+512*i<513)`` modifies the indexing so 
that the equations are true, and thus inbounds of the tensor. This mechanism 
has many drawbacks, including being error-prone and non-intuitive for Python 
developers. Therefore, this mechanism has been deprecated in Beta 2.

In NKI Beta 2, developers can use standard constructs from Python such as 
``min`` and ``slice`` to build indexing expressions that are in bounds for 
the tensor. For example, the above code can now be written as:

.. code-block:: python

   for i in range(2):
     p_start = i * 128
     p_end = min(129, pstart + 128)
     p = slice(p_start, p_end)  # a.k.a. (p_start:p_end)
     
     for j in range(2):
       f_start = j * 512
       f_end = min(513, f_start + 512)
       f = slice(f_start, f_end)  # a.k.a. (f_start:f_end)
       
       nisa.tensor_copy(result[p, f], t[p, f])

The developer may also choose to inline the slices, if that is more natural. 
The below syntax is common in NKI Beta 1.

.. code-block:: python

   nisa.tensor_copy(result[p_start:p_end, f_start:f_end],
                         t[p_start:p_end, f_start:f_end])

Improved Allocation API
-----------------------

The manual allocation API has been simplified. In Beta 2 the there is a new 
argument to ``nl.ndarray`` that allows the offset of each tensor to be specified: 
(partition_offset, free_offset). Similar to the Beta 1, while the partition offset 
corresponds to a physical partition lane on the hardware, the free dimension offset 
is the element offset within each partition. The free dimension offset is 
translated into physical SBUF address in the compiler.

.. code-block:: python

   # creates your buffer on parition 0, offset by 128 elements of your data type
   a_result = nl.ndarray(dtype=a.dtype, shape=a.shape, name="result", 
     address=(0, 128), buffer=nl.sbuf)

The address space for PSUM is now also 2D to be consistent with the hardware. 
Recall that PSUM on NeuronCore v2/v3/v4 is organized into 128 partitions, each 
consisting of 16KB of memory. Each partition is further divided into 8 PSUM banks, 
with each bank holding up to 2048 bit worth of values. The allocation for PSUM 
tensors must start at the beginning of each bank - the compiler will throw an 
error otherwise.

For example, the following code will allocate a PSUM tensor on bank 3:

.. code-block:: python

   bank_id = 3
   PSUM_BANK_SIZE = 2048
   psum_t = nl.ndarray(dtype=nl.bfloat16, shape=(128, 1024), 
     address=(0, bank_id*PSUM_BANK_SIZE))

Translate from the Beta 1 Direct Allocation API
-----------------------------------------------

To translate the direct allocated kernel in Beta 1, all data structures must 
not use the block dimension. This means reformatting tensors to place the 
partition-dimension on the left-most position, using either lists or 
multi-dimensional tensors for the rest of your dimensions. See 
:ref:`nki_block_dimension_migration_guide` for more information.

After this, translate the address of each block. For example, given the 
following tensor in the Beta 1 that uses the modular allocation.

.. code-block:: python

   # beta 1 - uses block dimension and mod allocator
   k_loaded = nl.ndarray((num_512_tiles_cur_section, nl.par_dim(p_k), n_k), 
    dtype=nl.bfloat16, 
    buffer=sb_mod(base_addr=sca, num_free_tiles=(num_512_tiles_cur_section, )

Now with Beta 2, developers can translate the block dimension into a list 
and compute the address for each block.

.. code-block:: python

   # beta 2 - use lists of tensors and get lists of virtual byte addresses
   k_loaded_tensors = []
   for i in range(num_512_tiles_cur_section):
     k_loaded_tensors.append(nl.ndarray(shape=(p_k,n_k), dtype=nl.bfloat16, 
     buffer=nl.sbuf, address=(0, sca + (i%num_512_tiles_cur_section)*n_k*2 ) )

Remove nki.jit decorator on sub-kernels
---------------------------------------

For kernels that call other kernels, or call any other functions that are 
decorated with a ``nki.jit`` decorator, the ``nki.jit`` decorated will need to
be removed from sub-kernels.

In NKI Beta 1, all the sub-kernels called from a top-level kernel could be 
decorated with ``nki.jit(mode='trace')`` decorator. This decorator needs to be 
removed for the new NKI Compiler. Otherwise, you will see an error about classes 
needing to inherent from ``nl.NKIObject`` thrown from the callsite of the sub-kernels.

If a kernel is being called by another kernel and it is also called standalone, the 
decorator can be applied on-the-fly at the call site to avoid this problem.

.. code-block:: python

   # Do not apply the decorator on the kernel definition
   def my_kernel(...):
     pass
     
   # When calling the kernel, apply the decorator
   a = torch.tensor(...)
   kernel_decorated = nki.jit(my_kernel)
   result = kernel_decorated(a)

Translation of Block Dimensions
-------------------------------

If the kernel uses block dimension, defined as a tensor with a partition 
dimension set to any position other than the left-most position, this has been 
removed in Beta 2. There are two performance-equivalent ways to translate block 
dimensions. The first is to use a Python-like list and the second is to use a 
differently-shaped tensor.

Use a Python-like list
^^^^^^^^^^^^^^^^^^^^^^

Block dimension of tensors in Beta 1 was syntactic sugar for a list of tensors 
managed by the compiler. In NKI Beta 2, users can directly code this patten using 
standard lists, without extra compiler support.

.. code-block:: python

   # Before migration
   t = nl.ndarray((8, nl.par_dim(128), 256), dtype=nl.float32, buffer=nl.sbuf)
   for i in range(8):
     t[i]

   # After migration
   # Create an explicit list of tensors
   t_lst = []
   for i in range(8):
     t_lst.append(nl.ndarray(128, 256), dtype=nl.float32, buffer=nl.sbuf)
   for i in range(8):
     t_list[i]

With this approach, the programs generated before and after migration are 
identical and should yield the same performance.

Not using Python list
^^^^^^^^^^^^^^^^^^^^^

If blocks need to be alive at the same time, move the block dimension into 
free dimension

.. code-block:: python

   a = nl.ndarray((8, par_dim(128), 512), buffer=nl.sbuf, dtype=bfloat16)

   # ----> Migrate to
   a = nl.ndarray((128, 8, 512), buffer=nl.sbuf, dtype=bfloat16)

As an example, if all 8 blocks of add_buf need to be live at the same time, then 
the block dimension needs to be folded into the free dimension.

.. code-block:: python

   @nki.jit
   def sb_blocks(inp):
       res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
       add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
       for i in range(8):
           nisa.dma_copy(add_buf[i], inp[i])
       for i in range(8):
           nisa.dma_copy(res[i], add_buf[i])
       return res

   # should migrate to
   @nki.jit
   def sb_blocks_migrated(inp):
       res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
       add_buf = nl.ndarray(shape=(128, 8, 512), dtype=inp.dtype, buffer=nl.sbuf)
       for i in range(8):
           nisa.dma_copy(add_buf[0:128, i, 0:512], inp[i])
       for i in range(8):
           nisa.dma_copy(res[i], add_buf[0:128, i, 0:512])
       return res

If blocks do not need to be alive at the same time, remove the block 
dimension and relocate tensor declaration.

.. code-block:: python

   a = nl.ndarray((8, par_dim(128), 256))
   for i in nl.affine_range(8):
     <do something with a[i]>

   # should be transformed to ....
   for i in nl.affine_range(8):
     a = nl.ndarray((128, 256))
     <do something with a>

As an example, if all 8 blocks of add_buf do not need to be live at the same 
time, then remove the block dimension and relocate the tensor declaration 
inside the loop.

.. code-block:: python

   @nki.jit
   def sb_blocks(inp):
       res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
       add_buf = nl.ndarray(shape=(8, nl.par_dim(128), 512), dtype=inp.dtype, buffer=nl.sbuf)
       for i in range(8):
           nisa.dma_copy(add_buf[i], inp[i])
           nisa.dma_copy(res[i], add_buf[i])
       return res

   # should migrate to
   @nki.jit
   def sb_blocks_migrated(inp):
       res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
       for i in range(8):
           add_buf = nl.ndarray(shape=(128, 512), dtype=inp.dtype, buffer=nl.sbuf)
           nisa.dma_copy(add_buf[0:128, 0:512], inp[i])
           nisa.dma_copy(res[i], add_buf[0:128, 0:512])
       return res

It is important to note that the dependency relationship between loop iterations 
is different in ``sb_blocks_migrated`` and the following ``sb_blocks_migrated_incorrect`` 
shown below.

.. code-block:: python

   @nki.jit
   def sb_blocks_migrated_incorrect(inp):
       res = nl.ndarray(shape=(8, 128, 512), dtype=inp.dtype, buffer=nl.shared_hbm)
       add_buf = nl.ndarray(shape=(128, 512), dtype=inp.dtype, buffer=nl.sbuf)
       for i in range(8):
           nisa.dma_copy(add_buf[0:128, 0:512], inp[i])
           nisa.dma_copy(res[i], add_buf[0:128, 0:512])
       return res

In ``sb_blocks_migrated``, the compiler could unroll the loop and materialize 
multiple copies of the tensor ``add_buf``. However, in the ``sb_blocks_migrated_incorrect``, 
the execution will be serialized because the loop carries a dependency on ``add_buf``.

Dynamic Access Pattern
----------------------

Follow this section for a kernel that uses dynamic access, i.e. using a runtime value 
to index another tensor.

The syntax for representing dynamic access patterns has changed. In NKI Beta 1, 
an access with a dynamic scalar offset could be represented as shown below where 
``batch_idx`` is a dynamic value in the SBUF:

.. code-block:: python

   batch_idx = nl.multiply(nl.bitwise_and(nl.load(dynamic_idx), y=3), 128)
   result = nl.ndarray((128, 256), A.dtype, buffer=nl.shared_hbm)
   batch_idx[...] = 4 # set a constant, but batch_idx is a runtime SBUF value
   i_p, i_f = nl.mgrid[0:128, 0:256]
   nisa.dma_copy(src=A[batch_idx, i_p, i_f], dst=result[...])

Scalar Dynamic Access
^^^^^^^^^^^^^^^^^^^^^

In Beta 2, we need to use a physical access pattern, specified with the ``.ap`` 
method, to represent this.

.. code-block:: python

   def indirect_scalar_dynamic_dma(A):
     # Assume input A is of shape (4*128, 512). We want to copy from A[3*128:, 0:256]
     # The 3*128 offset comes from a dynamic variable in SBUF
     assert A.shape = [512, 512]
     batch_idx = nl.ndarray((1, 1), nl.int32, buffer=nl.sbuf)
     nisa.memset(batch_idx, value=3*128)

     result = nl.ndarray((128, 256), A.dtype, buffer=nl.shared_hbm)

     nisa.dma_copy(src=A.ap(
       pattern=[[512, 128], [1, 256]], offset=0, 
       scalar_offset=batch_idx, indirect_dim=0
       ),
       dst=result[...])

     return result

The ``scalar_offset`` is an SBUF value that specifies the index on the 
``indirect_dim`` of the tensor. For example, the code block above accesses 
``batch_idx`` on the 0-th dimension of the tensor ``A``. It is important 
to note that the dimension is relative to the **bast tensor**, not relative
to the **pattern** specified. 

This example will access the memory from ``A`` starting at the element offset below.

.. code-block:: python

   # prod(A.shape[indirect_dim+1:]) is the accumulated shape 
   # to the right of indirect_dim
   offset + scalar_offset * prod(A.shape[indirect_dim+1:])

In the example above, the access would start from:

.. code-block:: python

   0 + batch_idx * 512

Again, we should notice that ``512`` is read from the shape of the **base tensor**, not 
from the access pattern. The shape of the access pattern is ``(128, 256)``.

In conventional NumPy syntax, the above means that we will are accessing 
``A[batch_idx:batch_idx+128, 0:256]``. Writing this in the canonical loop form, 
the result of the access is the following:

.. code-block:: python

   result = nl.ndarray(shape=(128, 256), dtype=A.dtype, buffer=nl.sbuf)
   for x in range(128):
     for y in range(256):
       result[x, y] = A.flatten()[0 + batch_idx*512 + x*512 + y*1]

Vector Dynamic Access
^^^^^^^^^^^^^^^^^^^^^

Vector dynamic access is similar to that of scalar, except that we need to specify 
the field ``vector_offset``. **Currently, only ``indirect_dim=0`` is supported**. 
The stride on the leading dimension must be the the total number of 
elements to the right of the leading dimension in the **base tensor**, and the stride
specified in the leading dimension of the pattern in the `.ap()` is currently ignored.
We still recommend setting the stride properly so that code would still work if this
limitation is lifted in the future.

.. code-block:: python

   def indirect_vector_dynamic_dma(A):
     # shape of A is (128, 512)
     dynamic_idx_legal = nl.ndarray((64, 1), nl.int32, nl.sbuf)
     nisa.iota(dynamic_idx_legal, [[1, 1]], 0, 2)
     
     result_sb = nl.ndarray((64, 512), nl.float32, buffer=nl.sbuf)
     result_hbm = nl.ndarray((64, 512), nl.float32, buffer=nl.shared_hbm)

     nisa.dma_copy(src=A.ap(
       [[512, 64], [1, 512]], 0, vector_offset=dynamic_idx_legal, indirect_dim=0
       ), dst=result_sb, name='inst0')
    
     nisa.dma_copy(result_hbm, result_sb, name="copy1")

     return result_hbm

For this particular case, the semantics of the access are the following. Note that,
the stride on the dynamic dimension is directly read from the **base tensor**.

.. code-block:: python

   indirect_dimension = 0

   for w in range(64):
     for z in range(512):
       dynamic_idx = dynamic_idx_legal[w]
           A[
                  // static offsets
                  offset +
                  // AP with the indirect dimension number replaced
                  // Note that the 512 is read from the shape of the **base** tensor.
                  1 * z + 512 * dynamic_idx
                 ]

Further reading
---------------

- :doc:`/nki/deep-dives/nki-compiler`
- :doc:`/nki/api/index`

