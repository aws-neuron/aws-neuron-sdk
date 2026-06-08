.. meta::
    :description: Comprehensive guide to the NKI language for AWS Neuron SDK, covering tensor operations, control flow, memory management, and programming patterns for Trainium accelerators.
    :keywords: NKI, AWS Neuron, Language Guide, Tensor Operations, Trainium
    :date-modified: 04/08/2026

.. _nki-language-guide:

NKI Language Guide
==================

The Neuron Kernel Interface (NKI) language is designed for writing kernel functions to accelerate machine learning workloads on Trainium devices. This guide is an introduction to the NKI language and the key concepts you will need to know to program in NKI effectively.

Let us start by looking at a simple NKI function.

.. code-block:: python

    @nki.jit
    def nki_tensor_add_kernel(a_input, b_input):
        """
        NKI kernel to compute element-wise addition of two input tensors.
        """

        # Check both input tensor shapes/dtypes are the same for element-wise operation.
        assert a_input.shape == b_input.shape
        assert a_input.dtype == b_input.dtype

        print(f"adding tensors of type {a_input.dtype} and shape {a_input.shape}")

        # Check the first dimension's size to ensure it does not exceed on-chip
        # memory tile size, since this simple kernel does not tile inputs.
        assert a_input.shape[0] <= nl.tile_size.pmax

        # Allocate space for the input tensors in SBUF and copy the inputs from HBM
        # to SBUF with DMA copy.
        a_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a_input)

        b_tile = nl.ndarray(shape=b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b_input)

        # Allocate space for the result and use tensor_tensor to perform
        # element-wise addition. Note: the first argument of 'tensor_tensor'
        # is the destination tensor.
        c_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

        # Create a tensor in HBM and copy the result into HBM.
        c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=c_output, src=c_tile)

        # Return kernel output as function output.
        return c_output
        
.. important::
   The first thing you may notice about this NKI function is that it looks very much like a Python function. In fact, all NKI functions are syntactically valid Python functions. However, it is important to understand that NKI functions are not Python functions: they will be compiled by the NKI compiler and run on the Trainium accelerator. Because of this, not all Python constructs and libraries are supported within a NKI function.

The second thing to notice is that NKI has a sequential programming model. This means that the logical order of operations follows the syntactic order of the statements in the function. As you learn more about the Trainium hardware, you will see that the hardware can often do many things at the same time across the different compute engines on the Trainium devices. When we compile NKI functions, we will respect the sequential order of operations written by the programmer. The compiler may reorder operations that have no data dependencies, but this is functionally transparent to NKI programmers. Later you will see how to control which engines operations run on and even how to influence the ordering of operations with no data dependencies for better performance, but all of this is done in the context of the sequential ordering of the code.

The third thing to notice about this simple function is that is has a print statement. You may be wondering: When does this print happen? Does the Trainium hardware output a string, where does it go? What about all those different engines we just talked about and the sequential ordering? The answer to these questions reveal a very important aspect of NKI programming. The answer is that the print is evaluated by the compiler at compile time, not at runtime. So, when you compile this NKI function, the NKI compiler will output a string like:

.. code-block:: text

   adding tensors of type float16 and shape (128, 512)

However, when we run this compiled function on Trainium devices they will not output anything. This is usually what you want. The compiler gives important debugging information during compilation, but when you deploy your function across 1000 Trainium devices, they will not waste any time generating debug output. 

**Note**: There is a special print function that does run on the Trainium devices, called ``device_print``, that can be used if this is really what you need, see the API references for more information.

We have just seen that the print statement is evaluated at compile-time, and not at runtime. In fact, most things in NKI programs are evaluated at compile time. In general, calls to nki.isa.* functions will result in on-device operations, and (almost) all other things will be evaluated by the compiler at compile time. We will discuss some exceptions to this rule below, but for now it is generally the case that only the nki.isa.* calls result in run-time operations, and everything else is evaluated by the compiler at compile-time.

This leads us to our the last observation about NKI functions. The nki.isa.* APIs are the heart of the matter. These APIs are designed to expose the underlying hardware capabilities in as direct a way as possible. If you write a nki.isa function, then the hardware will execute that operation at that point in the program. The NKI meta-programming language simply provides a convenient way to specify which ISA operations you want to run on your data.

In the rest of this guide we will focus on the NKI language, starting with the compilation model and namespaces, then the values you can manipulate in a NKI function. We will then cover tensor indexing, control flow, and end with a discussion of class support, interoperation with Python, and composable kernels.

Compilation Model
------------------

When you decorate a function with ``@nki.jit`` and call it, the NKI compiler processes your kernel in three stages:

1. **Specialization**: The compiler takes your Python function and evaluates all meta-programming constructs. This includes resolving tensor shapes, unrolling loops, inlining function calls, and evaluating if-statements with compile-time conditions. The result is a specialized, flat sequence of ``nki.isa.*`` operations with all compile-time values resolved.

2. **Compilation**: The specialized program is lowered to Trainium machine code. This stage performs instruction scheduling, register allocation, and memory layout.

3. **Graph-compiler linking**: The compiled kernel is linked into the larger computation graph managed by the Neuron graph compiler, which handles data movement between the host and device.

The specialization stage is key to understanding NKI programming. During specialization, the compiler acts as an interpreter for the meta-programming parts of your kernel. Everything that is not a ``nki.isa.*`` call or a ``dynamic_range`` loop is evaluated and resolved at this stage. This means:

- All ``for`` loops (except ``dynamic_range``) are **unrolled** at specialization time. The compiler expands the loop body once for each iteration.
- All function calls are **inlined** at specialization time. The compiler substitutes the function body at each call site.
- All ``if`` statements with compile-time conditions are **resolved** at specialization time. Only the taken branch is included in the specialized program.
- All Python expressions on compile-time values (integers, booleans, strings, shapes) are **evaluated** at specialization time.

The only constructs that survive specialization and become runtime operations are ``nki.isa.*`` calls and ``dynamic_range`` loops. Everything else is part of the meta-programming language that controls how the final sequence of ISA operations is generated.

.. note::

   Throughout this documentation, we use the term **NKI meta-programming language** to refer to the Python subset that is evaluated at specialization time (loops, conditionals, function calls, and expressions on compile-time values), and **NKI language** to refer to the runtime primitives (``nki.isa.*`` operations and ``dynamic_range`` loops) that execute on the device.

.. code-block:: python

   @nki.jit
   def example_kernel(a_input):
       # Meta-programming: this loop is unrolled at specialization time
       for i in range(4):
           tile = nl.ndarray((128, 512), dtype=nl.float16, buffer=nl.sbuf)
           nisa.dma_copy(dst=tile, src=a_input[i * 128:(i + 1) * 128, :])
           # Meta-programming: this if is resolved at specialization time
           if i % 2 == 0:
               nisa.tensor_scalar(dst=tile, data=tile, op0=nl.add, operand0=1.0)

After specialization, this kernel becomes a flat sequence of ``dma_copy`` and ``tensor_scalar`` operations, with the loop and if-statement fully resolved.

Kernel Caching
^^^^^^^^^^^^^^

NKI caches compiled kernels based on their input arguments (shapes, dtypes, and compile-time values). This means NKI kernels must be **pure functions** of their arguments — the kernel's output must be determined solely by its input arguments. If a kernel's behavior depends on external state such as global variables, closures over mutable objects, or side effects, the cache may return a stale compiled artifact and produce incorrect results.

NKI Namespaces
---------------

NKI is organized into several Python namespaces:

- ``nki`` — The top-level package. Provides the ``@nki.jit`` decorator for compiling kernel functions.
- ``nki.language`` (commonly imported as ``nl``) — The high-level language API. This includes tensor creation (``ndarray``), data types, memory buffers, loop ranges (``affine_range``, ``dynamic_range``), and high-level math operations (``nl.add``, ``nl.matmul``, ``nl.softmax``, etc.). Many of the functions in ``nki.language`` are convenience wrappers around one or more ``nki.isa`` operations.
- ``nki.isa`` (commonly imported as ``nisa``) — The low-level instruction set architecture API. Each function in this namespace maps directly to a Trainium hardware operation. These are the only calls that produce runtime operations on the device.
- ``nki.collectives`` — APIs for multi-device collective communication operations such as ``all_reduce``, ``all_gather``, and ``collective_permute``.

A typical NKI kernel imports these namespaces as follows:

.. code-block:: python

   import nki
   import nki.language as nl
   import nki.isa as nisa

The distinction between ``nki.language`` and ``nki.isa`` is important. When you call a ``nki.language`` function like ``nl.add(a, b)``, the compiler may lower this to one or more ``nki.isa`` operations depending on the tensor shapes and types. When you call a ``nki.isa`` function like ``nisa.tensor_tensor(...)``, you are directly specifying the hardware operation. Use ``nki.language`` for readability and portability; use ``nki.isa`` when you need precise control over which hardware engine executes an operation.

NKI Values
-----------

The NKI language supports six types of values:

1. The special None value
2. Boolean values (True and False)
3. 32-bit integer values
4. 32-bit IEEE floating-point values
5. String literals
6. Tensors (on-device tensor memory)

In addition, NKI supports the following container types:

1. Tuples of any fixed length
2. Lists of arbitrary length
3. Dictionaries with string-value keys
4. Simple user-defined classes

NKI values and containers are very similar to their Python equivalents. For instance, you can use most of the Python standard list functions, and they work in the same way as in Python.

.. code-block:: python

   l = [1,2,3]    # create a list with 3 elements 
   l.append(4.1)  # append a value to the list
   l.extend(("Hello", "List")) # extend list with multiple values
   size = len(l) # return number of elements in list
   third = l[2]  # get third element of list (index 2)

   # search list for a specific value
   if l.index(2):
     print("list contains 2")
     
   # remove a specific value from a list (if present)
   l.remove(1)

   # print out list in reverse order
   l.reverse()
   for x in l:
     print(x)

The NKI dictionary type is also similar to the Python version, but with the restriction that the keys must be string values.

.. code-block:: python

    d = dict() # create an empty dictionary
    d['a'] = 1 # set a value in the dictionary

    print(d.keys())  # print out keys in dictionary
    print(d.items())  # print out values in dictionary

    # print out dictionary
    for k in d.keys():
        v = d[k]
        print(k, v)

    # remove value from dictionary if present
    if d.pop('a'):
        print("removed 'a' from dictionary")

    # fetch value of a, set to 2 if not present
    a = d.setdefault('a', 2)

We will discuss user-defined classes later in the guide. For now, let's take a close look at the most important value in NKI, the tensor.

Tensor Values
--------------

The ``NkiTensor`` class represents an on-chip tensor. That is, an ``NkiTensor`` instance is really a reference to some region of memory on the Trainium device at runtime. At compile-time, we do not yet know the precise location nor the precise contents of this tensor, and therefore, code evaluated at compile-time will not be able to query the precise location nor the contents. At compile-time we can only query meta-data about the tensor, such as its shape and element type. ``NkiTensor`` exposes the following meta-data:

* ``t.dtype`` - The element type of the tensor, e.g. "float16"
* ``t.shape`` - The shape of the tensor, e.g. (128,64,64)
* ``t.ndim`` - The number of dimensions
* ``t.size`` - The total number of elements
* ``t.offset`` - The access pattern offset (discussed below)
* ``t.buffer`` - The memory buffer this tensor lives in (discussed below)
* ``t.get_pattern()`` - The access pattern (discussed below)

The most commonly used fields are dtype and shape. We have already seen an example of using these fields to check that argument tensors are compatible in our simple example. Another common case is using a dimension of a shape to iterate over a tensor:

.. code-block:: python

   # assume t is a 3-dimensional tensor, we can iterate over the
   # 2-D subtensors
   for i in range(t.shape[0]):
     my_function(t[i])

Note, because the shape is part of the meta-data of the tensor, the expression ``t.shape[0]`` is a compile-time constant. Therefore, the bounds of the for-loop are known at compile time. The compiler will unroll this loop into a sequence of calls to my_function, one for each subtensor of t.

In addition to the basic meta-data fields, ``NkiTensor`` provides two methods for creating alternate views of the same underlying storage:

``view(dtype)``
  Reinterpret the tensor's storage bits as a different data type. The underlying memory is not modified; only the interpretation changes. This is useful for bitwise manipulation, such as reinterpreting ``int32`` values as ``float32``.

  .. code-block:: python

     int_tensor = nl.ndarray((128, 256), dtype=nl.int32, buffer=nl.sbuf)
     float_tensor = int_tensor.view(nl.float32)

``ap(pattern, offset=0, scalar_offset=None, vector_offset=None, indirect_dim=0, dtype=None)``
  Create a tensor with an explicit hardware access pattern sharing the same storage. The ``pattern`` is a list of ``[step, num]`` tuples that define how elements are accessed. This is an advanced feature for controlling the exact memory access pattern used by the hardware. See the architecture guide for details on access patterns.

  .. code-block:: python

     t = nl.ndarray((128, 1024), dtype=nl.float16, buffer=nl.sbuf)
     # Access every other element in the free dimension
     u = t.ap(pattern=[(1, 128), (2, 512)])

Creating Tensors
-----------------

The easiest way to create tensors is using the ``nki.language.ndarray`` API. This function takes a shape, a dtype, and a memory type, and returns an ``NkiTensor`` representing a reference to a memory region in the given memory type large enough to hold the tensor.

.. note::

   ``ndarray`` does **not** initialize memory. The contents of a newly allocated tensor are undefined until explicitly written to (e.g., via ``nisa.dma_copy`` or ``nisa.memset``).

.. code-block:: python

   # A matrix of 128x128 16-bit float values in the SBUF memory
   t = nl.ndarray((128,128), nl.float16, nl.sbuf)
   assert t.shape = (128,128)
   assert t.dtype == nl.float16
   assert t.buffer == nl.sbuf

You can also pass an optional ``name`` argument to ``ndarray``. The name is a string label that is propagated through the compiler into the generated IR and debug information. This can be helpful when profiling or debugging compiled kernels, since the name will appear in compiler output and diagnostic messages.

.. code-block:: python

   # Named tensor for easier identification in compiler output
   t = nl.ndarray((128,128), nl.float16, nl.sbuf, name="my_weights")

You can also create a tensor from an existing tensor using the ``reshape`` method. The ``reshape`` method will create a new reference to the same memory with a different shape. The reshaped tensor must have the same total number of elements as the original.

.. code-block:: python

   # create an alternate view of t with shape 128x2x64
   u = t.reshape((128,2,64))

   # create an alternate view of t with shape 128x32x4
   v = t.reshape((128,32,4))

In both cases, ``u`` and ``v`` refer to the same underlying memory as ``t``; no data is copied.

Tensor Indexing
----------------

Next, we will examine two meta-data fields related to tensor indexing: offset and pattern. But before we talk about these fields, let's look at the most common way of indexing tensors using integers and slices.

Suppose you have a tensor t with shape 64x64x64 that is in the SBUF memory. The SBUF memory is a two-dimensional block of memory, so the underlying storage for this 3-D tensor is a 2-D region of the SBUF. Recall, in the SBUF, the first dimension is called the partition dimension and the second dimension is called the free dimension. By convention, the first dimension of a tensor always corresponds to the partition dimension, and the remaining dimensions are laid out in the free dimension. Therefore, in our example, we have 64 partitions, each with 64*64=4096 elements.

We can refer to specific elements of the tensor using an index expression.

.. code-block:: python

   # 11th element in partition 0
   u = t[0,0,10]

   # 65th element in partition 0
   u = t[0,1,0]

   # last element of the tensor
   u = t[63,63,63]

It is more common to refer to whole sub-tensors rather then single elements, and for this we can use slices. A slice is an expression of the form start:stop:step, which describes a range of elements starting with index start, up to (but not including) index stop, and incrementing by step. If any of start, stop, or step are not specified, defaults will be used.

.. code-block:: python

   # All first 64 elements of every partition
   u = t[0:64, 0, 0:64]

   # Same as above, but using defaults
   u = t[:, 0, :]

   # Only the even elements of the third dimension
   u = t[:, :, ::2]

Finally, you can also use the ellipsis (...) to indicate defaults for a range of dimensions.

.. code-block:: python

   # the whole tensor t
   u = t[...]

   # same as above
   u = t[:,...]

   # use defaults for second dimension
   # equivalent to t[0,0:64,0:64]
   u = t[0,...,:]

Note, when you index into a tensor, the result is another tensor. So, in the examples above, the tensor u also has the normal tensor fields and capabilities. This means you can query the shape of the result, or further index the tensor u.

.. code-block:: python

   u = t[0,...]
   assert u.shape = (64,64)

   v = u[0:32, :]
   assert v.shape = (32, 64)

In addition to querying the shape, you can also query the hardware access pattern that corresponds to the tensor value. For example, the code below will display the access pattern that would be used to query u, which is a sub-tensor of t.

.. code-block:: python

   u = t[0,...]

   # check hardware access pattern
   print(u.offset)
   print(u.get_pattern())

For advanced use cases, the hardware access pattern can be specified directly.

.. code-block:: python

   # Specify HW access pattern directly
   u = t.ap(offset = 0, pattern = [...])

For more details on hardware access patterns, see the architecture guide.

Control Flow
-------------

NKI supports basic control flow constructs, including if-statements, for-loops over ranges, lists or tuples, and while loops. All of these constructs work similarly their equivalents in Python, but with one important difference: they are all evaluated at specialization time. This means the compiler unrolls every loop and resolves every branch before generating device code. For example, the code below uses a simple loop with a nested if statement to process the even and odd elements of a list differently.

.. code-block:: python

    inputs = [a, b, c]
    outputs = [x, y, z]

    assert len(inputs) == len(outputs)
    for i in range(len(inputs)):
        if i % 2 == 0:
            nisa.nc_transpose(dst=outputs[i], data=inputs[i])
        else:
            nisa.reciprocal(dst=outputs[i], data=inputs[i])

The loop and if-statement above will ultimately be evaluated away by NKI Compiler. This means that the ISA instructions will be included in the final executable as a linear sequence:

.. code-block:: python

   nki.isa.nc_transpose(dst=x, data=a)
   nki.isa.reciprocal(dst=y, data=b)
   nki.isa.nc_transpose(dst=z, data=c)

A for-loop can also iterate over a list or tuple, similar to Python. The two loops below both print the numbers 1-3 in sequence.

.. code-block:: python

   l = [1,2,3]
   for x in l:
     print(x)

   t = (1,2,3)
   for x in t:
     print(x)

Finally, NKI also supports while loops. Again these loops are similar to Python, and will be unrolled by the compiler, just like the for-loops.

.. code-block:: python

   # print the numbers 0-9
   x = 0
   while x < 10:
     print(x)
     x += 1

Dynamic Control Flow
----------------------

In the previous section we looked at control-flow constructs that are ultimately expanded at compile-time. NKI also supports dynamic control-flow, or control-flow that runs on the device. Dynamic control-flow is not expanded by the compiler, but lowered to equivalent Trainium control-flow instructions.

The most basic dynamic loop is a for-loop with static bounds. A dynamic loop with static bounds can be written using the standard for-loop with a dynamic_range hint.

.. code-block:: python

   # create a dynamic loop that runs "on chip"
   for i in dynamic_range(10):
     process_tensor(t[i])

The for loop above will lower to a loop on the Trainium device. The loop will execute its body (process_tensor), 10 times and then continue. Because this is a dynamic loop, the loop index, i, will be stored in a hardware register during evaluation. Therefore, the type of i is register in NKI. Register values can be used to index tensors, and passed to nki.isa APIs. We can also use registers to create dynamic loops with dynamic bounds.

.. code-block:: python

   count = nki.isa.register_alloc(0)
   nisa.register_load(count, count_tensor)
   for i in dynamic_range(count):
     process_tensor(t[i])

The loop above uses a register value as the upper bound. This register is allocated with the ``register_alloc`` function, and then its value is populated from a tensor using ``register_load``. The for loop will then execute ``count`` times.

There are four register APIs that can be used to create, and load and store values to and from registers. Each register is 32-bit and supports multiple data types: ``u8``, ``u16``, ``u32``, ``i8``, ``i16``, ``i32``, and ``fp32`` (or a pair of registers for ``u64``/``i64``). Signed integers are supported, so negative values (e.g., ``count=-5``) are valid. The register APIs return and operate on ``VirtualRegister`` objects.

A ``VirtualRegister`` represents a scalar value stored in a hardware register on the Trainium device. Unlike compile-time integer values, a ``VirtualRegister`` holds a value that exists at runtime. You can use a ``VirtualRegister`` as a loop bound for ``dynamic_range``, as a condition for a dynamic ``while`` loop, or as a ``scalar_offset`` in a tensor access pattern for dynamic indexing.

.. note::

   The induction variable of a ``dynamic_range`` loop is also a ``VirtualRegister``, but it is frozen: you cannot write to it with ``register_move`` or ``register_load``. This prevents ambiguity about whether modifying the induction variable would affect loop termination.

.. code-block:: python

   # allocate a new register with initial value (32-bit integer)
   def register_alloc(x: int) -> VirtualRegister: ...

   # store a constant integer into a register
   def register_move(dst: VirtualRegister, imm: int): ...

   # load a value from an SBUF tensor into a register
   # the source tensor must be a 1x1 SBUF tile
   def register_load(dst: VirtualRegister, src: tensor): ...

   # store the value of a register into an SBUF tensor
   def register_store(dst: tensor, src: VirtualRegister): ...

Using the APIs above, we can also create dynamic while loops. A dynamic while loop is specified using the standard while-loop with a condition that is a single register value. The NKI compiler will preserve while loops with register conditions, and not unroll them.

.. code-block:: python

   # suppose cond is an SBUF tensor, perhaps declared as
   cond = nl.ndarray((1, 1), buffer=nl.sbuf, dtype=nl.int32)

   # allocate a register with initial value 1
   reg = nisa.register_alloc(1)

   # This while loop is dynamic because the condition is a register
   while reg:
      # perform a calculation that updates cond
      nisa.dma_copy(dst=cond, ...)

      # update register used in while-loop condition
      nisa.register_load(reg, cond)

The code above uses a 1x1 SBUF tensor called cond to store the condition. We update this tensor in the body of the loop and then use register_load to update the register. When the register reg holds the value 0 the loop will terminate.

Class Support
--------------

NKI has basic support for user-defined classes. In NKI all classes are similar to Python data classes. When you declare a class for use in a NKI kernel, the class must inherit from NKIObject and no other classes. This restriction is to ensure the NKI compiler only brings in class definitions that are intended for NKI. A simple NKI class can be declared similar to a Python data class:

.. code-block:: python

   @dataclass 
   class C(NKIObject):
     x : int
     y : bool = False
     
     def toggle(self):
       self.y = not self.y
       
   c = C(1)
   c.toggle()

   # prints 1 True
   print(c.x, c.y)

The @dataclass decorator is optional; classes with and without the @dataclass decorator will be compiled in the same way by the NKI compiler. The compiler will create the initializer functions __init__ and __post_init__, if they are not provided by the user. For the class above, the default initializers are:

.. code-block:: python

   # default if not provided by the user
   def __init__(self, x = None, y = False):
     self.x = x
     self.y = y
     self.__post_init__()

   # default if not provided by the user
   def __post_init__(self):
     pass

Classes can be declared in Python and passed as arguments to NKI functions. When a class is used as an argument to a NKI kernel, the NKI kernel will import the definition of the Python class, and convert the Python class instance to a NKI instance using the objects dictionary. Currently, NKI does not look at slots or other object features, only the object dictionary. For example, consider the code shown below.

.. code-block:: python

   class A(NKIObject):
     x : int = 1
     def __init__(self, x):
       self.x = x

   @nki.jit
   def kernel(a : A): ...

   kernel(A(1))

The class A is instantiated in Python as an argument to the kernel function. The NKI compiler will take this object and translate it to an instance of A on the NKI side. Roughly this translation is done by translating the object dictionary, in pseudo-code:

.. code-block:: python

   # pseudo-code "copy constuct" A on NKI side
   def kernel(python_a : A):
     # make a NKI instance of class A
     nki_a = new A
     # populate NKI instance from Python instance
     nki_a.__dict__ = python_a.__dict__

Enumerations
-------------

In addition to the basic data classes described, NKI also supports basic enumerations. For example, the following can be used in NK kernel functions.

.. code-block:: python

   class E(Enum):
     x = 1
     y = 2
     z = 3

   def f(e : E):
     if e == E.x: ...
     elif e == E.y: ...
     elif e == E.z: ...
     
   f(E.x)

Similar to Python, the NKI compiler will translate the enumration class E to the following:

.. code-block:: python

   class E(NKIObject):
     x = E("x", 1)
     y = E("y", 2)
     z = E("z", 3)
     
     def __init__(self, name, value):
       self.name = name
       self.value = value

Equality in NKI is structural, so no additional code is needed to replicate the behavior of == and != for objects of type E. No other binary operators on enum values are supported.

Composable Kernels
-------------------

Because all functions are inlined at specialization time, NKI supports a powerful composition pattern: you can pass functions as arguments to other functions, and the compiler will inline them at each call site. This allows you to write generic kernel templates that can be specialized with different operations.

For example, consider a generic tiled processing kernel that applies a user-supplied function to each tile:

.. code-block:: python

   def tiled_process(input_tensor, output_tensor, tile_fn):
       """Generic kernel that applies tile_fn to each tile of the input."""
       for i in range(input_tensor.shape[0] // nl.tile_size.pmax):
           tile = nl.ndarray((128, 512), dtype=input_tensor.dtype, buffer=nl.sbuf)
           nisa.dma_copy(dst=tile, src=input_tensor[i * 128:(i + 1) * 128, :])

           result = nl.ndarray((128, 512), dtype=input_tensor.dtype, buffer=nl.sbuf)
           tile_fn(dst=result, src=tile)

           nisa.dma_copy(dst=output_tensor[i * 128:(i + 1) * 128, :], src=result)

   def my_activation(dst, src):
       nisa.activation(dst=dst, data=src, op=nl.relu)

   def my_scale(dst, src):
       nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=0.5)

   @nki.jit
   def relu_kernel(a_input, a_output):
       tiled_process(a_input, a_output, my_activation)

   @nki.jit
   def scale_kernel(a_input, a_output):
       tiled_process(a_input, a_output, my_scale)

During specialization, the compiler inlines ``tiled_process`` and then inlines the specific ``tile_fn`` (either ``my_activation`` or ``my_scale``) at each call site. The result is a fully specialized kernel with no function call overhead.

This pattern is especially useful for building mega-kernels that compose multiple operations. You can pass function references as hyperparameters when using the kernel builder API:

.. code-block:: python

   from nki.compiler.kernel_builder import compile_kernel

   compile_kernel(
       tiled_process,
       inputs={"input_tensor": input_array},
       outputs={"output_tensor": output_array},
       compile_opts=opts,
       tile_fn=my_activation,  # passed as a hyperparameter
   )

Functions can also be stored in data structures, returned from other functions, and selected dynamically at specialization time based on compile-time conditions:

.. code-block:: python

   def select_activation(name):
       if name == "relu":
           return my_relu
       elif name == "gelu":
           return my_gelu

   @nki.jit
   def kernel(a_input, a_output):
       act_fn = select_activation("relu")
       # act_fn is resolved at specialization time; the selected
       # function is inlined directly
       act_fn(dst=a_output, src=a_input)

Because all of this resolution happens at specialization time, there is no runtime cost. The compiled kernel contains only the specific ISA operations for the chosen function.
