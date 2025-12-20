.. meta::
    :description: Comprehensive guide to the NKI language for AWS Neuron SDK, covering tensor operations, control flow, memory management, and programming patterns for Trainium accelerators.
    :keywords: NKI, AWS Neuron, Language Guide, Tensor Operations, Trainium
    :date-modified: 12/01/2025

.. _nki-language-guide:

NKI Language Guide
==================

The Neuron Kernel Interface (NKI) language is designed for writing kernel functions to accelerate machine learning workloads on Trainium devices. This guide is an introduction to the NKI language and the key concepts you will need to know to program in NKI effectively.

Let us start by looking at a simple NKI function.

.. code-block:: python

   @nki.jit
   def my_function(x : tensor, y : tensor) -> tensor:
     assert x.shape == y.shape, "expecting tensors of the same shape"
     assert x.dtype == y.dtype, "expecting tensors with the same element type"
     
     # allocate an output tensor for the result
     output = nki.language.ndarray(x.shape, x.dtype, buffer=sbuf)
     
     print(f"adding tensors of type {x.dtype} and {x.shape}")
     nki.isa.tensor_tensor(output, x, y, op=nki.langauge.add)
     return output

The first thing you may notice about this NKI function is that it looks very much like a Python function. In fact, all NKI functions are syntactically valid Python functions. However, it is important to understand that NKI functions are not Python functions: they will be compiled by the NKI compiler and run on the Trainium accelerator. Because of this, not all Python constructs and libraries are supported within a NKI function.

The second thing to notice is that NKI has a sequential programming model. This means that the logical order of operations follows the syntactic order of the statements in the function. As you learn more about the Trainium hardware, you will see that the hardware can often do many things at the same time across the different compute engines on the Trainium devices. When we compile NKI functions, we will respect the sequential order of operations written by the programmer. The compiler may reorder operations that have no data dependencies, but this is functionally transparent to NKI programmers. Later you will see how to control which engines operations run on and even how to influence the ordering of operations with no data dependencies for better performance, but all of this is done in the context of the sequential ordering of the code.

The third thing to notice about this simple function is that is has a print statement. You may be wondering: When does this print happen? Does the Trainium hardware output a string, where does it go? What about all those different engines we just talked about and the sequential ordering? The answer to these questions reveal a very important aspect of NKI programming. The answer is that the print is evaluated by the compiler at compile time, not at runtime. So, when you compile this NKI function, the NKI compiler will output a string like:

.. code-block:: text

   adding tensors of type float16 and shape (128,512)

However, when we run this compiled function on Trainium devices they will not output anything. This is usually what you want. The compiler gives important debugging information during compilation, but when you deploy your function across 1000 Trainium devices, they will not waste any time generating debug output. Note, there is a special print function that does run on the Trainium devices, called device_print, that can be used if this is really what you need, see the API references for more information.

We have just seen that the print statement is evaluated at compile-time, and not at runtime. In fact, most things in NKI programs are evaluated at compile time. In general, calls to nki.isa.* functions will result in on-device operations, and (almost) all other things will be evaluated by the compiler at compile time. We will discuss some exceptions to this rule below, but for now it is generally the case that only the nki.isa.* calls result in run-time operations, and everything else is evaluated by the compiler at compile-time.

This leads us to our the last observation about NKI functions. The nki.isa.* APIs are the heart of the matter. These APIs are designed to expose the underlying hardware capabilities in as direct a way as possible. If you write a nki.isa function, then the hardware will execute that operation at that point in the program. The NKI language simply provides a convenient way to specify which ISA operations you want to run on your data.

In the rest of this guide we will focus on the NKI language, starting with the values you can manipulate in a NKI function. We will then cover tensor indexing, control flow, and end with a discussion of class support and interoperation with Python.

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
   size = l.count() # return number of elements in list
   third = l[2]  # get third element of list (index 2)

   # search list for a specific value
   if l.index(2):
     print("list contains 2")
     
   # remove a specific value from a list (if present)
   l.remove(1)

   # print out list in reverse order
   for x in l.reverse():
     print(x)

The NKI dictionary type is also similar to the Python version, but with the restriction that the keys must be string values.

.. code-block:: python

   d = dict() # create an empty dictionary
   d['a'] = 1 # set a value in the dictionary

   print(d.keys())  # print out keys in dictionary
   print(d.items())  # print out values in dictionary

   # print out dictionary
   for k,v in d.values():
     print(k, v)

   # remove value from dictionary if present
   if d.pop('a'):
     print("removed 'a' from dictionary")

   # fetch value of a, set to 1 if not present
   a = d.setdefault('a', default=1)

We will discuss user-defined classes later in the guide. For now, lets take a close look at the most important value in NKI, the tensor.

Tensor Values
--------------

The NKI tensor type is a representation of an on-chip tensor. That is, a value of type tensor is really a reference to some region of memory on the Trainium device at runtime. At compile-time, we do not yet know the precise location nor the precise contents of this tensor, and therefore, code evaluated at compile-time will not be able to query the precise location nor the contents. At compile-time we can only query meta-data about the tensor, such as its shape and element type. Each tensor value supports the following meta-data as (read-only) fields:

* t.dtype - The element type of the tensor, e.g. "float16"
* t.shape - The shape of the tensor, e.g. (128,64,64)
* t.address - The (virtual) address of the tensor (discussed below)
* t.offset - The access pattern offset (discussed below)
* t.pattern - The access pattern (discussed below)
* t.buffer - The memory buffer this tensor lives in (discussed below)

The most commonly used fields are dtype and shape. We have already seen an example of using these fields to check that argument tensors are compatible in our simple example. Another common case is using a dimension of a shape to iterate over a tensor:

.. code-block:: python

   # assume t is a 3-dimensional tensor, we can iterate over the
   # 2-D subtensors
   for i in range(t.shape[0]):
     my_function(t[i])

Note, because the shape is part of the meta-data of the tensor, the expression t.shape[0] is a compile-time constant. Therefore, the bounds of the for-loop are known at compile time. The compiler will unroll this loop into a sequence of calls to my_function, one for each subtensor of t.

Creating Tensors
-----------------

The easiest way to create tensors is using the nki.language.ndarray API. This function takes a shape, a dtype, and a memory type, and creates a reference to a memory region in the given memory type large enough to hold the tensor.

.. code-block:: python

   # A matrix of 128x128 16-bit float values in the SBUF memory
   t = nl.ndarray((128,128), nl.float16, nl.sbuf)
   assert t.shape = (128,128)
   assert t.dtype == nl.float16
   assert t.buffer == nl.sbuf

You can also create a tensor from an existing tensor using the reshape method. The reshape method will create a new reference to the same memory with a different shape.

.. code-block:: python

   # create an alternate view of t with shape 128x2x64
   u = t.reshape((128,2,64))

   # create an alternate view of t with shape 128x32
   v = t.reshape((128,32))

When using reshape the new tensor must use the same or less memory than the original tensor. So, the tensor v, defined above, corresponds to one quarter of the original tensor t.

Creating Tensors (the hard way)
--------------------------------

The function nl.ndarray is an easy way to create tensors that covers the most common cases. For more precise control, you can also create tensors by first defining a memory region, and then creating a view of the memory region. There are several memory regions you can choose, but we will focus on the SBUF region, the most common case. To create a memory region, we start with an existing memory region and define which part of the existing region we want to use. The special region sbuf refers to the entire device SBUF memory, so we can start with that. Once we have a memory region, we can create a tensor by calling the view method.

.. code-block:: python

   # create a memory region in the SBUF of size 128x64 bytes
   region = sbuf.ptr(size=(128, 64))

   # create a tensor of size 128x32 with float16 elementes
   t = region.view(nl.float16, (128, 32))

Note, that the combination of ptr and view is similar to ndarray. In fact, this is what ndarray is, a view of a region that is just large enough to fit the desired tensor. In fact, you can pass a region directly to ndarray if you like, as long as it is big enough to hold the resulting tensor.

.. code-block:: python

   # equivalent to region.view above
   t = nl.ndarray((128,32), nl.float16, buffer=region)

So far, we haven't done anything that we couldn't do with ndarray. However, the ptr method has another argument, offset which lets us specify the (relative) offset of the region. Tensors built this way are known as "allocated tensors," because we have given the compiler some direction about how to allocate the tensors.

.. code-block:: python

   # create a tensor at offset 128 bytes from the beginning of the SBUF memory.
   region = sbuf.ptr(size=(128,64), offset=(0,128))
   t = region.view(nl.float16, (128,32))

Note, the offset is a virtual offset. This will be the location of the tensor relative to the overall memory assigned to your kernel function by the compiler. This is useful if you want to control the relative location of two tensors. For example, to create two tensors that are right next to each other in the SBUF, you could use:

.. code-block:: python

   region1 = sbuf.ptr(size=(128,64), offset=(0,0))
   region2 = sbuf.ptr(size=(128,64), offset=(0,64))

   t1 = region1.view(nl.float16, (128,32))
   t2 = region2.view(nl.float16, (128,32))

There is actually another way to achieve the same result, but using multiple views of a single region:

.. code-block:: python

   region = sbuf.ptr(size=(128,128))

   region1 = region.ptr(size=(128,64), offset=(0,0))
   region2 = region.ptr(size=(128,64), offset=(0,64))

   t1 = region1.view(nl.float16, (128,32))
   t2 = region2.view(nl.float16, (128,32))

In the above, we first create a region large enough to hold both tensors. Then we create two regions inside of the first region which each take up half of the space. Then, we create our two tensors in these regions. The main difference between the first and the second approach is that in the first approach, the two tensors have a fixed address relative to the rest of the memory of the kernel. In the second approach, the two tensors have a fixed address relative to each other, but not to the rest of the memory of the kernel. The region offset may be changed by the compiler, because it is not specified, but the offsets of region1 and region2, within region are fixed.

As a final note on creating tensors, you may have noticed that the lower-level creation routines allow you to create two tensors in the same memory region with different shapes, as long as they both fit in the memory. For example:

.. code-block:: python

   region2 = region.ptr(size=(128,64))

   # t1 and t2 use the same underlying memory
   t1 = region.view(nl.float16, (128,32))
   t2 = region.view(nl.float16, (128,2,16))

In fact, the tensor reshape method is just a short-hand notation for view:

.. code-block:: python

   # this is just a short-hand
   u = t.reshape(shape)

   # for this
   u = t.address.reshape(t.dtype, shape)

This is a common theme with the NKI tensor creation APIs: there are several nice convenience functions available, but everything can be achieved with the more primitive ptr and view methods.

Tensor Indexing
----------------

In the previous section we noted that there are six read-only fields you can query on a tensor value. We discussed four of them, but not offset or pattern. These last two fields are related to tensor indexing. Before we talk about these fields, first lets look at the most common way of indexing tensors using integers and slices.

Suppose you have a tensor t with shape 64x64x64 that is in the SBUF memory. The SBUF memory is a two dimensional memory, so the underlying storage for this 3-D tensor is a 2-D region of the SBUF. Recall, in the SBUF, the first dimension is called the partition dimension and the second dimension if called the free dimension. By convention, the first dimension of a tensor always corresponds to the partition dimension, and the remaining dimension are layed out in the free dimension. Note, this is a change from NKI Beta 1 where the partition dimension could be mapped to any dimension of the tensor. The first dimension is always the partition dimension. Therefore, in our example, we have 64 partitions, each with 64*64=4096 elements.

We can refer to specific elements of the tensor using an index expression.

.. code-block:: python

   # 10th element in partition 0
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
   print(u.pattern)

For advanced use cases, the hardware access pattern can be specified directly.

.. code-block:: python

   # Specify HW access pattern directly
   u = t.ap(offset = 0, pattern = [...])

For more details on hardware access patterns, see the architecture guide.

Control Flow
-------------

NKI supports basic control flow constructs, including if-statements, for-loops over ranges, lists or tuples, and while loops. All of these constructs work similarly their equivalents in Python. For example, the code below uses a simple loop with an nested if statement to process the even and odd elements of a list differently.

.. code-block:: python

   def kernel(outputs, inputs):
     for i in range(len(inputs)):
       if i % 2 == 0:
         nki.isa.nc_transpose(dst=outputs[i], data=inputs[i])
       else:
         nki.isa.reciprocal(dst=outputs[i], data=inputs[i])

The loop and if-statement above will ultimately be evaluated by the NKI compiler. This means the the ISA instructions will be output to the final executable as a linear sequence. For example, suppose we call kernel with these arguments.

.. code-block:: python

   kernel([a,b,c], [x,y,z])

where a,b,c and x,y,z are tensors. Then, this call is equivalent to the code:

.. code-block:: python

   nki.isa.nc_transpose(dst=a, data=x)
   nki.isa.reciprocal(dst=b, data=y)
   nki.isa.nc_transpose(dst=c, data=z)

We will see in the next section how to write loops that run on the Trainium hardware. First, let's look at some more common uses of control flow in NKI kernels. In addition to the standard range, NKI for-loops can also use:

.. code-block:: python

   for i in sequential_range(...): ...
   for i in static_range(...): ...
   for i in affine_range(...): ...

These special range function serve as hints to the compiler. They do not change the meaning the loop: the result comptued by the different loops are all equivalent to each other, and to the basic range loop. However, these hints can improve performance in some cases. See the reference manual for more details on these loop hints.

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

   count = nki.isa.register_alloc(count_tensor)
   for i in dynamic_range(count):
     process_tensor(t[i])

The loop above uses a register value as the upper bound. This register is initialized with the register_alloc function, which can take a SBUF tensor as an argument. In this case register_alloc will load a value from the SBUf tensor count_tensor and store it in the register count. The for loop will then execute count times.

There are four register APIs that can be used to create, and load and store values to and from registers.

.. code-block:: python

   # allocate a new register with initial value
   # either from constant integer, or a SBUF tensor
   def register_alloc(x: int | tensor) -> register: ...

   # store a constant integer into a register
   def register_move(dst: imm: int): ...

   # load a value from an SBUF tensor into a register
   def register_load(dst: register, src: tensor): ...

   # store the value of a register into an SBUF tensor
   def register_store(dst: tensor, src: register): ...

Using the APIs above, we can also create dynamic while loops. A dynamic while loop is specified using the standard while-loop with a condition that is a single register value. The NKI compiler will preserve while loops with register conditions, and not unroll them.

.. code-block:: python

   # suppose cond is an SBUF tensor, perhaps declared as
   cond = nl.ndarray((1, 1), buffer=nl.shared_hbm, dtype=np.int32)

   # allocate a register with initial value 1
   reg = register_alloc(1)

   # This while loop is dynamic because the condition is a register
   while reg:
     # perform a calculation that updates cond
     ...
     nl.store(dst=cond[0], ...)
     # update register used in while-loop condition
     register_load(reg, cond)

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

   # prints 1, True
   print(c.x, c.y)

The @dataclass decorator is optional; classes with and without the @dataclass decorator will be compiled in the same way by the NKI compiler. The compiler will create the initializer functions __init__ and __post_init__, if they are not provided by the user. For the class above, the default initializers are:

.. code-block:: python

   # default if not provided by the user
   def __init__(self, x = None, y = False):
     self.x = x
     self.y = y
     self.post_init()

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
     else if e == E.y: ...
     else if e == E.z: ...
     
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
