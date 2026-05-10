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

.. _tensor-values:

Tensor Values
--------------

Tensors are the main value you operate on in a NKI kernel. An ``NkiTensor`` is a view of a region of device memory — shape, dtype, and memory buffer on the outside; a strided layout over an underlying storage on the inside. The model is the same one PyTorch and NumPy use: creation routines allocate fresh storage and hand back a tensor; view operations (``slice``, ``reshape``, ``permute``, ``view``, ``ap``, …) share that storage and return new tensors pointing at the same data.

.. note::

   Tensors represent on-device memory at runtime. At compile time the compiler knows the *shape*, *dtype*, *strides*, and *buffer* — enough to reason about layout and schedule instructions — but not the actual element values. Any Python code that touches tensor contents directly (``t[0, 0] == 5.0``, printing tensor data, ...) is a kernel operation, not a compile-time expression.

Anatomy of a tensor
~~~~~~~~~~~~~~~~~~~

An ``NkiTensor`` is fully described by a small set of attributes. All of them are available at compile time.

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - Attribute
     - Meaning
   * - ``shape``
     - Tuple of positive integers. The size of each dimension of the view. For example, ``(128, 64)``.
   * - ``strides``
     - Tuple of integers, one per dimension. The element step between consecutive indices along that dimension in the underlying storage. A stride of ``0`` means the dimension is broadcast.
   * - ``offset``
     - Element offset (not byte offset) from the start of the underlying storage at which this view begins. A freshly allocated tensor has ``offset == 0``; slicing produces views with non-zero offsets.
   * - ``dtype``
     - Element type, e.g. ``nl.float32`` or ``nl.bfloat16``. See :ref:`nki-dtype` for the full list.
   * - ``buffer``
     - The memory region the storage lives in: ``nl.sbuf``, ``nl.psum``, ``nl.hbm``, ``nl.private_hbm``, or ``nl.shared_hbm``.
   * - ``ndim``, ``size``
     - Convenience attributes: ``len(shape)`` and ``prod(shape)``.
   * - ``name``
     - Optional debug label, propagated into compiler diagnostics and :ref:`scheduling <how-to-scheduling-apis>`.

Given these attributes, the element addressed by an index tuple ``(i0, i1, …, i_{n-1})`` sits at element position::

    offset + sum_k (strides[k] * i_k)

inside the underlying storage — the classic strided scheme used by NumPy and PyTorch.

**Example.** A contiguous SBUF tensor:

.. code-block:: python

   t = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)

   assert t.shape   == (128, 64)
   assert t.strides == (64, 1)
   assert t.offset  == 0
   assert t.dtype   == nl.float32
   assert t.buffer  == nl.sbuf
   assert t.ndim    == 2
   assert t.size    == 128 * 64

The free-dim stride is ``1`` (consecutive elements in the free dimension are adjacent in memory). The partition-dim stride is ``64``.

.. note::

   On on-chip tensors (SBUF, PSUM), the *first* dimension is the **partition dimension**. It maps to the NeuronCore's parallel partitions. Most view primitives are not allowed to alter the partition dimension — see individual method docs for exact rules. HBM tensors have no partition dim and can be reshaped freely.

Views share storage
~~~~~~~~~~~~~~~~~~~

View primitives return a new ``NkiTensor`` that points into the *same* storage as the input — no data is copied. Writing through one view is visible through any other view of the same storage.

.. code-block:: python

   t = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
   u = t.reshape((128, 8, 8))   # same storage, different shape
   v = t[:, 16:32]              # same storage, sliced view

   # Writing to u is observable through t (and v, where they overlap):
   nisa.memset(dst=u, value=1.0)
   # t[:, 16:32] now reads 1.0

Taking a view is free at runtime (no copy, no allocation). Taking a view of a view is also free — views compose.

Querying layout
~~~~~~~~~~~~~~~

Two predicates describe common layout questions:

* :meth:`NkiTensor.is_contiguous` — does this view cover its storage in dense row-major order? A fresh ``nl.ndarray`` is contiguous; ``reshape`` and ``view(dtype)`` require a contiguous view; ``permute`` produces a non-contiguous view.

* :meth:`NkiTensor.is_indirect` — does this view use runtime-resolved (dynamic) addressing? Indirect views are produced by :meth:`select` with a tensor index, :meth:`vector_select`, or :meth:`ap` with ``scalar_offset`` / ``vector_offset``. Once a view is indirect, the indirected dimension cannot be sliced or selected again — check ``is_indirect()`` to guard downstream logic.

For completeness, :meth:`NkiTensor.get_pattern` returns the view's layout as the ``[[stride, count], …]`` pairs that ``.ap()`` accepts — useful as a starting point when composing a new ``.ap()`` that reuses most of the current layout.

View primitives
~~~~~~~~~~~~~~~

The view primitives below all return a new ``NkiTensor`` sharing the same storage. They correspond closely to PyTorch / NumPy equivalents, with additional constraints where the Trainium hardware requires them. Full signatures and constraints live in the :doc:`NkiTensor API reference <../api/generated/nki.language.NkiTensor>`.

**Slicing (Python ``[...]`` syntax)** — Python indexing produces a view. Integer, slice, ellipsis, and tuple keys are supported.

.. code-block:: python

   t = nl.ndarray((128, 64, 32), dtype=nl.float32, buffer=nl.sbuf)
   b = t[:, 0:16, :]     # shape (128, 16, 32)
   c = t[..., ::2]       # shape (128, 64, 16) — every other element
   d = t[:, :, 0]        # shape (128, 64) — integer index on free dim drops it

On-chip tensors (SBUF, PSUM) are always at least 2-D — the partition dim is never removed. Integer indexing on the partition dim therefore keeps it as size 1 rather than dropping it, in the same way that NumPy and PyTorch keep a scalar selection from a 1-D array at 1-D rather than reducing to 0-D when the backing layout requires it:

.. code-block:: python

   t = nl.ndarray((128, 64, 32), dtype=nl.float32, buffer=nl.sbuf)
   t[0, :, :]            # shape (1, 64, 32) — partition dim stays at size 1
   t[0:1, :, :]          # shape (1, 64, 32) — same

HBM tensors have no partition dim and drop the dim as usual:

.. code-block:: python

   h = nl.ndarray((4, 128, 32), dtype=nl.float32, buffer=nl.shared_hbm)
   h[0, :, :]            # shape (128, 32) — outer dim dropped

See :ref:`tensor-indexing` for the full indexing rules.

**Single-dim slice with step** — :meth:`NkiTensor.slice` is the verbose form of the Python slice, equivalent to ``t[..., start:end:step, ...]``:

.. code-block:: python

   t = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
   u = t.slice(dim=1, start=8, end=24)   # shape (128, 16)
   v = t.slice(dim=1, start=0, end=64, step=2)  # shape (128, 32)

**Reshape** — :meth:`NkiTensor.reshape` replaces the shape without moving data; the total element count is preserved.

.. code-block:: python

   t = nl.ndarray((128, 4, 6), dtype=nl.float32, buffer=nl.sbuf)
   t.reshape((128, 24))    # merge the two trailing dims
   t.reshape((128, 2, 12)) # split differently

Like :meth:`torch.Tensor.view`, ``reshape`` only succeeds when the new shape is compatible with the current strides — each new dimension must either correspond to an existing dimension, or span several existing dimensions that are already contiguous in memory. If the layout cannot be expressed as a view, ``reshape`` fails; materialise a contiguous copy first (e.g. via :func:`nl.copy`). On-chip tensors additionally require ``shape[0] == t.shape[0]`` so the partition dimension is preserved.

:meth:`NkiTensor.flatten_dims` and :meth:`NkiTensor.reshape_dim` are targeted versions that merge / split a single dim without touching the others. Because their operation is localised, they succeed on many views where a full ``reshape`` does not.

**Permute** — :meth:`NkiTensor.permute` reorders dimensions.

.. code-block:: python

   t = nl.ndarray((128, 4, 8), dtype=nl.float32, buffer=nl.sbuf)
   t.permute((0, 2, 1))   # shape becomes (128, 8, 4)

The partition dim (dim 0) must remain at position 0 on on-chip tensors. Permuting free dims leaves the view non-contiguous along the permuted axes: consecutive elements in the new layout are no longer adjacent in memory. Targeted view operations (``reshape_dim``, ``flatten_dims`` within a still-contiguous range, ``slice``, ``select``) continue to work; a full :meth:`NkiTensor.reshape` across permuted dims does not, and needs a contiguous copy first.

**Broadcast** — :meth:`NkiTensor.broadcast` expands a size-1 dimension without copying data. The underlying stride becomes ``0`` so every "repeated" index maps to the same storage element.

.. code-block:: python

   t = nl.ndarray((128, 1, 64), dtype=nl.float32, buffer=nl.sbuf)
   t.broadcast(dim=1, size=8)   # shape (128, 8, 64), no data copied

**Add / remove size-1 dims** — :meth:`NkiTensor.expand_dim` inserts a size-1 dim at the given position; :meth:`NkiTensor.squeeze_dim` removes one.

.. code-block:: python

   t = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
   u = t.expand_dim(1)       # shape (128, 1, 64)
   v = u.squeeze_dim(1)      # back to (128, 64)

**Select** — :meth:`NkiTensor.select` picks a single index along a dimension and removes that dimension. The index can be an integer (static, resolved at compile time) or an SBUF tensor (dynamic, resolved at runtime).

.. code-block:: python

   t = nl.ndarray((128, 8, 64), dtype=nl.float32, buffer=nl.sbuf)
   s = t.select(dim=1, index=3)   # shape (128, 64), static

   # Dynamic select via a scalar SBUF tensor:
   hbm_t = nl.ndarray((4, 128, 8), dtype=nl.float32, buffer=nl.shared_hbm)
   idx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
   hbm_t.select(dim=0, index=idx) # shape (128, 8), dynamic

Dynamic select marks the view as indirect — see :meth:`NkiTensor.is_indirect`.

**dtype reinterpret** — :meth:`NkiTensor.view` reinterprets the storage bits as a different dtype. The last dimension rescales by the ratio of dtype sizes.

.. code-block:: python

   t = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
   t.view(nl.int32)     # same shape, bits reinterpreted (same size)
   t.view(nl.uint8)     # shape becomes (128, 256), 4× expansion

**einops-style rearrange** — :meth:`NkiTensor.rearrange` combines split, permute, and merge in a single named operation:

.. code-block:: python

   t = nl.ndarray((128, 24), dtype=nl.float32, buffer=nl.sbuf)
   t.rearrange(('b', ('h', 'w')), ('b', 'w', 'h'), {'h': 4})
   # dim 1 split into (h=4, w=6), then re-ordered: shape (128, 6, 4)

**Other view methods** — :meth:`NkiTensor.vector_select`, :meth:`NkiTensor.indirect`, and :meth:`NkiTensor.ap` are advanced primitives for gather patterns and hardware-native access patterns — see the API reference and the :doc:`NKI Access Patterns deep-dive <../deep-dives/nki-aps>` for details.

Composition
~~~~~~~~~~~

View primitives return ``NkiTensor`` and every primitive accepts any ``NkiTensor``, so they compose freely:

.. code-block:: python

   # Start from a 3-D SBUF tile, extract the partition window [0:64],
   # permute the two free dims, then pick a single column:
   t = nl.ndarray((128, 4, 16), dtype=nl.float32, buffer=nl.sbuf)
   u = t[0:64, :, :].permute((0, 2, 1))[:, 0, :]
   assert u.shape == (64, 4)

Each step costs nothing at runtime — the result is a new tensor whose shape/strides/offset describe the composed view directly.

The escape hatch: ``ap``
~~~~~~~~~~~~~~~~~~~~~~~~

When the high-level primitives cannot express the layout you need, :meth:`NkiTensor.ap` is the low-level escape hatch. It takes an explicit hardware access pattern — a list of ``[stride, count]`` pairs, leading pair describing the partition dimension — and returns a new view over the same storage. This is analogous to PyTorch's ``torch.as_strided``.

.. code-block:: python

   t = nl.ndarray((128, 1024), dtype=nl.float16, buffer=nl.sbuf)
   # Access every other element in the free dimension.
   # Partition stride is 1024 (storage's free-dim count), NOT 1.
   u = t.ap(pattern=[(1024, 128), (2, 512)])
   assert u.shape == (128, 512)

The partition stride must equal the storage's free-dim element count so that every partition performs the same access in parallel — a hardware rule for on-chip (SBUF / PSUM) tensors. HBM tensors have no partition dimension and accept any pattern. When you already have a tensor whose layout you want to start from, :meth:`NkiTensor.get_pattern` returns a pattern you can mutate as input to ``.ap()``:

.. code-block:: python

   pattern = t.get_pattern()
   pattern[0][1] = 64          # read 64 partitions instead of 128
   u = t.ap(pattern=pattern)   # same layout, partition count overridden

``offset`` defaults to ``None``, which means "inherit the current view's storage offset" (the same convention as ``torch.as_strided``'s ``storage_offset=None``). Pass an explicit integer to override.

Further details — nested indexing semantics, reinterpret cast with ``dtype=``, dynamic access with ``scalar_offset`` and ``vector_offset`` — are documented in :doc:`NKI Access Patterns <../deep-dives/nki-aps>`.

.. warning::

   Prefer the higher-level view primitives (``slice``, ``reshape``, ``permute``, ``broadcast``, ...) over ``.ap()`` — similar to the guidance on ``torch.as_strided``. ``.ap()`` is **not composable**: the pattern addresses the tensor's underlying storage directly, ignoring the shape, strides, and offset of the view it is called on. ``t.slice(...).ap(pattern=...)`` and ``t.ap(pattern=...)`` produce the same result.

   Reach for ``.ap()`` only when the layout you need cannot be expressed as a composition of higher-level view primitives.

Creating Tensors
-----------------

The easiest way to create tensors is using the ``nki.language.ndarray`` API. This function takes a shape, a dtype, and a memory type, and returns an ``NkiTensor`` representing a reference to a memory region in the given memory type large enough to hold the tensor.

.. note::

   ``ndarray`` does **not** initialize memory. The contents of a newly allocated tensor are undefined until explicitly written to (e.g., via ``nisa.dma_copy`` or ``nisa.memset``).

.. code-block:: python

   # A matrix of 128x128 16-bit float values in the SBUF memory
   t = nl.ndarray((128,128), nl.float16, nl.sbuf)
   assert t.shape == (128,128)
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

.. _tensor-indexing:

Tensor Indexing
----------------

We already saw in :ref:`Tensor Values <tensor-values>` that every ``NkiTensor`` has a ``shape``, ``strides``, ``offset``, and ``buffer``. Here we look in detail at the most common way of producing new views of a tensor — Python-style indexing with integers and slices.

Suppose you have an SBUF tensor ``t`` with shape ``(64, 64, 64)``. By convention the first dimension is the partition dimension and the remaining dimensions lay out the free dimension of each partition. You can refer to sub-tensors with an index expression.

.. code-block:: python

   t = nl.ndarray((64, 64, 64), dtype=nl.float32, buffer=nl.sbuf)

   # On-chip tensors stay at least 2-D: integer indexing on the partition
   # dim keeps it at size 1, so t[0,0,10] is a (1,1) view, not a scalar.
   u = t[0, 0, 10]
   assert u.shape == (1, 1)

   # Integer indexing on a free dim drops that dim — unless dropping would
   # make the result < 2-D, in which case the last free dim is kept at 1.
   u = t[:, 0]
   assert u.shape == (64, 64)

   u = t[:, 0, 10]
   assert u.shape == (64, 1)     # last dim kept at size 1 to stay ≥ 2-D

For larger sub-tensors use Python slice expressions — ``start:stop:step`` — or the ellipsis ``...`` for "defaults for a range of dimensions".

.. code-block:: python

   # All first 64 elements of every partition
   u = t[0:64, 0, 0:64]
   assert u.shape == (64, 64)

   # Same as above, using defaults
   u = t[:, 0, :]
   assert u.shape == (64, 64)

   # Only the even elements of the third dimension
   u = t[:, :, ::2]
   assert u.shape == (64, 64, 32)

.. code-block:: python

   # The whole tensor t
   u = t[...]
   assert u.shape == (64, 64, 64)

   # Same
   u = t[:, ...]
   assert u.shape == (64, 64, 64)

   # Use defaults for the inner dimensions; partition index 0 is kept at
   # size 1 because on-chip tensors stay ≥ 2-D.
   u = t[0, ..., :]
   assert u.shape == (1, 64, 64)

Every indexing expression returns a new ``NkiTensor`` sharing storage with ``t``. That means you can chain indexing, query the result's shape, strides, offset, and pattern, and pass it to any NKI ISA instruction that accepts a tensor.

.. code-block:: python

   u = t[0, ...]
   assert u.shape == (1, 64, 64)

   v = u[:, 0:32, :]
   assert v.shape == (1, 32, 64)

   # All attributes are available at compile time:
   print(u.shape, u.strides, u.offset)
   print(u.get_pattern())       # [[stride, count], ...]

For the rare case where the layout you need cannot be expressed via slicing or any of the other view primitives, :meth:`NkiTensor.ap` is the low-level escape hatch. See :doc:`NKI Access Patterns <../deep-dives/nki-aps>`.

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
