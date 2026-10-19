.. _how-to-scheduling-apis:

.. meta::
   :description: Learn how to use NKI Scheduling APIs to control automatic instruction scheduling by adding dependency edges and using no-reorder blocks.
   :keywords: NKI, scheduling, instruction scheduling, dependency edges, no-reorder, Neuron Kernel Interface
   :date-modified: 02/20/2026

How to Use the NKI Scheduling APIs
==================================

Learn how to control instruction execution order in your NKI kernels using scheduling APIs. This guide demonstrates how to add dependency edges between instructions with ``with_schedule()`` and how to use ``no_reorder`` blocks to prevent automatic instruction reordering, giving you fine-grained control over kernel performance optimization.

About the NKI Scheduling APIs
-----------------------------

The NKI Scheduling APIs provide additional control over automatic instruction scheduling. This control comes in the form of adding additional dependency edges before scheduling. The extra dependency edges constrain the reordering that the automatic scheduler will do.

Adding dependency edges
-----------------------

Below is an example showing how to specify the scheduling metadata for NKI kernel functions. The extra dependency edges are communicated to the NKI compiler by setting a property on the top-level kernel functions with the scheduling edges.

.. code-block:: python

   @nki.jit()
   def kernel(t):
       x = nl.ndarray(t.shape, t.dtype, buffer=nl.sbuf)
       nisa.dma_copy(dst=x, src=t)
       
       a = nl.ndarray(t.shape, t.dtype, buffer=nl.sbuf)
       b = nl.ndarray(t.shape, t.dtype, buffer=nl.sbuf)
       c = nl.ndarray(t.shape, t.dtype, buffer=nl.sbuf)
       
       nisa.reciprocal(dst=a, data=x, name="recip")
       nisa.tensor_scalar(dst=b, data=x, op0=nl.add, operand0=1, name="plus1")
       nisa.tensor_tensor(dst=c, data1=a, data2=b, op=nl.add)
       
       out = nl.ndarray(t.shape, t.dtype, buffer=nl.hbm)
       nisa.dma_copy(dst=out, src=c)
       
       return out

   # The named statements "recip" and "plus1" could execute in any order.
   # We can fix the order with a runtime dependency.
   # This property can be a list of pairs of instruction names.
   # Each pair is a set of dependency edges.
   # In this case "plus1" depends on "recip", and so will execute second.
   scheduled = kernel.with_schedule([
       ("plus1", "recip")
   ], kind=nisa.dependency.runtime)

   # The second component of each pair can be a single name or a list of names
   # This is equivalent to above. Using a list is convenient
   # for declaring multiple dependency edges.
   scheduled = kernel.with_schedule([
       ("plus1", ["recip"])
   ], kind=nisa.dependency.runtime)

The NKI compiler will collect the data from the ``with_schedule`` call, check that it makes sense, and propagate it to the scheduling pass of the compiler. Below is a more complicated example with programmatic meta-data generation. In this example, we will enforce a sequential order for all of the activation operations.

.. code-block:: python

   # compute exp on three tiles and return the result tiles
   @nki.jit
   def kernel(a, b, c):
       in_tiles = []
       for inp in (a,b,c):
           in_tile = nl.ndarray(inp.shape, inp.dtype, buffer=nl.sbuf)
           nisa.dma_copy(dst=in_tile, src=inp)
           in_tiles.append(in_tile)
       
       out_tiles = []
       for i in range(len(in_tiles)):
           tile = in_tiles[i]
           out_tile = nl.ndarray(tile.shape, tile.dtype, buffer=nl.sbuf)
           nisa.activation(dst=out_tile, data=tile, op=nl.exp, name=f"act{i}")
           out_tiles.append(out_tile)
       
       outs = []
       for tile in out_tiles:
           out = nl.ndarray(tile.shape, tile.dtype, buffer=nl.hbm)
           nisa.dma_copy(dst=out, src=tile)
           outs.append(out)
       
       return tuple(outs)

   # The activations have no data dependencies, and could execute in any order.
   # Make them execute serially by building a list of pairs of edges
   # act1 depends on act0
   # act2 depends on act1
   l = []
   for i in range(1,3):
       l.append((f"act{i}", f"act{i-1}"))
   
   # attach the dependencies to the kernel by calling with_schedule
   scheduled = kernel.with_schedule(l, kind=nisa.dependency.runtime)

Choosing a dependency kind
--------------------------

Both ``with_schedule()`` and ``NkiInstruction.depends_on()`` take a ``kind`` argument that decides how the edge is enforced. The two kinds cost different amounts and guarantee different things, so pick the weakest one that expresses what your kernel needs.

``nisa.dependency.runtime`` lowers to a hardware semaphore wait. The consumer does not begin until the producer has finished, which holds across engines. Because that wait fires unconditionally, the producer and the consumer must sit at the same dynamic-loop nesting, or the producer in an enclosing scope. Use it when the consumer genuinely must observe the producer's results.

``nisa.dependency.schedule`` emits no hardware synchronization at all. It only constrains the compiler's schedulers: the producer is issued before the consumer, but the consumer may start on the same cycle, so instructions on different engines can still overlap. The producer must appear before the consumer in the program. Use it to steer instruction order for performance, when no data actually flows between the two instructions.

``kind`` sets the default for the whole call. To mix kinds in one schedule, add the kind as a third element on the edges that need a different one:

.. code-block:: python

   scheduled = kernel.with_schedule([
       ("plus1", "recip", nisa.dependency.runtime),
       ("plus2", "recip2", nisa.dependency.schedule),
   ])

An edge that ends up with no kind, from neither its own third element nor ``kind``, preserves the previous release's behavior and is deprecated; a future release will require one.

Using no_reorder
----------------

Adding dependency edges can be tedious. To make the process more streamlined the NKI compiler also supports no-reorder blocks. A no-reorder block is a section of code where dependency edges are automatically between every pair of instructions. Using no-reorder blocks, the example above could be written as shown below.

.. code-block:: python

   # compute exp on three tiles and return the result tiles
   @nki.jit()
   def loop(a, b, c):
       in_tiles = []
       for inp in (a,b,c):
           in_tile = nl.ndarray(inp.shape, inp.dtype, buffer=nl.sbuf)
           nisa.dma_copy(dst=in_tile, src=inp)
           in_tiles.append(in_tile)
       
       out_tiles = []
       with nl.no_reorder():
           for i in range(len(in_tiles)):
               tile = in_tiles[i]
               out_tile = nl.ndarray(tile.shape, tile.dtype, buffer=nl.sbuf)
               nisa.activation(dst=out_tile, data=tile, op=nl.exp, name=f"act{i}")
               out_tiles.append(out_tile)
       
       outs = []
       for tile in out_tiles:
           out = nl.ndarray(tile.shape, tile.dtype, buffer=nl.hbm)
           nisa.dma_copy(dst=out, src=tile)
           outs.append(out)
       
       return tuple(outs)

The ``no_reorder`` block instructs the compiler to insert dependency edges between every instruction. Note, the ``no_reorder`` block is "dynamically scoped", meaning it applies to all of the code that would execute under the block, not just the code that is syntactically under the block. For example, the following code is equivalent to the above.

.. code-block:: python

   def loop_body(i, in_tiles, out_tiles):
       tile = in_tiles[i]
       out_tile = nl.ndarray(tile.shape, tile.dtype, buffer=nl.sbuf)
       nisa.activation(dst=out_tile, data=tile, op=nl.exp, name=f"act{i}")
       out_tiles.append(out_tile)

   @nki.jit
   def loop(a, b, c):
       in_tiles = []
       for inp in (a,b,c):
           in_tile = nl.ndarray(inp.shape, inp.dtype, buffer=nl.sbuf)
           nisa.dma_copy(dst=in_tile, src=inp)
           in_tiles.append(in_tile)
       
       out_tiles = []
       with nl.no_reorder():
           for i in range(len(in_tiles)):
               loop_body(i, in_tiles, out_tiles)
       
       outs = []
       for tile in out_tiles:
           out = nl.ndarray(tile.shape, tile.dtype, buffer=nl.hbm)
           nisa.dma_copy(dst=out, src=tile)
           outs.append(out)
       
       return tuple(outs)

Notice that even though the ``loop_body`` function is not syntactically under a ``no_reorder`` block, it will be evaluated as a no-reorder block because the function is called from under a ``no_reorder`` block.
