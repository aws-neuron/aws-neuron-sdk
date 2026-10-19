.. meta::
   :description: Deep dive into NKI loop constructs with specialization-time or runtime bounds on AWS Neuron hardware.
   :keywords: NKI, loop, fori_loop, while_loop, compiled loop, hardware loop, unroll, runtime bounds, VirtualRegister, NkiScalar, AWS Neuron, Trainium
   :date-modified: 10/08/2026

.. _nki-loops:

=========
NKI Loops
=========

This document covers the differences between specialization-time Python loops and NKI compiled loops.

.. contents:: On this page
   :local:
   :depth: 2

Choosing a Loop
---------------

NKI provides loop constructs that are unrolled eagerly at specialization-time, and loop constructs that run on device. For specialization-time loops, NKI provides Python ``for`` and ``while`` loops with specialization-time iteration bounds. For compiled loops, NKI provides ``nki.language.fori_loop`` and ``nki.language.while_loop`` with specialization-time or dynamic iteration bounds. The table below summarizes the key differences between loop constructs:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15

   * - Loop Construct
     - Behavior
     - When to Use
   * - Python ``for`` loops
     - Unrolled at specialization time
     - Small iteration count known at specialization time
   * - Python ``while`` loops
     - Unrolled at specialization time
     - Small iteration count, with a condition known at specialization time
   * - :func:`nki.language.fori_loop`
     - Compiled loop
     - Large iteration count, or iteration count unknown at specialization time
   * - :func:`nki.language.while_loop`
     - Compiled loop
     - Stopping condition known only within the loop, or unknown at specialization time


Earlier versions of NKI used range iterators instead of loop constructs. NKI provided three specialization-time range iterators and one dynamic range iterator.

In NKI 0.7 and later, the static range iterators :func:`nki.language.affine_range`, :func:`nki.language.sequential_range`, and :func:`nki.language.static_range` are **deprecated** and default to Python ``range``. The dynamic range iterator :func:`nki.language.dynamic_range` and Python ``while`` loops with a ``VirtualRegister`` condition are **unsupported**. Use NKI loop constructs instead.

.. note::

   To use the dynamic range iterator or Python ``while`` loops with a ``VirtualRegister`` condition, you must opt in to the NKI meta-programming language from releases before 0.7 by setting the environment variable ``NKI_COMPILER_FRONTEND=custom``. This option will be removed in a future version of NKI. For more information, see :ref:`nki-migrate-0-6-to-0-7`.

The table below shows how each loop construct and range iterator from before NKI 0.7 maps to its equivalent in NKI 0.7 and later:

.. list-table::
   :header-rows: 1
   :widths: 20 20 15 15

   * - Before NKI 0.7
     - NKI 0.7 and later
     - Behavior
     - When to Use
   * - Python ``for`` loops
     - Python ``for`` loops (unchanged)
     - Unrolled at specialization time
     - Small iteration count known at specialization time
   * - :func:`nki.language.affine_range` (alias for Python ``range``)
     - Python ``for`` loops
     - Unrolled at specialization time
     - Small iteration count known at specialization time
   * - :func:`nki.language.sequential_range` (alias for Python ``range``)
     - Python ``for`` loops
     - Unrolled at specialization time
     - Small iteration count known at specialization time
   * - :func:`nki.language.static_range` (alias for Python ``range``)
     - Python ``for`` loops
     - Unrolled at specialization time
     - Small iteration count known at specialization time
   * - :func:`nki.language.dynamic_range`
     - :func:`nki.language.fori_loop`
     - Hardware loop, one copy of the body
     - Iteration count unknown at specialization time
   * - Python ``while`` loops
     - Python ``while`` loops (unchanged)
     - Unrolled at specialization time
     - Small iteration count, with a condition known at specialization time
   * - Python ``while`` loops with a ``VirtualRegister`` condition
     - :func:`nki.language.while_loop`
     - Hardware loop, one copy of the body
     - Iteration count unknown at specialization time


Unrolled Loops
--------------

During specialization, Python ``for`` and ``while`` loops are fully unrolled, turning the loop bodies into straight-line code. Loop unrolling is a transformation technique designed to improve performance at the cost of code size. Unrolling loops helps the compiler eliminate redundant instructions, reducing overhead.

Consider an example using Python loops that print the values from 0 to 4:

.. code-block:: python

   # First loop, traditional Python for loop.
   for i in range(3):
       print(i)

   # Second loop, traditional Python while loop.
   j = 3
   while j < 5:
       print(j)
       j += 1


After specialization, the loops unroll to the following Python code:

.. code-block:: python

   print(0)
   print(1)
   print(2)
   print(3)
   print(4)


Implications of Loop Unrolling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although loop unrolling can reduce overhead by converting loop bodies into straight-line code, loops over a large number of iterations can greatly increase compile time --- each iteration inlines a copy of the loop body, which NKI must trace. For loops with many iterations, consider using compiled loops ``nki.language.fori_loop`` or ``nki.language.while_loop`` instead.

.. code-block:: python

   # Loop over a very large number of iterations.
   for i in range(1_000_000):
      nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=1.0)


Loop unrolling can also generate many instructions, which leads to larger NEFFs that take longer to load. For example, after specialization, the loop above unrolls into 1 million instructions in the NEFF:

.. code-block:: python

   nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=1.0)
   nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=1.0)
   # ... many nisa.tensor_scalar instructions
   nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=1.0)
   nisa.tensor_scalar(dst=acc, data=acc, op0=nl.add, operand0=1.0)


As a rule of thumb, use Python loops for small iteration counts, where straight-line code helps performance, and compiled loops for large or unknown iteration counts, where unrolling increases compile time and NEFF size.


Compiled Loops
--------------

The body of a compiled loop is traced once, during specialization, into an NKI IR loop. The compiler lowers the NKI IR loop to NeuronISA. NKI offers two types of compiled loops: ``nki.language.fori_loop`` and ``nki.language.while_loop``.


Using nki.language.fori_loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``nki.language.fori_loop(lower, upper, body_fun, step=1)`` runs a counted loop, similar to ``for i in range(lower, upper, step)``. Instead of a loop body, you pass a function, ``body_fun``, that takes the loop variable ``i`` and returns nothing. NKI traces ``body_fun`` once. The following example uses ``nki.language.fori_loop``:

.. code-block:: python

   import numpy as np

   import nki
   import nki.isa as nisa
   import nki.language as nl


   @nki.jit
   def sum_rows_fori(data, count):
       ROWS, COLS = data.shape

       # Read the run-time row count into a scalar to use as the loop bound.
       count_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
       nisa.dma_copy(dst=count_sb, src=count)
       n = nisa.scalar_load(count_sb)

       total = nl.zeros((1, COLS), dtype=nl.float32, buffer=nl.sbuf)
       row = nl.ndarray((1, COLS), dtype=nl.float32, buffer=nl.sbuf)

       def body(i):
           # The loop variable selects row i of data at run time.
           nisa.dma_copy(
               dst=row,
               src=data.ap([[COLS, 1], [1, COLS]], scalar_offset=i, indirect_dim=0),
           )
           nisa.tensor_tensor(dst=total, data1=total, data2=row, op=nl.add)

       nl.fori_loop(0, n, body)

       out = nl.ndarray((1, COLS), dtype=nl.float32, buffer=nl.shared_hbm)
       nisa.dma_copy(dst=out, src=total)
       return out


   data = np.arange(32, dtype=np.float32).reshape(4, 8)
   count = np.array([[3]], dtype=np.int32)
   sum_rows_fori(data, count)
   # [[24. 27. 30. 33. 36. 39. 42. 45.]]

Parameter Constraints
~~~~~~~~~~~~~~~~~~~~~

``lower`` / ``upper``
   Can be a Python ``int``, ``NkiScalar``, **or** ``VirtualRegister`` object. When either ``lower`` or ``upper`` is a ``NkiScalar`` **or** both ``lower`` and ``upper`` are Python ``int``, the loop variable is a ``NkiScalar``. Otherwise, the loop variable is a ``VirtualRegister``.

``body_fun``
   A function that takes the loop variable and returns ``None``. Writing to the loop variable does not change the loop. To keep state across iterations, use SBUF or HBM tensors.

``step``
   **Must** be a positive Python ``int``. Passing a ``NkiScalar`` or ``VirtualRegister`` raises an ``AssertionError``.


Using nki.language.while_loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``nki.language.while_loop(init, body_fun)`` runs while a condition register is not zero, similar to ``while r != 0``. NKI checks the condition before each iteration. If ``init`` is zero, the body does not run. The body function takes the current condition register and must return the register to check on the next iteration.


Parameter Constraints
~~~~~~~~~~~~~~~~~~~~~

``init``
   **Must** be a ``VirtualRegister``. Passing a ``NkiScalar`` raises an ``AssertionError``.

``body_fun``
   A function that takes the condition register and returns the register to check on the next iteration. Returning ``None`` raises a ``TypeError``.



Interaction with ``no_reorder``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compiled loops (``nki.language.fori_loop``, ``nki.language.while_loop``) and dynamic range iterators (``nki.language.dynamic_range``) inside :func:`nki.language.no_reorder` blocks are currently not supported. The following code patterns will produce an error:

.. code-block:: python

   # Not supported: nl.fori_loop inside nl.no_reorder
   with nl.no_reorder():
       nl.fori_loop(0, n, body)

   # Not supported: nl.while_loop inside nl.no_reorder
   with nl.no_reorder():
       nl.while_loop(cond, body)
