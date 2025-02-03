.. _nki_programming_model:

NKI Programming Model
=====================

The NKI programming model enables developers to create custom kernels to program
NeuronCores, where every kernel consists of three main stages:

1. **Loading of inputs** from device memory (High Bandwidth Memory, or HBM)
   to the on-chip SRAM (State Buffer, or SBUF).
2. **Computation definition**, to be executed on the NeuronCore compute
   engines.
3. **Storing of outputs** from on-chip SRAM (SBUF) back to device memory
   (HBM).

:numref:`Fig. %s <nki-fig-pm-nc>` below is a simplified diagram of
a NeuronCore along with its attached HBM
device memory. NKI kernels in current release can target a single NeuronCore-v2
or up to two NeuronCore-v3.

As shown in :numref:`Fig. %s <nki-fig-pm-nc>`, a single NeuronCore consists
of two on-chip SRAMs (SBUF and PSUM)
and four heterogenous compute engines: the Tensor
Engine, Vector Engine, Scalar Engine, and GpSimd Engine.
For more information about the compute engine capabilities, see
:doc:`NeuronDevice Architecture Guide <nki_arch_guides>`.
Next, let's dive into the memory hierarchy design of NeuronCore,
which provides the necessary architecture knowledge to
understand the NKI programming model.

.. The Tensor Engine is based on a power-optimized systolic array, which is
.. highly efficient for tensor computations (e.g., GEMM, CONV, Reshape, and
.. Transpose). The engine supports mixed-precision computations (e.g.,
.. FP16/BF16/FP8 inputs and FP32 outputs), and also various tile sizes
.. including 32x32, 64x64, and 128x128. The Scalar Engine is optimized for
.. scalar computations, in which every element of the output is dependent
.. on only one element of input, e.g., non-linear functions such as GELU,
.. SIGMOID or EXP. The Scalar Engine is highly parallelized (128 parallel
.. lanes) and can perform a non-linear operation on up to 128 elements per
.. cycle. The Vector Engine is optimized for vector computations, in which
.. every element of output is dependent on multiple input elements.
.. Examples include ‘axpy' operations (Z=aX+Y), Layer Normalization,
.. Pooling operations, and many more. The Vector Engine also has 128
.. parallel lanes, and can perform 1024 floating point operations per
.. cycle. Lastly, the GpSimd Engine consists of eight deeply embedded,
.. fully programmable, 512-bit wide general purpose vector processors.
.. These processors can execute straight-line C-code, have direct access to
.. the embedded on-chip SRAM memories, and can synchronize directly with
.. other NeuronCore engines. Currently, NKI automatically maps instructions
.. to the engines they will be executed on.

.. In addition to the compute engines, each NeuronCore has two levels of
.. on-chip SRAM, PSUM and SBUF. The SBUF memory (stands for ‘State Buffer')
.. is the main on-chip SRAM, which is used for storing input/output tiles
.. for the different compute engines to operate on. It is software-managed
.. and used to minimize HBM accesses by maximizing data locality. The PSUM
.. memory (stands for ‘Partial Sum Buffer') is a dedicated memory for
.. storing Tensor Engine output. It is unique in its ability to
.. read-add-write into every address, thus it is useful when performing
.. large matrix multiplication (MatMult) calculations using multiple tiles
.. (see Figure 2).
.. PSUM can also be read/written by all the other engines (Scalar Engine,
.. Vector Engine, GpSimd Engine). The Tensor Engine must read its inputs
.. from SBUF and write its output to PSUM, as indicated by the directional
.. arrows in Figure 1. All other engines can read/write from either SBUF or
.. PSUM. Since PSUM is much smaller than SBUF, it is good practice to use
.. it only for storing MatMult results and evict it as soon as possible.
.. Data movement across NeuronCores is not yet supported by NKI, and will
.. be added in a future release. For further architectural details see :ref:`neuron-core-v2`.

.. _nki-fig-pm-nc:

.. figure:: img/pm-nc.png
   :align: center
   :width: 60%

   NeuronCore Architecture (multiple NeuronCores available per NeuronDevice)

.. _nki-pm-memory:

Memory hierarchy
-----------------

:numref:`Fig. %s <nki-fig-pm-memory>` below shows the four-level
memory hierarchy available to a single NeuronCore. The
ranges provided in the figure are intended to calibrate the programmer's mental
model. See :doc:`NeuronDevice Architecture Guide <nki_arch_guides>` for the exact values.

Similar to standard memory hierarchy in other devices, memories near the top of the hierarchy
are the closest to the compute engines; therefore, they are designed to provide the highest
bandwidth and lowest latency. However, the faster memories have smaller capacities
compared to memories near the bottom.
Unlike memory hierarchy for traditional processors (e.g., CPU, GPU), all the memories
available to a NeuronCore are software-managed. They are managed either directly by the programmers
or the Neuron SDK. In other words, NeuronCore does not
have a hardware cache system to perform any data movement across memories that is opaque
to the program. Next, let's discuss the different memories bottom-up.

.. NKI programmers can initialize tensors in any of the memories by passing
.. the appropriate ``buffer`` parameter (``nki.language.sbuf``,
.. ``nki.language.psum`` or ``nki.language.hbm``) into the ``ndarray`` ,
.. ``zeros``, ``ones``, ``full`` or ``rand`` APIs.

.. _nki-fig-pm-memory:

.. figure:: img/pm-memory.png
   :align: center
   :width: 80%

   NeuronCore Memory Hierarchy with Capacity and Bandwidth Ranges

NeuronCore external memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two memories at the bottom of the hierarchy, host memory
and device memory, are both considered *external* memory for a NeuronCore.
These memories are **linear memory**, where multi-dimensional tensors must
be stored in a flattened manner.

The **host memory** is the CPU-attached DRAM, which is accessible by the host CPUs
and all the NeuronCores attached to the instance. NKI kernels currently
do not provide APIs to move data in and out of the host memory directly, but we can rely on ML
frameworks such as PyTorch or JAX to send input data from host memory into NeuronDevice and vice versa. For an example
of this, see :ref:`Getting Started with NKI <running-the-kernel>`.

The **device memory** resides within a NeuronDevice and uses High Bandwidth Memory (HBM) technologies
starting from NeuronDevice v2. This means that device memory and HBM refer to the same thing within NKI.
Currently, the input and output parameters to NKI kernels
must be HBM tensor references. Input tensors in HBM must be loaded into memory within a
NeuronCore before any computation can take place.

NeuronCore internal memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The two memories at the top of the hierarchy, SBUF and PSUM, are both considered
*internal*, on-chip memory for a NeuronCore.
Both memories are **two-dimensional** memory, organized in **128 partitions**.
The partitions size of PSUM is typically much smaller than SBUF, and PSUM/SBUF
partition sizes vary with NeuronCore generations.

State Buffer (SBUF) memory is the main software-managed on-chip SRAM.
The SBUF is accessible by all the compute engines within a NeuronCore.
NKI kernel input tensors from HBM must be loaded into the SBUF for computation using
:doc:`nki.language.load <api/generated/nki.language.load>`, and
computed output tensors of the kernel must be stored back into the HBM from SBUF
using :doc:`nki.language.store <api/generated/nki.language.store>` before the host can
access them.
In addition, SBUF is used for storing intermediate data within the kernel,
generated by the compute engines. Note, SBUF has **~20x higher bandwidth** than HBM,
but needs to be carefully managed to minimize HBM accesses for better
performance.

Lastly, Partial Sum Buffer (PSUM) memory is a small, dedicated
memory designed for storing matrix multiplication (MatMult) results computed by the tensor engine.
Tensor Engine is able to read-add-write to every address in PSUM.
Therefore, PSUM is useful for performing
large MatMult calculations using multiple tiles where multiple MatMult instructions
need to accumulate into the same output tile.
As is shown in :numref:`Fig. %s <nki-fig-pm-nc>`, PSUM memory can also be
read and written by the vector and scalar engines. However, due to the limited capacity of PSUM,
we recommend that you reserve PSUM space for the tensor engine
to write MatMult outputs and
to use the vector and scalar engines to evict MatMult results back to SBUF as soon as possible.

.. TODO: link to tutorials that showcase tiling/fusion

Note that to optimize kernel performance, it is a good practice for NKI
programmers to be mindful of SBUF and PSUM usage
through careful :ref:`tiling <nki-pm-tile>` and
loop fusion. However, ultimately the Neuron compiler performs memory
allocation for SBUF and PSUM and assigns memory addresses to kernel
intermediate data.
When the cumulative size of live data defined by the NKI
kernel overflows the capacity of any on-chip memory, the Neuron compiler
inserts the necessary spills or refills between that memory and
the next-tier memory in the hierarchy.

.. _pm_represent_data:

Representing data in NKI
------------------------

NKI represents data in NeuronCore's memory hierarchy with built-in type ``Tensor`` and its subclasses.

A ``Tensor`` is a multi-dimensional array which contains elements with
the same data type. Programmers can pass ``Tensor`` in and out of NKI kernels,
and declare or initialize ``Tensor`` in any memory within the NeuronDevice
(PSUM, SBUF, HBM) using APIs such as
:doc:`nki.language.ndarray <api/generated/nki.language.ndarray>`,
:doc:`nki.language.zeros <api/generated/nki.language.zeros>`, and
:doc:`nki.language.full <api/generated/nki.language.full>`.
Input and output tensors from ML frameworks to NKI kernels can be reinterpreted
as NKI ``Tensor`` of ``hbm`` buffer type in the same underlying memory buffer.

``Tensor`` in NeuronCore's internal memories (SBUF and PSUM) also have a dimension mapped to the
partitions of the internal memories. We call this dimension the ``partition dimension``.
By default, NKI infers the first dimension (that is, the left most dimension)
as the ``partition dimension`` of ``Tensor``. Users could also explicitly annotate the
partition dimension with ``par_dim`` from `nki.language`. For example:

.. code-block::

   # NKI infers the left most dimension as the partition dimension (size 128 below)
   x = nl.ndarray((128, 32, 512), dtype=nl.float32, buffer=nl.sbuf)

   # Same as above but more verbose
   y = nl.ndarray((nl.par_dim(128), 32, 512), dtype=nl.float32, buffer=nl.sbuf)

   # We can also explicitly annotate the partition dimension if we want the partition dimension
   # to be on the other dimensions. In the following code we are creating a tensor whose partition
   # dimension is the second from the left most dimension
   z = nl.ndarray((128, nl.par_dim(32), 512), dtype=nl.float32, buffer=nl.sbuf)


There is a special subclass of ``Tensor`` called ``Index``. ``Index`` represents the result of the
affine expression over variables produced by index-generating APIs,
such as loop variables, :doc:`nki.language.program_id <api/generated/nki.language.program_id>`,
:doc:`nki.language.affine_range <api/generated/nki.language.affine_range>`,
and :doc:`nki.language.arange <api/generated/nki.language.arange>`.


A ``Tensor`` whose ``partition dimension`` is the first dimension is also called a ``Tile`` in NKI.
In the above code example, ``x`` and ``y`` is a ``Tile``, ``z`` is not a ``Tile``.
All NKI APIs take ``Tile`` as input and return a ``Tile`` as output. We will give more explanation
in :ref:`Tile-based operations <nki-pm-tile>`.



.. _nki-pm-tile:

Tile-based operations
----------------------

All NKI APIs operate on Tile, which aligns with NeuronCore instruction set architecture (NeuronCore ISA).

.. code-block::

   x = nl.ndarray((128, 32, 512), dtype=nl.float32, buffer=nl.sbuf)
   xx = nl.exp(x) # works

   z = nl.ndarray((128, nl.par_dim(32), 512), dtype=nl.float32, buffer=nl.sbuf)
   zz = nl.exp(z) # not supported


To call NKI APIs to process data in a ``Tensor`` whose partition dimension is not the first dimension,
users need to generate Tiles from the ``Tensor``. This can be done by indexing the ``Tensor`` with a tuple
of ``Index``, following standard Python syntax ``Tensor[Index, Index, ...]``. For example:


.. code-block::

   z = nl.ndarray((128, nl.par_dim(32), 512), dtype=nl.float32, buffer=nl.sbuf)
   for i in range(128):
     zz = nl.exp(z[i, :, :]) # works

We will provide more discussion of the indexing in :ref:`Tensor Indexing <nki-tensor-indexing>`.
Next, let's discuss two important considerations when working with tile-based operations in NKI:
:ref:`data layout <nki-pm-layout>` and :ref:`tile size <nki-tile-size>` constraints.


.. _nki-pm-layout:

Layout considerations
-----------------------

When working with multi-dimensional arrays in any platform, it is
important to consider the physical memory layout of the arrays, or how
data is stored in memory. For example, in the context of 1D linear
memory, we can store a 2D array in a row-major layout or a
column-major layout. Row-major layouts place elements within each row in contiguous memory, and
column-major layouts place elements within each column in contiguous memory.

As discussed in the :ref:`Memory hierarchy <nki-pm-memory>` section,
the on-chip memories, SBUF and PSUM, are arranged as 2D memory
arrays. The first dimension is the partition dimension ``P`` with
128 memory partitions that can be read and written in parallel by compute engines.
The second dimension is the free dimension ``F`` where elements are
read and written sequentially. A tensor is placed in SBUF and PSUM across
both ``P`` and ``F``, with the same start offset across all ``P``
partitions used by the tensor.
:numref:`Fig. %s <nki-fig-pm-layout>`
below illustrates a default tensor layout. Note that a tile in NKI must
map ``shape[0]`` to the partition dimension.

.. _nki-fig-pm-layout:

.. figure:: img/pm-layout.png
   :align: center
   :width: 60%

   Tensor mapped to partition and free dimensions of SBUF and PSUM

Similar to other domain-specific languages that operate on tensors, NKI
defines a *contraction axis* of a tensor as the axis over which
reduction is performed, for example the summation axis in a dot product. NKI
also defines a *parallel axis* as an axis over which the same operation
is performed on all elements. For example, if we take a ``[100, 200]``
matrix and sum each row independently to get an output of shape
``[100, 1]``, then the row-axis (``axis[0]``, left-most) is the
parallel axis, and the column-axis (``axis[1]``, right-most) is the
contraction axis.

To summarize, the partition and free dimensions of a NKI tensor dictate how the tensor
is stored in the 2D on-chip memories physically, while the parallel and contraction
axes of a tensor are logical axes that are determined by the computation
to be done on the tensor.

The NeuronCore compute engines impose two layout constraints:

- **[LC#1]** For matrix multiplication operations, the contraction axis
  of both input tiles must be mapped to the ``P`` dimension.

- **[LC#2]** For operations that are not matrix multiplication operations,
  such as scalar or vector operations,
  the parallel axis should be mapped to the ``P`` dimension.


LC#1 means that to perform a matrix multiplication of shapes ``[M, K]`` and ``[K, N]``,
Tensor Engine (the engine performing this operation) requires the K dimension to be mapped
to the partition dimension in SBUF for both input matrices.
Therefore, you need to pass shapes ``[K, M]`` and ``[K, N]`` into
the :doc:`nki.isa.nc_matmul <api/generated/nki.isa.nc_matmul>` API,
as the partition dimension is always the left-most dimension
for an input tile to any NKI compute API.

To help developers get started with NKI quickly, NKI also provides a high-level API
:doc:`nki.language.matmul <api/generated/nki.language.matmul>` that can take ``[M, K]`` and ``[K, N]``
input shapes and invoke the necessary layout shuffling on the input data before sending it
to the Tensor Engine matmul instruction.

LC#2, on the other hand, is applicable to many instructions supported on Vector, Scalar and GpSimd
Engines. See :doc:`nki.isa.tensor_reduce <api/generated/nki.isa.tensor_reduce>` API as an example.


.. _nki-tile-size:

Tile size considerations
-------------------------

Besides layout constraints, NeuronCore hardware further imposes three
tile-size constraints in NKI:

- **[TC#1]** The ``P``
  dimension size of a tile in both SBUF and PSUM must never exceed
  ``nki.tile_size.pmax == 128``.

- **[TC#2]** For tiles in PSUM, the ``F``
  dimension size must not exceed ``nki.tile_size.psum_fmax == 512``.

- **[TC#3]**
  Matrix multiplication input tiles ``F`` dimension size must not exceed
  ``nki.tile_size.gemm_stationary_fmax == 128`` on the left-hand side (LHS), or
  ``nki.tile_size.gemm_moving_fmax == 512`` on the right-hand side (RHS).

You are responsible for breaking your tensors according to
these tile-size constraints. If the constraints are not met properly,
the NKI kernel compilation throws a ``SyntaxError`` indicating
which constraint is violated.
For example, below we show a simple kernel that applies the exponential
function to every element of an input tensor. To start, let's write a
kernel that expects a hard-coded shape of ``(128, 512)`` for both input
and output tensors:

.. nki_example:: examples/layout-pass.py
   :language: python
   :linenos:
   :whole-file:

As expected, the output tensor is an element-wise exponentiation of the
input-tensor (a tensor of ones):

::

   tensor([[2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
           [2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
           [2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
           ...,
           [2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
           [2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
           [2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188]],
           device='xla:1', dtype=torch.bfloat16)

.. _nki-output-garbage-data:

Now let's examine what happens if the input/output tensor shapes do not
match the shape of the compute kernel. As an example, we can change the
input and output tensor shape from ``[128,512]`` to ``[256,512]``:


.. nki_example:: examples/layout-violation.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_12
   :emphasize-lines: 7

Since the compute kernel is expecting ``(128, 512)`` input/output
tensors, but we used a ``(256, 512)`` input/output tensor instead, the
bottom half of the output tensor becomes garbage data:

::

   tensor([[2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
           [2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
           [2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
           ...,
           [0.5273, 0.6055, 0.4336, ..., 0.9648, 0.9414, 0.4062],
           [0.7109, 0.2539, 0.7227, ..., 0.7344, 0.2539, 0.1211],
           [0.8867, 0.2109, 0.8789, ..., 0.8477, 0.2227, 0.1406]],
           device='xla:1', dtype=torch.bfloat16)

We could try to fix this by changing the tile size inside the compute
kernel to ``(256, 512)`` as well, and see what happens: (*NOTE: This
violates tile-size constraint #1!*):

.. nki_example:: examples/layout-violation.py
   :language: python
   :linenos:
   :emphasize-lines: 18, 37
   :whole-file:

Here, Neuron compiler identifies the tile-size constraint violation and
fails compilation with the following exception:

::

   SyntaxError: Size of partition dimension 256 exceeds architecture limitation of 128.

Now, let's see how NKI developers can build a kernel that properly
handles ``(256, 512)`` input/output tensors with a simple loop. We can
use the ``nki.language.tile_size.pmax`` constant defined in NKI as the maximum
partition dimension size in a tile.

.. nki_example:: examples/layout-loop.py
   :language: python
   :linenos:
   :emphasize-lines: 18, 20
   :whole-file:

The ``nl.affine_range(2)`` API call returns a list of integers
``[0, 1]``. :doc:`nl.affine_range <api/generated/nki.language.affine_range>`
should be the default loop iterator choice in NKI, when the loop
has no loop-carried dependency. Note, associative reductions are not considered
loop carried dependencies in this context. One such example is
accumulating results of multiple matrix multiplication calls into the same output buffer using ``+=``
(see :doc:`Matmul Tutorial <tutorials/matrix_multiplication>` for an example).
Otherwise, :doc:`nl.sequential_range <api/generated/nki.language.sequential_range>`
should be used to handle loop-carried dependency.
Note, Neuron compiler transforms any usage of Python ``range()``
API into ``nl.sequential_range()`` under the hood.
See :ref:`NKI iterator API <nl_iterators>`
for a detailed discussion of various loop iterator options in NKI.


While the code above does handle ``(256, 512)`` tensors correctly, it is
rather inflexible since it only supports input shape of
``(256, 512)``. Therefore, as a last step, we extend this kernel to handle
varying input/output sizes:

.. nki_example:: examples/layout-dynamic-loop.py
   :language: python
   :linenos:
   :emphasize-lines: 14, 19, 21, 24
   :whole-file:

The above example handles cases where in_tensor.shape[0] is not a multiple of 128
by passing a ``mask`` field into the ``nl.load`` and ``nl.store`` API calls.
For more information, refer to :ref:`NKI API Masking <nki-mask>`.

Later in this guide, we'll explore another way to launch a
kernel with varying input/output shapes, with a single program multiple data programming model, or :ref:`SPMD <nki-pm-spmd>`.
The SPMD programming model removes the need for explicit looping over
different tiles with variable trip counts, which could lead to cleaner
and more readable code.

.. _pm_sec_tile_indexing:

..  _nki-tensor-indexing:

Tensor Indexing
---------------
As mentioned above, we can index ``Tensor`` with standard Python syntax to produce ``Tiles``.
There are two styles of indexing: Basic and Advanced Tensor Indexing.
Note that currently NKI does not support mixing Basic and Advanced Tensor Indexing in the same ``Index`` tuple.

..   _nki-basic-tensor-indexing:

Basic Tensor Indexing
^^^^^^^^^^^^^^^^^^^^^
We can index a ``Tensor`` with fewer indices than dimensions, we get a *view* of the original tensor
as a sub-dimensional tensor. For example:

.. code-block::

   x = nl.ndarray((2, 2, 2), dtype=nl.float32, buffer=nl.hbm)

   # `x[1]` return a view of x with shape of [2, 2]
   # [[x[1, 0, 0], x[1, 0 ,1]], [x[1, 1, 0], x[1, 1 ,1]]]
   assert x[1].shape == [2, 2]

By indexing a ``Tensor`` like this, we can generate a ``Tile`` with the partition dimension in the
first dimension and feed the Tile to NKI compute APIs:

.. code-block::

   # Not a tile, cannot directly feed to a NKI compute API
   x = nl.ndarray((2, nl.par_dim(2), 2), dtype=nl.float32)
   # Error
   y = nl.exp(x)

   # `x[1]` have shape [2, 2], and the first dimension is the partition dimension of the original
   # tensor. We can feed it to a NKI compute API.
   y = nl.exp(x[1])

NKI also supports **slicing** in basic tensor indexing:

.. code-block::

   x = nl.ndarray((2, 128, 1024), dtype=nl.float32, buffer=nl.hbm)

   # `x[1, :, :]` is the same as `x[1]`
   assert x[1, :, :].shape == [128, 1024]

   # Get a smaller view of the third dimension
   assert x[1, :, 0:512].shape == [128, 512]

   # `x[:, 1, 0:2]` returns a view of x with shape of [2, 2]
   # [[x[0, 1, 0], x[0, 1 ,1]], [x[1, 1, 0], x[1, 1 ,1]]]
   assert x[:, 1, 0:2].shape == [2, 2]


..   _nki-advanced-tensor-indexing:

Advanced Tensor Indexing
^^^^^^^^^^^^^^^^^^^^^^^^

So far we have only shown basic indexing in tensors. However,
NeuronCore offers much more flexible tensorized memory access in its
on-chip SRAMs along the free dimension. You can use this to
efficiently stride the
SBUF/PSUM memories at high performance for all NKI APIs that access on-chip memories.
However, such flexible indexing is not supported along the partition dimension.
That being said, device memory (HBM) is always more performant when accessed sequentially.

In this section, we share several use cases that benefit from advanced
memory access patterns and demonstrate how to implement them in NKI.

Advanced Tensor Indexing in NKI leverages the `nl.arange` API.

Case #1 - Tensor split to even and odd columns
``````````````````````````````````````````````

Here we split an input tensor into two output tensors, where the first
output tensor gathers all the even columns from the input tensor,
and the second output tensor gathers all the odd columns from the
input tensor. We assume the rows of the input tensor are mapped to SBUF
partitions. Therefore, we are effectively gathering elements along
the free dimension of the input tensor. :numref:`Fig. %s <nki-fig-pm-index-1>`
below visualizes the input and output tensors.

.. _nki-fig-pm-index-1:

.. figure:: img/pm-index-1.png
   :align: center
   :width: 60%

   Tensor split to even and odd columns

.. nki_example:: examples/index-case-1.py
   :language: python
   :linenos:
   :whole-file:

The main concept in this example is that we introduced the even
(``i_f_even``) and odd ( ``i_f_odd`` ) indices. Note that both indices
are affine expressions of the form ``start + stride * nl.arange(size)`` with a
specific start offset (0/1 respectively) and stride (2 for both cases).
This allows us to stride through the ``in_tile`` memory and copy it to
both output tiles (``out_tile_even`` and ``out_tile_odd``), according to
the desired pattern.

Case #2 - Transpose tensor along the f axis
```````````````````````````````````````````

In this example we transpose a tensor along two of its axes. Note,
there are two main types of transposition in NKI:

1. Transpose between the partition-dimension axis and one of the
   free-dimension axes, which is achieved via the
   :doc:`nki.isa.nc_transpose <api/generated/nki.isa.nc_transpose>` API.
2. Transpose between two free-dimension axes, which is achieved
   via a :doc:`nki.language.copy <api/generated/nki.language.copy>` API,
   with indexing manipulation
   in the transposed axes to re-arrange the data.

In this example, we'll focus on the second case: consider a
three-dimensional input tensor ``[P, F1, F2]``, where the ``P`` axis is mapped
to the different SBUF partitions and the ``F1`` and ``F2`` axes are
flattened and placed in each partition, with ``F1`` being the major
dimension. Our goal in this example is to transpose the ``F1`` and
``F2`` axes with a parallel dimension ``P``,
which would re-arrange the data within each partition. :numref:`Fig. %s <nki-fig-index-2>`
below illustrates the input and output tensor layouts.

.. _nki-fig-index-2:

.. figure:: img/pm-index-2.png
   :align: center
   :width: 60%

   Tensor F1:F2 Transpose

.. nki_example:: examples/transpose2d/transpose2d_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_33

The main concept introduced in this example is a 2D memory access
pattern per partition, via additional indices. We copy ``in_tile`` into
``out_tile``, while traversing the memory in different access patterns
between the source and destination, thus achieving the desired
transposition.

You may download the full runnable script from :ref:`Transpose2d tutorial <tutorial_transpose2d_code>`.

Case #3 - 2D pooling operation
``````````````````````````````

Lastly, we examine a case of
dimensionality reduction. We implement a 2D MaxPool operation, which
is used in many vision neural networks. This operation takes
``C x [H,W]`` matrices and reduces each matrix along the ``H`` and ``W``
axes. To leverage free-dimension flexible indexing, we can map the ``C``
(parallel) axis to the ``P`` dimension and ``H/W`` (contraction)
axes to the ``F`` dimension.
Performing such a 2D pooling operation requires a 4D memory access
pattern in the ``F`` dimension, with reduction along two axes.
:numref:`Fig. %s <nki-fig-index-3>`
below illustrates the input and output tensor layouts.

.. _nki-fig-index-3:

.. figure:: img/pm-index-3.png
   :align: center
   :width: 60%

   2D-Pooling Operation (reducing on axes F2 and F4)

.. nki_example:: examples/index-case-3.py
   :language: python
   :linenos:
   :whole-file:

.. _nki-pm-spmd:

SPMD: Launching multiple instances of a kernel
------------------------------------------------

So far we have discussed how to launch a single NKI kernel instance,
in which the full input tensor is processed. In
this section, we discuss how to launch multiple instances of the same
kernel and slice the full input tensor across kernel instances
using a single program multiple data programming model (SPMD).

.. note::
   In current NKI release, adopting the SPMD programming model has **no**
   impact on performance of NKI kernel, and therefore is considered **optional**.
   A SPMD program is compiled into an executable that targets one NeuronCore,
   and the different instances of the SPMD program are executed serially on a single NeuronCore.
   This is subject to changes in future releases.

NKI allows users to launch multiple instances of a kernel, which are
organized in a user-defined multi-dimensional grid. The grid indices are
then used by the different kernel instances to select which input and
output data to access. There is no restriction on the number of
dimensions in an SPMD grid, nor on the size of each dimension. Each
kernel instance can find its coordinates within the launch grid using the
:doc:`nki.language.program_id <api/generated/nki.language.program_id>`
API. Neuron compiler translates the SPMD
launch grid into nested loops of compute-kernel invocations, which are
then executed on the NeuronCore.

As an example, we'll perform a ``C=A@B`` matrix multiplication, where
``A`` and ``B`` are of shape ``(512, 128)`` and ``(128, 1024)`` respectively.
We partition the output tensor C of shape ``(512, 1024)``
into ``4x2`` tiles and assign the task of computing each output
tile to a different kernel instance. A ``4x2`` launch-grid is
chosen in this case, in order to make each compute kernel instance operate on a
single tile in ``A`` and a single tile in ``B``, while adhering to the :ref:`tile-size
constraints <nki-pm-tile>`.

With a 2D ``4x2`` launch grid,
the ``(i,j)`` kernel instance is responsible for computing the
``(i,j)`` tile of ``C``. The computation of the ``(i,j)``
tile requires the corresponding rows of ``A`` and columns of
``B``. This induces a four-way row-wise partitioning of ``A`` and a two-way
column-wise partitioning of ``B``, as shown in :numref:`Fig. %s <nki-fig-spmd>`.

.. _nki-fig-spmd:

.. figure:: img/pm-spmd.png
   :align: center
   :width: 80%

   Visualization of 512x128x1024 matrix multiplication using SPMD

In this SPMD kernel example, we will use the high-level
:doc:`nki.language.matmul <api/generated/nki.language.matmul>` API,
so that we can focus on the concept of SPMD without worrying about the layout requirement
of Tensor Engine (:ref:`LC#1 <nki-pm-layout>`). To achieve the best performance,
we suggest transposing input ``A`` and invoking :download:`another NKI kernel <examples/mm-nisa-spmd.py>` instead,
which solely performs matmul operations on Tensor Engine
using :doc:`nki.isa.nc_matmul <api/generated/nki.isa.nc_matmul>` without extra overhead in changing
input layouts to meet :ref:`LC#1 <nki-pm-layout>`.

.. nki_example:: examples/mm-nl-spmd.py
   :language: python
   :linenos:
   :whole-file:
