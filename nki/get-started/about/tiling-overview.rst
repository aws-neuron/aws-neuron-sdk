.. meta::
   :description: Overview of Tiling process for NKI programmers
   :date_updated: 12/02/2025

.. _nki-about-tiling:

=======================
What is Tiling?
=======================

This topic covers tiling and how it applies to developing NKI kernels with the AWS Neuron SDK. Tiling is the process of dividing a large tensor up in to smaller tensors that can be processed by single Neuron ISA instructions. When writing NKI kernels, all tensors must be tiled to fit within the constraints of the hardware.

Tile-based operations
----------------------

All NKI APIs operate on tiles. A tile is just a tensor that resides in either the SBUF or PSUM memory with a size and layout that satisfies the constraints of the Neuron instruction set architecture (NeuronCore ISA). Since the SBUF and PSUM memories have 128 partitions, most APIs are limited to tiles with a first dimension (also called the "Partition Dimension") no larger than 128 elements. So, for example, to compute the reciprocal of a matrix of size 256x256, you will need to split the computation up into (at least) two parts:

.. code-block::

   # Example how to split 256x256 into tiles with 128 partition dimensions
   # Assume input and output are tensors of size 256 x 256

   # The hardware supports up to 128 partitions
   P_DIM = nki.language.tile_size.pmax

   # allocating memory for input and output tiles
   # note that memory allocation does not initialize
   in_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)
   out_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)

   # process first tile from input to output
   nki.isa.dma_copy(dst=in_tile, src=input[0:P_DIM, 0:256])
   nki.isa.reciprocal(dst=out_tile, data=in_tile)
   nki.isa.dma_copy(dst=output[0:P_DIM, 0:256], src=out_tile)

   # process second tile
   nki.isa.dma_copy(dst=in_tile, src=input[P_DIM:256, 0:256])
   nki.isa.reciprocal(dst=out_tile, data=in_tile)
   nki.isa.dma_copy(dst=output[P_DIM:256, 0:256], src=out_tile)

In the code above, we allocate two SBUF tensors to store our tiles: one for the input and one for the result. These two tiles are available within the kernel that they are declared in, and will be automatically recycled by the compiler when no longer needed. Then we copy the first 128 rows of our matrix from the input in HBM to the input tile in SBUF, and compute the reciprocal placing the result into the output tile in SBUF. Finally, we copy the result back to the output tensor, in HBM. Of course, this could also be done with a loop, as shown below.

.. code-block::

   # allocate memory for input and output tiles
   in_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)
   out_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)
   # process tiles
   for i in range(input.shape[0] // P_DIM):
       s = nl.ds(i * P_DIM, P_DIM) # equivalent to i * P_DIM : (i + 1) * P_DIM
       nki.isa.dma_copy(dst=in_tile, src=input[s, 0:256])
       nki.isa.reciprocal(dst=out_tile, data=in_tile)
       nki.isa.dma_copy(dst=output[s, 0:256], src=out_tile)

We will provide more discussion of the indexing in :ref:`Tensor Indexing <nki-tensor-indexing>`. Next, let's discuss two important considerations when working with tile-based operations in NKI: :ref:`data layout <nki-tile-layout>` and :ref:`tile size <nki-tile-size>` constraints.

.. _nki-tile-layout:

Layout considerations
-----------------------

When working with multi-dimensional arrays in any platform, it is important to consider the physical memory layout of the arrays, or how data is stored in memory. For example, in the context of 1D linear memory, we can store a 2D array in a row-major layout or a column-major layout. Row-major layouts place elements within each row in contiguous memory, and column-major layouts place elements within each column in contiguous memory.

As discussed in :ref:`Memory hierarchy <nki-about-memory>`, the on-chip memories, SBUF and PSUM, are arranged as 2D memory arrays. The first dimension is always the partition dimension ``P`` with 128 memory partitions that can be read and written in parallel by compute engines. The second dimension is the free dimension ``F`` where elements are read and written sequentially. A tensor is placed in SBUF and PSUM across both P and ``F``, with the same start offset across all ``P`` partitions used by the tensor. The figure below illustrates a default tensor layout. Note that a tile in NKI must map shape[0] to the partition dimension.

.. _nki-fig-pm-layout:

.. figure:: /nki/img/overviews/tiling-1.png
   :align: center
   :width: 70%

   Tensor mapped to partition and free dimensions of SBUF and PSUM

Similar to other domain-specific languages that operate on tensors, NKI defines a contraction axis of a tensor as the axis over which reduction is performed, for example the summation axis in a dot product. NKI also defines a parallel axis as an axis over which the same operation is performed on all elements. For example, if we take a ``[100, 200]`` matrix and sum each row independently to get an output of shape ``[100, 1``], then the row-axis (``axis[0]``, left-most) is the parallel axis, and the column-axis (``axis[1``], right-most) is the contraction axis.

To summarize, the partition and free dimensions of a NKI tensor dictate how the tensor is stored in the 2D on-chip memories physically, while the parallel and contraction axes of a tensor are logical axes that are determined by the computation to be done on the tensor.

The NeuronCore compute engines impose two layout constraints (LC):

* **[Layout Constraint #1]** For matrix multiplication operations, the contraction axis of both input tiles must be mapped to the Partition (P or P_DIM) dimension which is typically 128 for current hardware.
* **[Layout Constraint #2]** For operations that are not matrix multiplication operations, such as scalar or vector operations, the parallel axis should be mapped to the Partition (``P`` or ``P_DIM``) dimension.

Layout Constraint #1 means that to perform a matrix multiplication of shapes ``[M, K]`` and ``[K, N]`` that contracts on K to generate ``[M, N]``, Tensor Engine (the engine performing this matmul operation) requires the K dimension to be mapped to the partition dimension in SBUF for both input matrices. Therefore, you need to pass shapes ``[K, M]`` and ``[K, N]`` into the :doc:`nki.isa.nc_matmul </nki/api/generated/nki.isa.nc_matmul>` API, as the partition dimension is always the left-most dimension for an input tile to any NKI compute API.

To help developers get started with NKI quickly, NKI also provides a high-level API :doc:`nki.isa.nc_matmul </nki/api/generated/nki.isa.nc_matmul>` that can take ``[M, K]`` and ``[K, N]`` input shapes and invoke the necessary layout shuffling on the input data before sending it to the Tensor Engine matmul instruction.

LC#2, on the other hand, is applicable to many instructions supported on Vector, Scalar and GpSimd Engines. See :doc:`nki.isa.tensor_reduce </nki/api/generated/nki.isa.tensor_reduce>` API as an example.

.. _nki-tile-size:

Tile size considerations
-------------------------

Besides layout constraints, NeuronCore hardware further imposes three tile-size constraints (TC) in NKI:

* **[Tile-Size Constraint#1]** The P dimension size of a tile in both SBUF and PSUM must never exceed ``nki.tile_size.pmax == 128``.
* **[Tile-Size Constraint#2]** For tiles in PSUM, the F dimension size must not exceed ``nki.tile_size.psum_fmax == 512``.
* **[TileSize Constraint#3]** Matrix multiplication input tiles F dimension size must not exceed ``nki.tile_size.gemm_stationary_fmax == 128`` on the left-hand side (LHS), or ``nki.tile_size.gemm_moving_fmax == 512`` on the right-hand side (RHS).

Programmers are responsible for breaking up your tensors according to these tile-size constraints. For example, below is a simple kernel that applies the exponential function to every element of an input tensor. The kernel expects a shape of ``(128, 512)`` for both input and output tensors:

.. code-block::

   import nki.isa as nisa
   import nki.language as nl
   import nki

   # The hardware supports up to 128 partitions
   P_DIM = nki.language.tile_size.pmax

   @nki.jit
   def tensor_kernel(in_tensor):
    """NKI kernel to compute elementwise reciprocal of an input tensor
    Args:
    in_tensor: an input tensor of shape [128,512]
    Returns:
    out_tensor: an output tensor of shape [128,512]
    """
     X_SIZE = 128
     Y_SIZE = 512
     
     # allocate space for the result
     out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
     # allocate space for tile memory
     in_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)
     out_tile = nl.ndarray((P_DIM, 256), dtype=nl.float32, buffer=nl.sbuf)

     # Process first tile
     nki.isa.dma_copy(dst=in_tile, src=in_tensor[0:P_DIM, 0:256])
     nki.isa.reciprocal(dst=out_tile, data=in_tile)
     nki.isa.dma_copy(dst=out_tensor[0:P_DIM, 0:256], src=out_tile)
     
     return out_tensor

As expected, the output tensor is an element-wise exponentiation of the input-tensor (a tensor of ones):

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

Now let's examine what happens if the input/output tensor shapes do not match the shape of the compute kernel. As an example, we can change the input and output tensor shape from ``[128,512]`` to ``[256,512]``:

Since the compute kernel is expecting ``(128, 512)`` input/output tensors, but we used a ``(256, 512)`` input/output tensor instead, the bottom half of the output tensor becomes garbage data:

::

   tensor([[2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
   [2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
   [2.7188, 2.7188, 2.7188, ..., 2.7188, 2.7188, 2.7188],
   ...,
   [0.5273, 0.6055, 0.4336, ..., 0.9648, 0.9414, 0.4062],
   [0.7109, 0.2539, 0.7227, ..., 0.7344, 0.2539, 0.1211],
   [0.8867, 0.2109, 0.8789, ..., 0.8477, 0.2227, 0.1406]],
   device='xla:1', dtype=torch.bfloat16)

We could try to fix this by changing the tile size inside the compute kernel to ``(256, 512)`` as well, and see what happens: (**Note**: This violates tile-size constraint #1!) 

Here, the Neuron Graph Compiler identifies the tile-size constraint violation and fails compilation with the following exception:

::

   Size of partition dimension 256 exceeds architecture limitation of 128.

Now, let's see how to build a kernel that properly handles ``(256, 512)`` input/output tensors with a simple loop. We can use the ``nki.language.tile_size.pmax`` constant defined in NKI as the maximum partition dimension size in a tile.

.. code-block::

   import nki.isa as nisa
   import nki.language as nl
   import nki

   # The hardware supports up to 128 partitions
   P_DIM = nki.language.tile_size.pmax

   @nki.jit
   def tensor_exp_kernel_(in_tensor):
     """NKI kernel to compute elementwise exponential of an input tensor
     Args:
         in_tensor: an input tensor of shape [256,512]
     Returns:
         out_tensor: an output tensor of shape [256,512]
     """
     X_SIZE = 128
     Y_SIZE = 512
     assert in_tensor.shape == (X_SIZE, Y_SIZE)
     # allocate space for the result
     out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
     # allocate space for tile memory
     in_tile = nl.ndarray((P_DIM, Y_SIZE), dtype=nl.float32, buffer=nl.sbuf)
     out_tile = nl.ndarray((P_DIM, Y_SIZE), dtype=nl.float32, buffer=nl.sbuf)

     for k in nl.affine_range(in_tensor.shape[0] / nl.tile_size.pmax):
       # Generate tensor indices for the input/output tensors
       p_start = k * nl.tile_size.pmax
       i_p = nl.ds(p_start, nl.tile_size.pmax)

       # Process tile
       nki.isa.dma_copy(dst=in_tile, src=in_tensor[i_p, :])
       nki.isa.reciprocal(dst=out_tile, data=in_tile)
       nki.isa.dma_copy(dst=out_tensor[i_p, :], src=out_tile)
     
     return out_tensor

The ``nl.affine_range(2)`` API call is similar to the Python ``range`` function, and you can think of it as returning ``[0, 1]``. See :ref:`NKI iterator API <nl_iterators>` for a detailed discussion of various loop iterator options in NKI.

While the code above does handle ``(256, 512)`` tensors correctly, it is rather inflexible since it only supports an input shape of ``(256, 512)``. Therefore, as a last step, we extend this kernel to handle varying input/output sizes:

.. code-block::

   import nki.isa as nisa
   import nki.language as nl
   import nki
   import math

   # The hardware supports up to 128 partitions
   P_DIM = nki.language.tile_size.pmax

   @nki.jit
   def tensor_exp_kernel_(in_tensor):
     """NKI kernel to compute elementwise exponential of an input tensor
     Args:
         in_tensor: an input tensor of ANY 2D shape (up to SBUF size)
     Returns:
         out_tensor: an output tensor of ANY 2D shape (up to SBUF size)
     """

     sz_p, sz_f = in_tensor.shape
     assert sz_f < nl.tile_size.total_available_sbuf_size
    
     # allocate space for the result
     out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype, buffer=nl.shared_hbm)
     # allocate space for tile memory
     in_tile = nl.ndarray((P_DIM, sz_f), dtype=nl.float32, buffer=nl.sbuf)
     out_tile = nl.ndarray((P_DIM, sz_f), dtype=nl.float32, buffer=nl.sbuf)
     
     for p in nl.affine_range(math.ceil(sz_p / P_DIM)):
       # Generate tensor indices for the input/output tensors
       p_start = p * P_DIM
       p_end = p_start + P_DIM
       i_p = slice(p_start, min(p_end, sz_p)) # same as nl.ds(p_start, min(p_end, sz_p) - p_start)

       # Process tile
       nki.isa.dma_copy(dst=in_tile, src=in_tensor[i_p, :])
       nki.isa.reciprocal(dst=out_tile, data=in_tile)
       nki.isa.dma_copy(dst=out_tensor[i_p, :], src=out_tile)
       
     return out_tensor

The above example handles cases where ``in_tensor.shape[0]`` is not a multiple of 128 by using the standard Python ``min`` function to make sure the tensor access is in bounds.

Further reading
---------------

- :ref:`Logical Neuron Cores (LNC) <nki-about-lnc>`
