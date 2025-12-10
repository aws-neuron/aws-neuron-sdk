.. meta::
   :description: Trainium2 Architecture Guide for NKI
   :keywords: AWS Neuron, Trainium2, NeuronCore-v3, NKI, architecture
   :date-modified: 12/01/2025

.. _trainium2_arch:

Trainium2 Architecture Guide for NKI
===============================================

In this guide, we will dive into hardware architecture of third-generation NeuronDevices: Trainium2. This guide will highlight major architectural updates compared to the previous generation. Therefore, we assume readers have gone through :doc:`Trainium/Inferentia2 Architecture Guide <trainium_inferentia2_arch>` in detail to understand the basics of NeuronDevice Architecture.

The diagram below shows a block diagram of a Trainium2 device, which consists of:

* 8 NeuronCores (v3).
* 4 HBM stacks with a total device memory capacity of 96GiB and bandwidth of 3TB/s.
* 128 DMA (Direct Memory Access) engines to move data within and across devices.
* 20 CC-Cores for collective communication.
* 4 NeuronLink-v3 for device-to-device collective communication.

.. _fig-arch-neuron-device-v3:

.. image:: /nki/img/arch_images/neuron_device3.png

Trainium2 Device Diagram.

For a high-level architecture specification comparison from Trainium1 to Trainium2, check out the
:doc:`Neuron architecture guide for Trainium2 </about-neuron/arch/neuron-hardware/trainium2>`. The rest of this guide will provide details on new features or improvements in NeuronCore-v3 compute engines and memory subsystem compared to NeuronCore-v2.

NeuronCore-v3 Compute Engine Updates
------------------------------------

The figure below is a simplified NeuronCore-v3 diagram of the compute engines and their connectivity to the two on-chip SRAMs, SBUF and PSUM. This is similar to NeuronCore-v2.

.. _fig-neuroncore-v3-diagram:

.. image:: /nki/img/arch_images/nki-trn2-arch-1.png

NeuronCore-v3 SBUF capacity is **28MiB** (or, 128 partitions of 224KiB), up from 24 MiB in NeuronCore-v2. PSUM capacity remains the same at 2MiB. Engine data-path width and frequency are updated to the following:

.. list-table:: Compute Engine Specifications
   :widths: 20 20 40 20
   :header-rows: 1

   * - Device Architecture
     - Compute Engine
     - Data-path Width (elements/cycle)
     - Frequency (GHz)
   * - Trainium2
     - Tensor
     - 4x128 (dense FP8_E4/FP8_E5 input), 2x128 (dense BF16/FP16 input) or 5x128 (sparse input); 1x128 (output)
     - 2.4
   * - 
     - Vector
     - 512 BF16/FP16 input/output; 256 input/output for other data types
     - 0.96
   * - 
     - Scalar
     - 128 input/output
     - 1.2
   * - 
     - GpSimd
     - 
     - 1.2

Next, we will go over major updates to each compute engine.

Tensor Engine
--------------

The Tensor Engine is optimized for tensor computations such as GEMM, CONV, and Transpose. A NeuronCore-v3 Tensor Engine delivers 158 FP8, 79 BF16/FP16/TF32 and 20 FP32 dense TFLOPS of tensor computations. It also delivers 316 FP8/BF16/FP16/TF32 sparse TFLOPS. The rest of this section describes new architectural features introduced in NeuronCore-v3 Tensor Engine. 

Double FP8 Matmul Performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NeuronCore-v3 TensorEngine (TensorE from now on) supports matrix multiplications (matmuls) of FP8 input matrices (including FP8_E4 and FP8_E5 formats [1]_) at **double** the throughput compared to BF16/FP16. Mixing FP8_E4 in one input matrix and FP8_E5 in the other is also allowed. This FP8 double performance mode uses FP32 as the accumulation data type, similar to BF16/FP16 matmul.

.. [1] FP8_E3 format is still supported by NeuronCore-v3 TensorE similar to NeuronCore-v2, but its matmul performance is the same as BF16/FP16.

Logically, TensorE doubles the FP8 matmul performance by doubling the maximum contraction dimension of a matmul instruction from 128 (for BF16/FP16) to 256, effectively presenting a 256x128 systolic array to the programmer. Under the hood, since the systolic array is still organized as a grid of 128x128 processing elements, each processing element performs two pairs of FP8 multiplications and also accumulation of the two multiplication results per cycle. The remaining section discusses the semantics of a single double-FP8 matmul instruction. Multiple such instructions can be used to accommodate larger matrix multiplications than the allowed instruction-level tile sizes.

A double-FP8 matmul can perform a multiplication of a 128x256 matrix and a 256x512 matrix (that is, MxKxN matmul, M=128, K=256, N=512). The figure below shows a visualization of the two input matrices (x and y) and the matmul output matrix (output). The figure also highlights two elements (red and yellow) in the first row of the x matrix and in the first column of the y matrix. These two elements are 128 (K//2) elements apart within the rows and columns. We will use these elements to illustrate the SBUF layout requirements for these matrices next. 


.. _fig-double-fp8-matmul:

.. image:: /nki/img/arch_images/nki-trn2-arch-2.png

These tensors must still fit in the 128-partition SBUF, with each partition feeding data into each row of processing elements inside the TensorE. The contraction of size 256 is therefore split into two dimensions: (1) the partition dimension of size 128 and (2) the most major (slowest) free dimension of size 2. This is illustrated in the figure below. Both the stationary matrix (x in above figure) and the moving matrix (y in above figure) are sliced in two tiles, where the first and second tile correspond to first and second halves of the contraction dimension, respectively. 

.. _fig-double-fp8-sbuf-layout:

.. image:: /nki/img/arch_images/nki-trn2-arch-3.png

Next, we invoke the LoadStationary and MultiplyMoving instructions to perform the matrix multiplications using the above tensors in SBUF. This is illustrated in figure below. The LoadStationary instruction loads the stationary tensor (K/2=128, 2, M=128) into TensorE, which stores two data elements into a single processing element (for example, the red and yellow elements land in the first processing element of TensorE as shown in ❶). Next, the MultiplyMoving instruction streams the moving tensor horizontally across the loaded stationary tensor. Similar to LoadStationary, two elements of moving tensor are sent to the same processing element simultaneously as shown in ❷, such that they can get multiplied with the corresponding pair of loaded stationary elements.

.. _fig-double-fp8-instruction:

.. image:: /nki/img/arch_images/nki-trn2-arch-4.png

Note that the above double FP8 ``LoadStationary``/``MultiplyMoving`` instruction sequence with a 256 contraction dimension takes the same amount of time as the regular BF16/FP16 LoadStationary/MultiplyMoving instruction sequence with a 128 contraction dimension. Since the double FP8 instruction performs double the FLOPs, overall double FP8 matmul on TensorE can achieve double the throughput compared to BF16/FP16 matmuls.

NKI programmers can invoke double FP8 matmul using the ``nisa.nc_matmul()`` API on NeuronCore-v3:

.. code-block:: python

   import nki.isa as nisa

   # stationary: [128, 2, 128]
   # moving: [128, 2, 512]
   # dst: [128, 512]
   nisa.nc_matmul(dst, stationary, moving, 
                  perf_mode=nisa.matmul_perf_mode.double_row, ...)

The ``nt.tensor[128, 2, 128]`` stationary and ``nt.tensor[128, 2, 512]`` moving tensor shapes reflect the maximum tile sizes for the double FP8 matmul instruction. Smaller tile sizes are supported, though the second dimension (the most major free dimension) of both input tensors must be two. In other words, if the contraction dimension of the matmul is not a multiple of two, programmers are required to explicitly pad the input tensors with zeros to enable the performance mode.

A full NKI kernel example performing double FP8 matmul is available on `Github <https://github.com/aws-neuron/nki-samples/blob/main/src/nki_samples/reference/double_row_matmul.py>`_.

Note that Double FP8 matmul performance mode cannot be combined with the following TensorE features:

* Column tiling mode
* Sparse matmul (new in NeuronCore-v3, discussion below)
* Transpose mode (new in NeuronCore-v3, more discussion below)

.. TODO: Uncomment and unindent when the NISA API ships
   M:N Structured Sparsity
   ^^^^^^^^^^^^^^^^^^^^^^^^

   Trainium2 TensorE introduces sparse matmul (matrix multiplication) support for M:N structured sparsity. This new functionality multiplies a regular dense moving matrix with a sparse stationary matrix that exhibits a M:N sparsity pattern, where every N elements only have up to M non-zero values along the contraction dimension. Trainium2 hardware supports up to 4x compression ratio and therefore 4x faster matmul performance compared to dense, with the largest value of N being 16. Programmers also have the flexibility to choose a lower compression ratio (CR=N/M) for better model accuracy. NKI currently supports the following M:N patterns: 4:8 (2x compression), 4:12 (3x) and 4:16 (4x), through the ``nki.isa.sparse_matmul`` API.

   To exercise sparse matmul, the sparse stationary matrix must be compressed to store only M out of every N elements, along with a tag tensor which indicates the original positions of the remaining M non-zero elements. In Figure below, a stationary matrix with a compression ratio of 16:4, along with its compressed representation.

   .. _fig-sparse-matmul:

   .. image:: /nki/img/arch_images/nki-trn2-arch-5.png

   The ``nki.isa.sparse_matmul`` API takes the following arguments ``nc_matmul_sparse(moving, stationary, tag, compress_ratio)``.

   Each row TensorE is able to read from 4, 2, or 1 SBUF partitions corresponding to the maximum compression ratio supported by the sparsity on TRN2 ratio. In order to efficiently utilize the TensorE the input moving should have shape ``[Partition Dimension <=128, Compression Ratio, Tile Free Dimension <= 512]``. The stationary matrix represents a 128x128 compressed weight tensor.

   Finally, the tag tensor is a 128x32 tensor of uint16. Each position of uncompressed elements is encoded as an 4-bit integer, which is the minimal width to relative position within N=16. Four tags are then packed into a uint16 datatype which forms the [128,32] tensor.

   A sample ``nki.isa.sparse_matmul`` can be found here:

   .. code-block:: python

      def mm_sparse_128_512_cr4(moving_tensor, stationary_tensor, tag_tensor, output):
      """
      Args:
         moving: Input tensor of shape [128, 4, 512], which represents activation tensor
         stationary: Input tensor of shape [128, 128], which represents the 
                     compressed weight tensor
         tag: Input tensor of shape [128, 32] of uint16 datatype where each tag 
               represents the indices of non-zero elements of the weight tensor.
               Tags are uint4 datatypes, 4 tags are packed into 1 uint16 datatype
         output: reference to the resulting output tensor of shape [128, 512]
      """
      _, compress_ratio, _ = moving_tensor.shape
      
      moving = nl.load(moving_tensor)
      stationary = nl.load(stationary_tensor)
      tag = nl.load(tag_tensor)

      psum_buf = nc_matmul_sparse(moving, # [128P, 4, 512F] 
                                    stationary, # [128P, 128F]
                                    tag, # [128, 32]
                                    compress_ratio)

      nl.store(output, value=psum_buf)

      # Sparse matmul     
      def test_nc_matmul_sparse(self):
         M = 512
         N = 128
         K = 128
         sparsity_pattern = (16, 4)
         L, R = sparsity_pattern
         ratio = L // R

         moving = np.random.random_sample((K, ratio, M)).astype(dtype) # activation
         stationary = np.random.random_sample((K, N)).astype(dtype) # compressed weight 
         
         # For demonstration purpose, we use random values between 0~15 in the tag tensor
         tag = np.random.randint(0, 16, size=(K, N), dtype=np.ubyte)
         
         # maps logical tags to physical tags and pack 4 uint4 to 1 uint
         squeezed_tag = squeeze_tags(tag, ratio) 
         
         # Generate NKI output
         nki_output = np.zeros((N, M), dtype=dtype)
         mm_sparse_128_512_cr4(moving, stationary, squeezed_tag, nki_output)

Built-in Transpose Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As discussed in :doc:`Trainium/Inferentia2 Architecture Guide </nki/about/trainium_inferentia2_arch>`, one common use of TensorE besides matrix multiplication operations is transposition of a 2D SBUF tensor, which swaps the partition and free dimension of the matrix. Such a transposition is done through a matmul of the tensor to be transposed (stationary tensor) and an identity matrix (moving tensor). Prior to NeuronCore-v3, TensorE has to perform multiplication of each data element with 1.0 or 0.0 and accumulation along the contraction dimension normally. However, if the tensor to be transposed contains NaN/Inf floating point values, the matmul result will not be a bit-accurate transposition of the original matrix - the NaN/Inf values will propagate through the accumulation chain and spread across the output tensor.

Starting with NeuronCore-v3, TensorE supports an explicit transpose mode, which can correctly transpose input tensors with NaN/Inf. In addition, the transpose mode provides the following benefits:

* 2x speedup in FP32 transpose, vs. no transpose mode enabled.
* FP16/BF16 PSUM output for FP16/BF16 transpose, vs. FP32 (default matmul output data type) PSUM output when no transpose mode enabled. This allows faster PSUM data eviction back to SBUF.

.. note:: NeuronCore-v3 TensorE transpose mode for FP8 input data produces 16-bit output elements in PSUM, with the upper 8 bits filled with zeros.

NKI programmers can enable TensorE transpose mode on NeuronCore-v3 through the following APIs:

.. code-block:: python

   nisa.nc_matmul(..., is_transpose=True)
   # OR
   nisa.nc_transpose(..., engine=nisa.constants.engine.tensor)

Vector Engine
----------------

Vector Engine (VectorE) is specially designed to accelerate vector operations where every element in the output tensor typically depends on multiple elements from input tensor(s), such as vector reduction and element-wise operators between two tensors. NeuronCore-v3 Vector Engine delivers a total of 1.0 TFLOPS of FP32 computations and can handle various input/output data-types, including FP8, FP16, BF16, TF32, FP32, INT8, INT16, and INT32. 

Vector Engine Performance Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NeuronCore-v3 Vector Engine provides a new performance mode BF16/FP16 data types, which quadruples or doubles the instruction throughput depending on the instruction type compared to NeuronCore-v2 (more details below). Enabling this performance mode does not change the computation precision - all computation is still done in FP32, similar to NeuronCore-v2 Vector Engine.

In particular, the following instructions could see a 4x throughput lift compared to NeuronCore-v2:

1. ``nisa.tensor_copy`` and ``nisa.tensor_scalar`` when both input/output tensors:
    a. are in SBUF
    b. are in BF16/FP16 (input and output data types do not need to match)
    c. have physically contiguous elements in the inner-most (most minor) free dimension

The following instructions could see a 2x throughput lift compared to NeuronCore-v2:

1. ``nisa.tensor_copy`` and ``nisa.tensor_scalar``:
    a. when both input/output tensors satisfy 1a and 1b, but not 1c conditions above, or
    b. when both input/output tensors satisfy 1b and 1c, but one of input and output tensors is in PSUM
2. ``nisa.tensor_tensor``:
    a. when both input tensors are SBUF and all of input/output tensors are in BF16/FP16

Note, NKI programmers are not required to explicitly enable VectorE performance mode. VectorE detects the above conditions and enables performance mode automatically in hardware.

Scalar Engine
---------------

As discussed in Trainium/Inferentia2 Architecture Guide, Scalar Engine (ScalarE) is specially designed to accelerate scalar operations where every element in the output tensor only depends on one element of the input tensor. In addition, ScalarE provides hardware acceleration to evaluate non-linear functions such as Gelu and Sqrt. All architectural capabilities from NeuronCore-v2 Scalar Engine are applicable to NeuronCore-v3. NeuronCore-v3 Scalar Engine additionally supports bit-accurate tensor copies without intermediate FP32 data type casting, similar to VectorE and Gpsimd Engine (see details in ``nisa.tensor_copy``).

Gpsimd Engine
--------------

GpSimd Engine (GpSimdE) is intended to be a general-purpose engine that can run any ML operators that cannot be lowered onto the other highly specialized compute engines discussed above efficiently, such as applying a triangular mask to a tensor. A GpSimdE consists of eight fully programmable processors that can execute arbitrary C/C++ programs.

In NeuronCore-v3, each processor in GpsimdE also comes with an integrated DMA engine that can move data in parallel to computation on GpsimdE and also parallel to data movements done by the main DMA engines on the Neuron Device. These integrated DMA engines can reach any SBUF/HBM on-chip or off-chip in the same trn2 instance. All eight processors together have a total integrated DMA bandwidth of 307 GB/s (153 GB/s per read/write direction).

In NeuronCore-v3, each processor in GpsimdE also comes with an integrated DMA engine that can move data in parallel to computation on GpsimdE and also parallel to data movements done by the main DMA engines on the Neuron Device. These integrated DMA engines can reach any SBUF/HBM on-chip or off-chip in the same trn2 instance. All eight processors together have a total integrated DMA bandwidth of 307 GB/s (153 GB/s per read/write direction). 

Data Movement Updates
----------------------

Trainium2 consists of a three-tiered memory hierarchy: HBM, SBUF and PSUM, from highest to lowest memory capacity. Figures below show the specifications of these memories and their connectivity for one NeuronCore-v3.

.. _fig-memory-hierarchy:

.. image:: /nki/img/arch_images/nki-trn2-arch-5-1.png

.. _fig-memory-hierarchy-2:

.. image:: /nki/img/arch_images/nki-trn2-arch-6.png

As shown in the above figures, data movement between HBM and SBUF is performed using on-chip DMA (Direct Memory Access) engines, which can run in parallel to computation within the NeuronCore. Data movement between PSUM and SBUF is done through ISA instructions on the compute engines. In NeuronCore-v3, two restrictions in engine parallel accesses to SBUF/PSUM are lifted to improve programming flexibility compared to NeuronCore-v2:

1. VectorE and GpSimdE can access SBUF in parallel.
    a. This was disallowed in NeuronCore-v2.
    b. VectorE's performance mode leverages a shared memory bus between the VectorE and GpsimdE engines to deliver 2-4x performance improvement for select VectorE instructions. The hardware automatically coordinates access between engines to optimize bus utilization, including arbitrating between GpsimdE and relevant VectorE instructions.
2. VectorE and ScalarE can access PSUM in parallel.
    a. This was disallowed in NeuronCore-v2.
    b. Both VectorE and ScalarE can access PSUM at full bandwidth in parallel, as long as their accesses do not collide on the same PSUM bank.

DMA Transpose
^^^^^^^^^^^^^^^

Trainium2 DMA engines can perform a tensor transpose while moving data from HBM into SBUF, or from SBUF to SBUF itself. The figure below illustrates these two supported DMA transpose data flows. Trainium2 DMA transpose supports bit-accurate transposition for both 2-byte and 4-byte data types.

.. _fig-dma-transpose:

.. image:: /nki/img/arch_images/nki-trn2-arch-7.png

HBM2SBUF DMA transpose
""""""""""""""""""""""

Before diving into how HBM2SBUF transpose works, let's revisit a simple DMA copy from a packed HBM tensor ``[128, 512]`` to an SBUF tensor ``[nl.par_dim(128), 512]``. Following Numpy convention, these tensor shapes follow a major to minor ordering. The figure below visualizes these HBM and SBUF tensors. A packed ``[128, 512]`` HBM tensor consists of 128 chunks of 512 elements, laid out back to back in the HBM linear memory. The most minor (that is, inner-most) dimension consists of 512 contiguous elements in memory. Once loaded into the SBUF, the most minor HBM tensor dimension (512) is mapped to the free dimension of the SBUF, while the most major dimension is mapped to the SBUF partition dimension.

In Trainium2, each NeuronCore-v3 is typically paired with 16x DMA engines to drive its corresponding SBUF bandwidth. In the above DMA copy, each DMA engine would be responsible for moving 128/16 = 8 chunks of 512 elements.

* HBM tensor [128, 512]: 512 is the inner-most (minor) dimension

.. _fig-hbm2sbuf-dma-copy:

.. image:: /nki/img/arch_images/nki-trn2-arch-8.png

In contrast, in a DMA transpose operation, we take an HBM tensor of opposite layout [512, 128]:

.. _fig-hbm2sbuf-dma-transpose:

.. image:: /nki/img/arch_images/nki-trn2-arch-9.png

In a DMA transposition, the most minor dimension of the source HBM tensor now becomes the partition dimension of the SBUF in destination. Compared to the above DMA copy operation where each DMA engine reads and writes an independent slice of 512 elements, DMA transpose requires all 16x DMA engines to work co-operatively to deliver the best throughput - these 16x DMA engines should write into a single ``[nl.par_dim(128), 16]`` SBUF tile in parallel at a time, where the 16 elements along free dimension must be contiguous. Having a multiple of 128 and a multiple of 16 in the output SBUF partition and inner-most free dimension sizes is a pre-requisite to achieve best DMA throughput efficiency possible with DMA transpose. However, it is not a functionality requirement - DMA transpose can flexible tile sizes for DMA transpose at the cost of DMA performance. 

HBM2SBUF DMA transpose is commonly seen in ML workloads where the data layout in HBM differs from the format needed by the initial compute engine that processes the data. For example, in the LLM decode phase, the K cache typically has an HBM layout of ``[seqlen, d_head]``, where ``seqlen`` and ``d_head`` are the sequence length and head dimensions respectively. However, when K is consumed by TensorE in the Q@K operator in self-attention, ``d_head`` is the contraction dimension of the matrix multiplication. Therefore, the most-minor d_head dimension in HBM should become the partition dimension to satisfy TensorE layout requirements (see :ref:`Layout Consideration #1 in the NKI Programming Guide <nki-pm-layout>`: Contraction dimension must map to partition dimension). Mapping most minor HBM tensor dimension to SBUF partition dimension is exactly an HBM2SBUF DMA transpose operation on Trainium2. 

In NKI, programmers can invoke an HBM2SBUF DMA transpose using the ``nisa.dma_transpose`` API.

.. code-block:: python

   import neuronxcc.nki as nki
   import neuronxcc.nki.language as nl
   import neuronxcc.nki.isa as nisa

   # hbm_src: nt.tensor[512, 128]
   # sbuf_dst: nt.tensor[128, 512]
   sbuf_dst = nisa.dma_transpose(src=hbm_src)

.. admonition:: Performance Consideration

   DMA transpose on Trainium2 can achieve up to 90% DMA throughput utilization given hardware-friendly tensor access patterns, compared to up to 100% throughput utilization for a DMA copy.

SBUF2SBUF DMA transpose
"""""""""""""""""""""""

SBUF2SBUF DMA transpose works in a similar fashion as HBM2SBUF transpose, where the most minor dimension of the input SBUF tensor, i.e., inner-most free dimension, becomes the partition dimension of the output SBUF tensor. Therefore, SBUF2SBUF DMA transpose is a way to swap partition and free axis of an SBUF tensor, an alternative to TensorE transpose.

The same ``nisa.dma_transpose`` API can be used to perform an SBUF2SBUF DMA transpose:

.. code-block:: python

   import neuronxcc.nki as nki
   import neuronxcc.nki.language as nl
   import neuronxcc.nki.isa as nisa

   # sbuf_src: nt.tensor[128, 128]
   # sbuf_dst: nt.tensor[128, 128]
   sbuf_dst = nisa.dma_transpose(src=hbm_src)

Performance Consideration. SBUF2SBUF transpose can achieve up to 50% of DMA throughput on Trainium2. Compared to TensorE transpose that is more performant but requires ScalarE/VectorE to evict the transposed output from PSUM back to SBUF, DMA transpose can read from and write to SBUF directly. Therefore, DMA transpose is particularly useful in operators that are ScalarE/VectorE bound, such as self attention.

.. _dge_arch:

Descriptor Generation Engine (DGE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Descriptor Generation Engine (DGE) is a new hardware block in NeuronCore-v3 that accelerates DMA descriptor generation to perform either DMA copy or transpose on the DMA engines. Each NeuronCore-v3 comes with two instances of DGE, which can be commanded through either SyncE or ScalarE sequencer. The figure below shows the connectivity of the DGE instances.

.. _fig-dge:

.. image:: /nki/img/arch_images/nki-trn2-arch-10.png

Prior to Trainium2, DMA descriptor generation was handled in two ways. They were either generated statically on the host when loading a NEFF onto a Neuron Device (i.e., static DMA), or created dynamically through custom kernels on GpsimdE during NEFF execution (i.e., software DGE). The static approach stored all descriptors in HBM, consuming valuable memory space that could otherwise be used for model parameters or computation data. The software-based approach used a portion of SBUF for storing descriptors generated during execution and occupies GpsimdE that could otherwise perform useful computation.

In comparison, the new hardware-based DGE in Trainium2 generates descriptors on demand without requiring additional memory storage. It also frees up GpsimdE to perform useful computation. Therefore, it is recommended to leverage hardware-based DGE on Trainium2 whenever possible to initiate a DMA transfer.

NKI programmers can invoke hardware-based DGE on NeuronCore-v3 using ``nisa.dma_copy`` and ``nisa.dma_transpose`` APIs, by setting ``dge_mode=nisa.dge_mode.hw_dge``. The compute engine to initiate a DGE command (Sync Engine or ScalarE) is currently determined by NKI compiler (subject to changes).

.. note::
   NeuronCore-v3 hardware DGE currently does not support indirect DMA operations (gather/scatter). Refer to nisa API documentation for detailed implementation guidelines.

.. admonition:: Performance Consideration

   When triggered from ScalarE, execution of the DGE-based DMA instruction could be hidden behind earlier compute instructions (such as ``nisa.activate()``) in program order, since DGE and the compute pipeline of ScalarE are independent hardware resources. Each DGE-based DMA instruction takes about 600 ns to execute on NeuronCore-v3.

