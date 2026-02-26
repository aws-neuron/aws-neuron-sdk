.. meta::
    :description: Guide for implementating MXFP4/8 matrix multiplication using NKI on AWS Neuron hardware.
    :keywords: MXFP8, MXFP4, Matrix Multiplication, NKI, Neuron
    :date-modified: 12/19/2025

MXFP Matrix Multiplication with NKI on AWS Neuron
===================================================

In this guide, you'll learn how to perform MXFP4/8 matrix multiplication, quantization, and Neuron's recommended best practices for writing MX kernels.


Before You start
-----------------

* Read the MX-related sections of the :ref:`Trainium 3 Architecture Guide for NKI <trainium3_arch>` and become familiar with basic matrix multiplication concepts on Neuron in the :doc:`Matrix Multiplication tutorial </nki/guides/tutorials/matrix_multiplication>`.

.. note::
    The code snippets in this guide are taken from the `tutorial code package <https://github.com/aws-neuron/nki-samples/tree/main/src/nki_samples/tutorials/mxfp-matmul>`_ which demonstrates how to execute all MX kernel examples from Torch. We recommend you browse and run the code as you read the tutorial.

What is MXFP4/8 Matrix Multiplication?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MXFP4/8 matrix multiplication uses microscaling (MX) quantization as defined in the OCP standard. Unlike traditional quantization that uses tensor- or channel-wide scale factors, microscaling calculates quantization scales from small groups of values. Specifically, groups of 32 elements along the matrix multiplication contraction dimension share the same 8-bit MX scale value.

This approach preserves significantly more information in quantized values by preventing high-magnitude outliers from "squeezing" the entire data distribution. The NeuronCore-v4 Tensor Engine performs matrix multiplication of MXFP4 or MXFP8 input matrices and dequantization with MX scales in a single instruction, achieving 4x throughput compared to BF16/FP16 matrix multiplication while outputting results in FP32 or BF16.

Layout and Tile Size Requirements
----------------------------------

Before diving into code examples of MX multiplication, it's important to review the layout and tile-size requirements of MX. MX quantized tensors are represented with separate data and scale tensors, each with distinct requirements.

Data Tensor
~~~~~~~~~~~~

Compared to BF16/FP32 matrix multiplication, the performance uplift from Matmul-MX comes from the ability to contract 4x more elements during one matmul operation as each TensorE processing element is able to perform four simultaneous, FP4/FP8, multiply-accumulate computations. This means the maximum effective contraction dimension has increased from 128 → 512. 

First, let's examine the tile-size constraints for MX so we can allocate the correct space for tensors. MX data is represented in NKI using quad (x4) packed data types (:doc:`float8_e5m2_x4 </nki/api/generated/nki.language.float8_e5m2_x4>`, :doc:`float8_e4m3fn_x4 </nki/api/generated/nki.language.float8_e4m3fn_x4>`, and :doc:`float4_e2m1fn_x4 </nki/api/generated/nki.language.float4_e2m1fn_x4>`, herein referred to collectively as ``MXFP_x4``). The ``float8_*_x4`` types are 32-bits wide and physically contain four ``float8`` elements. The ``float4_*_x4`` type is 16-bits wide and physically contains four ``float4`` elements. As expressed in ``_x4`` elements, the TensorE maximum tile sizes in NKI code continue to be given by the existing hardware constraints, summarized below.

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Matrix Type
     - Data Type
     - Implied Physical Size
     - Max Tile Size in Code
   * - Stationary
     - BF16
     - [128P, 128F]
     - [128P, 128F]
   * - Stationary
     - MXFP_x4
     - [512P, 128F]
     - [128P, 128F]
   * - Moving
     - BF16
     - [128P, 512F]
     - [128P, 512F]
   * - Moving
     - MXFP_x4
     - [512P, 512F]
     - [128P, 512F]

This means that we will allocate data tensors, of type ``MXFP_x4``, in our NKI code with the same shapes as we would for BF16/FP32, but it's implied they contain 4x more contraction elements as shown in the subsequent diagrams.

Now let's examine a BF16 tile destined to be quantized into a max-sized moving tile for Matmul-MX (``[128P, 512F] MXFP_x4``). Note that the following concepts are equally applicable to the stationary tile whose max size is ``[128P, 128F]``.

Since a 4x larger contraction dimension is supported we'll start with a BF16 tile of size ``[512, 512]`` as shown below. To help us in the subsequent step we'll also view it as being sectioned into 4 regions of 128 rows (i.e. reshaped as ``[4, 128, 512]``). This view is mathematical (i.e. not residing in any particular memory).

.. image:: /nki/img/deep-dives/mxfp84-matmul-guide-1.png
   :width: 50%
   :align: center

As explained in the :doc:`Trainium 3 Architecture Guide for NKI </nki/guides/architecture/trainium3_arch>` we must take 4 elements originating 128 apart on the contraction axis and pack them together on the SBUF free-dimension as shown below. We'll call this transformation "interleaving".

.. image:: /nki/img/deep-dives/mxfp84-matmul-guide-2.png
   :align: center

Notice the SBUF shape has become ``[128P, 2048F]``. In a subsequent code example we'll see that it's useful to view/reshape this as ``[128P, 512F, 4F]``, making it clear we have 512 groups of 4 packed elements.

Next, let's Quantize-MX this tile, which will preserve the layout but pack groups of 4 free-dimension elements into a single ``MXFP_x4`` element, as shown below. Note that Quantize-MX does not support an FP4 output but Matmul-MX does support FP4 input.

.. image:: /nki/img/deep-dives/mxfp84-matmul-guide-3.png
   :width: 50%
   :align: center

Notice the shape is now ``[128P, 512F]`` which is the max moving tile size we aimed for. But each ``MXFP_x4`` element, shown in red, physically contains four quantized elements from the original tile. Recall that each TensorE processing element ingests enough data to perform four, FP4/FP8 multiply-accumulate operations, which is why four elements from the original contraction axis must be packed together in this fashion.

With this understanding we'll state the space allocation rules for quantized ``MXFP_x4`` data tiles.

.. code-block:: none

    Unquantized Interleaved Data Tile = [P,F] BF16 in SBUF

    MX Quantized Data Tile = [P, F//4] MXFP_x4 in SBUF

Scale Tensor
~~~~~~~~~~~~~

Let's revisit the BF16 tile with the interleaved SBUF layout but this time with one of the ``[8P, 4F]`` scaling groups overlaid.

.. image:: /nki/img/deep-dives/mxfp84-matmul-guide-4.png
   :align: center

MX scales are represented using a ``UINT8`` tile containing one element for each scaling group.

As explained in the :doc:`Trainium 3 Architecture Guide for NKI </nki/guides/architecture/trainium3_arch>`, we view the partition-dimension of SBUF as being split into 4 quadrants of 32 partitions each. Scales must be placed in the quadrant from which the corresponding scaling group originated, as shown below.


.. image:: /nki/img/deep-dives/mxfp84-matmul-guide-5.png
   :width: 50%
   :align: center


Notice the allocated shape is ``[128P, 512F]`` despite the underlying useful shape being ``[16P, 512F]``. See the :doc:`quantize_mx API </nki/api/generated/nki.isa.quantize_mx>` for an example of how to improve memory usage by packing scales, from other quantized tensors, into the same allocation.

With this understanding we'll state the space allocation rules for quantized MX scale tiles.

.. code-block:: none

    Unquantized Interleaved Data Tile = [P,F] BF16 in SBUF

    If P <= 32 (Oversize optional)

    MX Quantized Scale = [P//8, F//4] UINT8 in SBUF

    If P > 32 (Oversize required)

    MX Quantized Scale = [P, F//4] UINT8 in SBUF

Basic Matmul-MX
----------------

This NKI example performs a single Matmul-MX using offline-quantized, max-sized input tiles. For simplicity, it assumes the MX *data* tiles in HBM already satisfy the layout requirements so they may be simply loaded straight into SBUF. The MX *scale* tiles require some shuffling. Note that subsequent examples, instead, show how to establish this layout yourself in SBUF.

.. literalinclude:: src/mxfp-matmul/mx_kernels.py
   :language: python
   :start-after: [start-kernel_offline_quantized_mx_matmul]
   :end-before: [end-kernel_offline_quantized_mx_matmul]

A few notes about the above example:

* The ``MXFP_x4`` packed data types are custom to NKI and are not supported in Torch. Therefore, we mimic the packed data using ``uint8`` in Torch and simply view it as ``MXFP_x4`` in the kernel, as shown.
* The ``load_scales_scattered()`` helper function reads contiguously packed offline scales from HBM and spreads them across partition-dim quadrants.
* The PSUM output tile is allocated with data type BF16 to indicate the desired output data type of the Matmul-MX. Note that Matmul-MX (:doc:`nki.isa.nc_matmul </nki/api/generated/nki.isa.nc_matmul_mx>`) supports both BF16 and FP32 output dtypes.

Let's also look at the host code which calls this kernel as all subsequent examples use the same structure.

.. literalinclude:: src/mxfp-matmul/mx_toplevel.py
   :language: python
   :start-after: [start-run_offline_quantized_matmul_mx_test]
   :end-before: [end-run_offline_quantized_matmul_mx_test]

* The ``generate_stabilized_mx_data()`` helper function is used to generate MX data on the host. "Stabilized" means the data is randomly generated but injected with certain properties to allow for lossless quantization/dequantization, including constraining the data to be in the FP4/8 range. It conveniently returns MX data as ``ml_dtypes`` FP4/FP8, the same data packed into ``uint`` to mimic the ``MXFP_x4`` packing (suitable for sending to a NKI kernel), MX scales, and a corresponding unquantized FP32 tensor. The input shape argument specifies the unquantized shape. The unquantized tensor is viewed as being in the required layout for MX operations. Therefore to generate an MX data tile of maximum size we must specify an unquantized free-dimension that is 4x larger. In this example the moving unquantized shape is ``[128P, 2048F]`` and the function will return a ``[128P, 512F]`` packed MX data tensor, as desired.
* ``nc_matmul_mx_golden()`` is a utility to mimic the hardware's Matmul-MX operation and is therefore useful for verifying the hardware output. It assumes the input tensors meet the SBUF layout requirements and the data tensor is packed to mimic ``MXFP_x4``. Hence it can directly accept MX data generated by ``generate_stabilized_mx_data()``.
* ``compare_and_print_results()`` uses ``numpy.allclose()`` to check data correctness and print the tensors to ``stdout``.
* Although this is a single-tile Matmul-MX, larger MX tensors can be multiplied by using the same tiling techniques shown in the non-MX :doc:`Matrix Multiplication tutorial </nki/guides/tutorials/matrix_multiplication>`.

Quantize-MX + Matmul-MX
-----------------------

Next we'll replace one of the Matmul-MX inputs with a tile that we quantize on the VectorE using Quantize-MX. Again, it assumes the interleaved SBUF layout requirement is already satisfied. The source data for Quantize-MX must be in SBUF (cannot be in PSUM).

The two main changes in this example are:

* The ``allocate_mx_tiles()`` helper function implements the data and scale tile allocation rules mentioned above.
* ``load_scales_scattered()`` is again used for the stationary scales but is unnecessary for the moving scales since Quantize-MX will correctly spread the data across SBUF partition-dim quadrants.

.. literalinclude:: src/mxfp-matmul/mx_kernel_utils.py
   :language: python
   :start-after: [start-allocate_mx_tiles]
   :end-before: [end-allocate_mx_tiles]

.. literalinclude:: src/mxfp-matmul/mx_kernels.py
   :language: python
   :start-after: [start-kernel_on_device_quantize_matmul_mx]
   :end-before: [end-kernel_on_device_quantize_matmul_mx]

Please see the code package for the host code that calls this kernel.

SBUF Layout Using Strided Access
--------------------------------

Here we present two techniques for establishing the interleaved layout required for MX operations. Both produce the same result but have different performance tradeoffs. Therefore it's useful to think of them as tools in a toolbox where you use the one that's appropriate for your given situation.

It's important to note that these techniques operate on unquantized tensors (BF16 in these examples) as the layout must be established before calling Quantize-MX. If you already have offline MX weights (already quantized), it's suggested you establish the required layout offline so you may perform a direct load to SBUF.

The techniques are first explained then followed by a combined code example.

VectorE/ScalarE Strided Access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we use either VectorE or ScalarE to write data to SBUF in the required layout. The simplest operation is a TensorCopy (shown below) but it's usually more performant to apply the strided access pattern to some prior useful computation already occurring on these engines.

For completeness the example loads an HBM tensor to SBUF prior to rearranging the data on-device using an SBUF-to-SBUF TensorCopy. The load is needed for this to be a standalone executable example but in practice it's expected your data would already be in SBUF from some previous operation. The TensorCopy strided access pattern is the key takeaway from this example.

Also note the TensorCopy source could be PSUM if you want to rearrange the data immediately after a prior matmul.

DMA Strided Access
~~~~~~~~~~~~~~~~~~~

Here we DMA a tensor from HBM to SBUF using a strided access pattern. It's conceptually similar to the above technique except the source of the copy is in HBM. This technique is typically significantly slower than on-device techniques but it can be useful in heavily compute-bound workloads where the DMA may overlap with compute.

Code
~~~~

This example demonstrates both techniques, selected by the ``use_tensor_copy`` argument. They are very similar but with slightly different read access patterns. It's useful to refer to the above layout diagrams as you read this code as the reshapes and access patterns directly correspond.

.. literalinclude:: src/mxfp-matmul/mx_kernel_utils.py
   :language: python
   :start-after: [start-copy_data_strided]
   :end-before: [end-copy_data_strided]

See the code package for an example kernel that calls ``copy_data_strided()`` to establish the interleaved layout for stationary and moving tiles, quantize both, and perform a Matmul-MX.

.. _nki-mxfp-scale-packing:

Packing Scale Values
--------------------

As discussed in `Scale Tensor`_, each element of a scale tensor corresponds to a group of 32 elements in the unquantized source tensor.
Each scaling group spans 8 partitions, with 4 free elements per partition, giving scale tensors a logical size of ``[P // 8, F // 4]``.
However, due to connectivity constraints between SBUF and VectorE, scale values must be placed in the same quadrant as their corresponding scaling group.
When the unquantized source tensor spans multiple partitions (i.e., ``src.shape[0] > 32``), the scale tensor has physical shape ``[P, F // 4]``.
Only the first 4 partitions (= 32 partitions per quadrant divided by 8 partitions per scaling group) of each quadrant are occupied, leaving the remaining 28 unused.

Quantize-MX allows you to utilize some of this space by packing scale values from multiple Quantize-MX calls.
Quantize-MX and Matmul-MX support writing/reading scales at an offset of 0, 4, 8, or 12 within each partition, allowing you to pack scale values from up to four tensors into a single tile.
To illustrate, consider the scale tile from `Scale Tensor`_ shown with and without scale packing:

.. image:: /nki/img/deep-dives/mxfp84-matmul-guide-6.drawio.png
   :align: center

Code
~~~~

This example demonstrates how to pack scale values from multiple Quantize-MX calls into a single tensor in SBUF, as mentioned in the :ref:`Trainium3 Architecture Guide <arch-trn3-quad-mxfp>`.
We use tensor slicing to control the offset into each quadrant at which Quantize-MX writes scale values.

.. literalinclude:: src/mxfp-matmul/mx_kernels.py
   :language: python
   :start-after: [start-kernel_copy_strided_quantize_matmul_mx_packed_scale]
   :end-before: [end-kernel_copy_strided_quantize_matmul_mx_packed_scale]


Additional Tips
----------------

* It's important to plan where in your design you'll pay the cost of interleaving the data. Ideally you minimize the cost by finding existing, prior compute on which you can apply the strided access pattern. Or find existing compute against which you can overlap the interleave process. For offline MX weights prepare the layout offline on CPU so you may load the data to SBUF directly in a contiguous/unstrided fashion.

* As with all compute on Neuron, it's generally performant to spread it across multiple engines operating in parallel. Given that Quantize-MX runs exclusively on the VectorE a bit more care may be needed to alleviate VectorE contention by becoming familiar with operations that may be relegated other engines, like ScalarE.

* The TensorE operates at double the clock frequency of VectorE, therefore Matmul-MX produces data at double the rate that Quantize-MX can consume it. It may seem that the TensorE could be back-pressured in a situation where a Matmul-MX quickly feeds a subsequent Matmul-MX (since you must Quantize-MX in between at half the speed), but that only happens for small tensors. Larger tensors require tiled matrix multiplication which inherently reuses input (quantized) tiles, allowing time for prior matmul output data to be quantized.

Matmul-MX supports PE-tiling (row-tiling only) where matmuls with a small (<= 64) contraction-dimension (partition-dimension) may be parallelized on the TensorE. This becomes more relevant for MX since a 4x-larger effective contraction-dimension is supported, meaning it's useful for an ``MXFP_x4`` contraction-dimension <= 64 or an equivalent unquantized contraction-dimension <= 256.

Executing the Code
------------------
After downloading the `tutorial code package <https://github.com/aws-neuron/nki-samples/tree/main/src/nki_samples/tutorials/mxfp-matmul>`_ to your Trainium3 Neuron environment, simply execute it as follows and observe the sample output.

.. code-block:: bash

  $ python3 mx_toplevel.py

  =====================================================================================
      OFFLINE_QUANTIZED_MX_MATMUL - stationary <float8_e5m2> @ moving <float8_e5m2>
  =====================================================================================

  Result shape: (128, 512)

  np.allclose pass? True

  Device Output:
  [[0.02526855 0.59765625 1.15625   ] ... [-0.09033203 -0.10888672 -0.84375   ]]
  ...
  [[ 0.25585938  0.18554688 -0.546875  ] ... [-0.71875    -0.6015625  -0.46484375]]

  Golden:
  [[0.02535721 0.5957752  1.1556101 ] ... [-0.09036541 -0.10906862 -0.8448767 ]]
  ...
  [[ 0.2551025   0.1856966  -0.54681885] ... [-0.71797514 -0.6026518  -0.4641544 ]]


  =========================================================================================
      OFFLINE_QUANTIZED_MX_MATMUL - stationary <float4_e2m1fn> @ moving <float4_e2m1fn>
  =========================================================================================

  Result shape: (128, 512)

  np.allclose pass? True

  Device Output:
  [[-0.02038574  0.02648926  0.10351562] ... [-0.25        0.02404785  0.08154297]]
  ...
  [[ 0.234375  -0.0456543  1.140625 ] ... [ 1.1015625   0.04833984 -0.17675781]]

  Golden:
  [[-0.02036181  0.02647817  0.10362364] ... [-0.24955288  0.02399684  0.08132255]]
  ...
  [[ 0.23485765 -0.04565394  1.1424086 ] ... [ 1.0981529   0.04839906 -0.17722145]]


  ========================================================================================
      ON_DEVICE_QUANTIZE_MATMUL_MX - stationary <float4_e2m1fn> @ moving <float8_e5m2>
  ========================================================================================

  Result shape: (128, 512)

  np.allclose pass? True

  Device Output:
  [[-0.12792969  0.02685547 -0.19140625] ... [ 0.05883789 -0.01916504 -0.66796875]]
  ...
  [[ 0.03198242 -0.24316406 -0.1640625 ] ... [ 0.06591797 -0.11914062  0.6015625 ]]

  Golden:
  [[-0.1284121   0.02687968 -0.19178611] ... [ 0.05882631 -0.01915852 -0.666565  ]]
  ...
  [[ 0.03191248 -0.24304396 -0.16389877] ... [ 0.06606946 -0.11931092  0.60205466]]


  ======================================================================================
      ON_DEVICE_QUANTIZE_MATMUL_MX - stationary <float8_e5m2> @ moving <float8_e5m2>
  ======================================================================================

  Result shape: (128, 512)

  np.allclose pass? True

  Device Output:
  [[ 0.02832031 -0.29296875  0.04394531] ... [-0.13671875 -0.00704956 -0.47265625]]
  ...
  [[ 0.03442383 -0.75        0.11572266] ... [ 0.86328125 -0.00735474  0.33007812]]

  Golden:
  [[ 0.02831857 -0.29297137  0.04390652] ... [-0.13685682 -0.00703458 -0.47168562]]
  ...
  [[ 0.03451066 -0.7511592   0.11560257] ... [ 0.86369723 -0.00734489  0.3300762 ]]


  ================================================================
      COPY_STRIDED_TENSOR_COPY - <float8_e5m2> @ <float8_e5m2>
  ================================================================

  Result shape: (128, 512)

  np.allclose pass? True

  Device Output:
  [[ 0.56640625 -1.28125     0.26953125] ... [ 0.5859375   0.31054688 -0.60546875]]
  ...
  [[ 1.2421875 -0.859375  -1.140625 ] ... [-0.06542969  0.11425781  0.6015625 ]]

  Golden:
  [[ 0.5663527  -1.2832397   0.26900524] ... [ 0.5861912  0.3109728 -0.6038357]]
  ...
  [[ 1.2426924  -0.85944945 -1.1438001 ] ... [-0.0654989   0.11429967  0.6028823 ]]


  ============================================================
      COPY_STRIDED_DMA - <float8_e5m2> @ <float8_e5m2>
  ============================================================

  Result shape: (128, 512)

  np.allclose pass? True

  Device Output:
  [[ 0.32421875  0.43359375 -0.09814453] ... [ 0.82421875 -2.171875    0.71484375]]
  ...
  [[-0.47070312 -0.734375    0.09765625] ... [ 1.328125   -1.09375    -0.32226562]]

  Golden:
  [[ 0.32461044  0.43410686 -0.09810834] ... [ 0.82437325 -2.1703691   0.71522826]]
  ...
  [[-0.47003102 -0.733371    0.09745546] ... [ 1.3250915  -1.0969493  -0.32166338]]

