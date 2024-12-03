.. _neuroncores-v3-arch:

NeuronCore-v3 Architecture
--------------------------

NeuronCore-v3 is the third-generation NeuronCore that powers Trainium2 devices. It is a fully-independent heterogenous compute 
unit consisting of 4 main engines: Tensor, Vector, Scalar, and GPSIMD, with on-chip software-managed SRAM memory to maximize data 
locality and optimize data prefetch. The following diagram shows a high-level overview of the NeuronCore-V3 architecture.

.. image:: /images/architecture/NeuronCore/nc-v3.png
    :align: center
    :width: 250
|
NeuronCore-v3 is made up of the following components:

On-chip SRAM 
""""""""""""
Each NeuronCore-v3 has a total of 28MB of on-chip SRAM. NeuronCore-v3 on-chip SRAM is software-managed to maximize data locality 
and optimize data prefetch. 

Tensor Engine
"""""""""""""

Tensor engines are based on a power-optimized systolic array. They are highly optimized for tensor computations such as GEMM, CONV, and 
Transpose. Tensor Engines support mixed-precision computations, including cFP8, FP16, BF16, TF32, and FP32 inputs and outputs. 
A NeuronCore-v3 Tensor Engine delivers 158 cFP8 TFLOPS, and 79 BF16/FP16/TF32 TFLOPS of tensor computations. 

Like NeuronCore-v2, NeuronCore-v3 supports control flow, dynamic shapes, and programmable rounding mode (RNE & Stochastic-rounding). 
NeuronCore-v3 also supports adjustable exponent biasing for the cFP8 data type.
   
The NeuronCore-v3 Tensor Engine also supports Structured Sparsity, delivering up to 316 TFLOPS of cFP8/FP16/BF16/TF32 
compute. This is useful when one of the input tensors to matrix multiplication exhibits a M:N sparsity pattern, where only M elements 
out of every N contiguous elements are non-zero. NeuronCore-v3 supports several sparsity patterns, including 4:16, 4:12, 4:8, 2:8, 
2:4, 1:4, and 1:2. 

Vector Engine
""""""""""""""

Optimized for vector computations, in which every element of the output is dependent on multiple input elements. Examples include 
axpi operations (Z=aX+Y), Layer Normalization, and Pooling operations. 

Vector Engines are highly parallelized, and deliver a total of 1 TFLOPS of FP32 computations. NeuronCore-v3 Vector Engines can handle 
various data-types, including cFP8, FP16, BF16, TF32, FP32, INT8, INT16, and INT32. 

Scalar engine
"""""""""""""

Optimized for scalar computations in which every element of the output is dependent on one element of the input. Scalar Engines are 
highly parallelized, and deliver a total of 1.2 TFLOPS of FP32 computations. NeuronCore-v3 Scalar Engines support multiple data 
types, including cFP8, FP16, BF16, TF32, FP32, INT8, INT16, and INT32.

GPSIMD engine
"""""""""""""

Each GPSIMD engine consists of eight fully-programmable 512-bit wide vector processors. They can execute general purpose C-code and 
access the embedded on-chip SRAM, allowing you to implement custom operators and execute them directly on the NeuronCores.

