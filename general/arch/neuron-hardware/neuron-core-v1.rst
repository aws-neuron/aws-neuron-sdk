.. _neuroncores-v1-arch:


NeuronCore-v1 Architecture
--------------------------

NeuronCore-v1 is the first generation of the NeuronCore engine, powering
the Inferentia NeuronDevices. Each NeuronCore-v1 is a fully-independent
heterogenous compute-unit, with 3 main engines (Tensor/Vector/Scalar
Engines), and on-chip software-managed SRAM memory, for
maximizing data locality (compiler managed, for maximum data locality
and optimized data prefetch).

.. image:: /images/nc-v1.png


The ScalarEngine is optimized for scalar-computations, in which every
element of the output is dependent on one element of the input, e.g.
non-linearities like GELU, SIGMOID or EXP. The ScalarEngine is highly
parallelized, and can process 512 floating point operations per cycle.
It can handle various data-types, including FP16, BF16, FP32, INT8,
INT16 and INT32. 

The VectorEngine is optimized for vector-computations,
in which every element of the output is dependent on multiple input
elements. Examples include ‘axpy’ operations (Z=aX+Y), Layer
Normalization, Pooling operations, and many more. The VectorEngine is
also highly parallelized, and can perform 256 floating point operations
per cycle. It can handle various data-types, including FP16, BF16, FP32,
INT8, INT16 and INT32.

The TensorEngine is based on a power-optimized systolic-array which is
highly optimized for tensor computations (e.g. GEMM, CONV, Reshape,
Transpose), and supports mixed-precision computations (FP16/BF16/INT8
inputs, FP32/INT32 outputs). Each NeuronCore-v1 TensorEngine delivers 16
TFLOPS of FP16/BF16 tensor computations.