.. _neuronx-cc-rn:

Neuron Compiler (``neuronx-cc``) release notes
==============================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Neuron Compiler [2.1.0.76]
-----------------------------
Date: 10/5/2022


The Neuron Compiler is an Ahead-of-Time compiler that accelerates models for
execution on NeuronCores. This release supports compiling models for training
on a Trn1 instance using Pytorch Neuron. Users typically access the compiler via
the Framework to perform model compilation, although it can also be run
as a command line tool (*neuronx-cc*).


The Neuron Compiler supports compiling models for mixed precision calculations. 
The trn1 hardware supports matrix multiplication using FP16, BF16, and FP32 on
its Matrix Multiplication Engine, and accumulations using FP32. Operators such as 
activations or vector operations are supported using FP16, BF16, and FP32.
Tensor transpose can be accomplished in FP16, BF16, FP32, or TF32 datatypes.
By default, scalar and vector operations on FP32 values will be done in FP32,
while matrix multiplications are cast to BF16 and transpose operations are cast to FP32.
This default casting will generate the highest performance for a FP32 trained model.

By default, the compiler will target maximum performance by automatically casting
the model to mixed precision. It also provides an option (``--auto-cast``) that
allows the user to make tradeoffs between higher performance and optimal accuracy.
The decision on what option argument to use with the ``--auto-cast`` option will be
application specific. Compiler CLI options can be passed to the compiler via the framework.


Supported Operators
^^^^^^^^^^^^^^^^^^^

The following XLA operators are supported by the Neuron Compiler. 
Future releases will broaden model support by providing additional XLA operators defined in
https://www.tensorflow.org/xla/operation_semantics.

The list of supported operators can also be retrieved from the command line using :ref:`neuronx-cc list-operators<neuronx-cc-list-operators>`.

+-------------------------+-------------------------------------------+
| Supported XLA Operators | Notes                                     |
+=========================+===========================================+
| Abs                     |                                           |
+-------------------------+-------------------------------------------+
| Add                     |                                           |
+-------------------------+-------------------------------------------+
| Allgather               |                                           |
+-------------------------+-------------------------------------------+
| Allreduce               |                                           |
+-------------------------+-------------------------------------------+
| Batchnorm               |                                           |
+-------------------------+-------------------------------------------+
| Batchnormgrad           |                                           |
+-------------------------+-------------------------------------------+
| Batchnorminference      |                                           |
+-------------------------+-------------------------------------------+
| Broadcast               |                                           |
+-------------------------+-------------------------------------------+
| BroadcastInDim          |                                           |
+-------------------------+-------------------------------------------+
| Ceil                    |                                           |
+-------------------------+-------------------------------------------+
| Clamp                   |                                           |
+-------------------------+-------------------------------------------+
| Compare                 |                                           |
+-------------------------+-------------------------------------------+
| Concatenate             |                                           |
+-------------------------+-------------------------------------------+
| Constant                |                                           |
+-------------------------+-------------------------------------------+
| ConstantLiteral         |                                           |
+-------------------------+-------------------------------------------+
| ConvertElementType      |                                           |
+-------------------------+-------------------------------------------+
| Cos                     |                                           |
+-------------------------+-------------------------------------------+
| Customcall              |                                           |
+-------------------------+-------------------------------------------+
| Div                     |                                           |
+-------------------------+-------------------------------------------+
| Dot                     |                                           |
+-------------------------+-------------------------------------------+
| DotGeneral              |                                           |
+-------------------------+-------------------------------------------+
| Eq                      |                                           |
+-------------------------+-------------------------------------------+
| Exp                     |                                           |
+-------------------------+-------------------------------------------+
| Floor                   |                                           |
+-------------------------+-------------------------------------------+
| Gather                  | Supports only disjoint start_index_map    |
|                         | and remapped_offset_dims                  |
+-------------------------+-------------------------------------------+
| Ge                      |                                           |
+-------------------------+-------------------------------------------+
| GetTupleElement         |                                           |
+-------------------------+-------------------------------------------+
| Gt                      |                                           |
+-------------------------+-------------------------------------------+
| Iota                    |                                           |
+-------------------------+-------------------------------------------+
| Le                      |                                           |
+-------------------------+-------------------------------------------+
| Log                     |                                           |
+-------------------------+-------------------------------------------+
| LogicalAnd              |                                           |
+-------------------------+-------------------------------------------+
| Lt                      |                                           |
+-------------------------+-------------------------------------------+
| Max                     |                                           |
+-------------------------+-------------------------------------------+
| Min                     |                                           |
+-------------------------+-------------------------------------------+
| Mul                     |                                           |
+-------------------------+-------------------------------------------+
| Ne                      |                                           |
+-------------------------+-------------------------------------------+
| Neg                     |                                           |
+-------------------------+-------------------------------------------+
| Pad                     |                                           |
+-------------------------+-------------------------------------------+
| Pow                     | Exponent argument must be a compile-time  |
|                         | integer constant                          |
+-------------------------+-------------------------------------------+
| Reduce                  | Min, Max, Add and Mul are the only        |
|                         | supported computations. Init_values must  |
|                         | be constant                               |
+-------------------------+-------------------------------------------+
| Reshape                 |                                           |
+-------------------------+-------------------------------------------+
| RngBitGenerator         | Ignores user seed                         |
+-------------------------+-------------------------------------------+
| RngUniform              |                                           |
+-------------------------+-------------------------------------------+
| Rsqrt                   |                                           |
+-------------------------+-------------------------------------------+
| Scatter                 |                                           |
+-------------------------+-------------------------------------------+
| Select                  |                                           |
+-------------------------+-------------------------------------------+
| ShiftRightLogical       |                                           |
+-------------------------+-------------------------------------------+
| Sign                    |                                           |
+-------------------------+-------------------------------------------+
| Sin                     |                                           |
+-------------------------+-------------------------------------------+
| Slice                   |                                           |
+-------------------------+-------------------------------------------+
| Sqrt                    |                                           |
+-------------------------+-------------------------------------------+
| Sub                     |                                           |
+-------------------------+-------------------------------------------+
| Tanh                    |                                           |
+-------------------------+-------------------------------------------+
| Transpose               |                                           |
+-------------------------+-------------------------------------------+
| Tuple                   |                                           |
+-------------------------+-------------------------------------------+


Known issues
^^^^^^^^^^^^

-  The Random Number Generator operation can be passed an initial seed
   value, however setting the seed is not supported in this release.
-  The exponent value of the pow() function must be a compile-time
   integer constant.
-  The compiler treats INT64 datatypes as INT32 by truncating the
   high-order bits. If possible, cast these values to 32 bits .
-  Model compilation time is proportional to the model size and
   operators used. For some larger NLP models it may be upwards of 30
   minutes.
