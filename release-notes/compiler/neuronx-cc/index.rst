.. _neuronx-cc-rn:

Neuron Compiler (``neuronx-cc``) release notes
==============================================
.. contents:: Table of Contents
   :local:

   :depth: 2



Neuron Compiler [2.13.68.0]
----------------------------
Date: 04/10/2024

* This release fixes hang issues related to Triton Inference Server.


Neuron Compiler [2.13.66.0]
----------------------------
Date: 04/01/2024

* This release introduces a new ``--enable-mixed-precision-accumulation`` compiler option. This option instructs the compiler to perform intermediate calculations of reduction operators (such as the dot or reduce operators) in FP32 regardless of the operation's defined datatype. The final result of the operator will be cast from FP32 to the model-designated datatype (e.g., BF16). This helps to improve the operator's resulting acccuracy.


Neuron Compiler [2.12.68.0]
----------------------------
Date: 01/18/2024

* Patch release with bug fixes.


Neuron Compiler [2.12.54.0]
-------------------------
Date: 12/21/2023

* The compiler now generates instructions to check if a model references an embedding table with an illegal index. The check is made at model execution time. If an attempted invalid table index is encountered, the model execution will continue and the user will see an error similar to:

      WARNING: Received notification generated at runtime: failed to run scatter/gather (indirect memory copy with branch_label_id = xx), due to out-of-bound access.

When this occurs, users are encouraged to review the model's gather/scatter input values to determine if there is a coding error.


Neuron Compiler [2.11.0.35]
-------------------------
Date: 11/17/2023

* This release addresses performance related issues when training through ``neuronx-nemo-megatron`` library.


Neuron Compiler [2.11.0.34]
-----------------------------
Date: 10/26/2023

* This release introduces the option-argument ``llm-training`` to the existing ``--distribution_strategy`` compiler option. This option-argument allows the compiler to make specific optimizations related to training distributed models. This new option-argument is equivalent to the previously introduced ``nemo`` option-argument, which will be deprecated in a future release.


Neuron Compiler [2.10.0.35]
-----------------------------
Date: 09/26/2023

* This release addresses a compilation regression for certain configurations of Llama and Llama-2 inference models when it fails compilation with this error "IndirectLoad/Save requires contiguous indirect access per partition" .

There is still a known issue for some configurations of the model with the error "Too many instructions after unroll for function sg0000" . To mitigate this, recompile using the ``--optlevel 1 (-O1)`` option. A complete fix will be coming in the future release which will not require this option

Neuron Compiler [2.10.0.34]
-----------------------------
Date: 09/15/2023

* This release introduces a new ``--optlevel (-O)`` compiler option. This option allows the user to balance between compile-time and optimizations performed.
  Three levels are supported. Level ``--optlevel 1 (-O1)`` aims to minimize compile-time and allow for a more rapid model development cycle. Model execution
  time may be reduced. Level ``--optlevel 3 (-O3)`` performs whole-model optimization. This level will deliver the best performance however there will be longer
  compile-times and the compiler will use more host DRAM, potentially requiring a larger instance to compile the model.
  The default is ``--optlevel 2 (-O2)`` which provides a balance between model performance and compile time. 

  The previous ``—enable-experimental-O1`` flag introduced in the 02/08/2023 Neuron Compiler [2.4.0.21] release is now deprecated. Using this flag
  will generate a message similar to:
      WARNING: Option —enable-experimental-O1 is deprecated and will be removed in a future release." Use ``--optlevel 1 (-O1)`` instead.

Neuron Compiler [2.9.0.16]
-----------------------------
Date: 08/28/2023

* This release fixes an issue where any initial seed passed into the Random Number Generator operator was not honored. The RngBitGenerator operator now correctly accepts and uses setting the seed. Note that the current RNG implementation only supports 32-bit seeds.

Neuron Compiler [2.8.0.25]
-----------------------------
Date: 07/19/2023

* This release introduces a new optional ``--distribution_strategy`` compiler option. This option informs the compiler what type of distributed APIs are used to shard the model and allows the compiler to make API-specific optimizations. Currently following option-arguments are supported: ``nemo``.

Neuron Compiler [2.7.0.40]
-----------------------------
Date: 06/14/2023

* This release introduces a new ``--enable-saturate-infinity`` compiler option. A computation that can generate +/- infinity is at a high
  risk of generating Not-a-Number (NaN) values when the infinity value is used in subsequent computations. This option helps avoid this
  by converting +Inf/-Inf values to MAX/MIN_FLOAT before operations that could produce NaN values for +Inf/-Inf inputs on the target
  architecture. While this option helps to avoid NaN values, there is a potential performance degradation that occurs during model
  execution when this conversion is enabled.
  
Neuron Compiler [2.6.0.19]
-----------------------------
Date: 05/01/2023

* This release introduces a new ``model-type`` option argument: ``unet-inference``.
  This option instructs the compiler to perform model-specific optimizations that produce executable models with improved performance
  on the specified target instance.
  
* Added support for the HLO operator ``BitcastConvertType`` and also added support for ``TopK`` (sampling mode) operator.

Neuron Compiler [2.5.0.28]
-----------------------------
Date: 03/28/2023

* This release introduces the ``trn1n`` option argument to the compiler ``target`` option to specify that it should
  generate code for a trn1n instance type. Example usage: ``neuronx-cc compile --target=trn1n ...``
  
* The compiler's usage message now includes the ``inf2`` option argument.

* A new 8-bit floating point data type, ``fp8_e4m3``, is now supported and can be specificed using the ``auto-cast-type`` option.
  This instructs the compiler to convert the FP32 operations selected via the ``--auto-cast`` option to a signed FP8 size
  with 4-bit exponent and 3-bit mantissa. Care must be taken to ensure that the down-casted values are representable within the 8-bit data range.

Neuron Compiler [2.4.0.21]
-----------------------------
Date: 02/24/2023

* This release introduces the ``inf2`` option argument to the compiler ``target`` option to specify that it should
  generate code for an inf2 instance type. Example usage: ``neuronx-cc compile --target=inf2 ...``
  The ``inf2`` option argument does not appear in the compiler's usage message. It will be added in the next release.

Neuron Compiler [2.4.0.21]
-----------------------------
Date: 02/08/2023

* Added support for the following HLO operators: ``SelectAndScatter``.
* Beta: ``--enable-experimental-O1`` flag: This option reduces the compile-time with a neglible impact on model execution performance.
  It allows the compiler to execute compiler passes in parallel to perform the compilation. By default the compiler uses 8 processes.
  This can be changed via the CLI option ``--num-parallel-jobs``. This option is expected to become the default in a future SDK release.

Neuron Compiler [2.3.0.4]
-----------------------------
Date: 12/09/2022

* Added support for the following HLO operators: ``rev (reverse)``.
* The ``pow()`` function can now handle both integer and floating-point exponents.
* Optimization enhancements and bug fixes to improve model execution performance.


Neuron Compiler [2.2.0.73]
-----------------------------
Date: 10/27/2022

* Adding support for the following HLO operators: ``LogicalNot``, ``atan2`` and ``DynamicUpdateSlice`` (for constant index).

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


Supported Operators
-------------------

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
| Atan2                   |                                           |
+-------------------------+-------------------------------------------+
| Batchnorm               |                                           |
+-------------------------+-------------------------------------------+
| Batchnormgrad           |                                           |
+-------------------------+-------------------------------------------+
| Batchnorminference      |                                           |
+-------------------------+-------------------------------------------+
| BitcastConvertType      |                                           |
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
| DynamicUpdateSlice      | Supports only for constant index          |
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
| LogicalNot              |                                           |
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
| Rev (reverse)           |                                           |
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
| SelectAndScatter        |                                           |
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

