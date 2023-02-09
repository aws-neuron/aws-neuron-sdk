.. _neuronx-cc-training-mixed-precision:

Mixed Precision and Performance-accuracy Tuning (``neuronx-cc``)
================================================================

.. contents:: Table of contents
   :local:
   :depth: 2

Overview
--------

The Neuron Compiler supports machine learning models with FP32, TF32, FP16 and BF16 (Bfloat16) tensors and operators. The Neuron hardware supports a mix of 32, 16, and 8 bit datatypes. This guide explains how to apply the available auto-cast methods and their performance / accuracy trade-offs when compiling a model with Neuron.

.. note:: Neuron Compiler support for INT8 is planned for a future Neuron SDK release. See `Neuron Compiler: Enable Neuron INT8 support <https://github.com/aws/aws-neuron-sdk/issues/36>`_ for details.

Neuron Hardware
---------------

The Neuron v2 hardware supports matrix multiplication using FP16, BF16, TF32, and FP32 on its matrix multiply ("matmult") engine, and accumulations using FP32. Operators such as activations or vector operations are supported using FP32, TF32, FP16, and BF16. Supporting FP16 and BF16 allows Neuron to have significantly higher performance than executing everything as FP32.


Performance-accuracy tradeoffs
------------------------------

**By default**, the Neuron Compiler will **automatically cast FP32 matrix multiplication operations to BF16**. The remaining operations are performed in the data type specified by the model. The Neuron Compiler provides CLI options that direct the compiler to cast to other data types, thereby giving the ability to choose an accuracy-to-performance tradeoff in model execution. Deciding what CLI settings to use will be application specific and may require some experimentation. See :ref:`Neuron Compiler CLI Reference Guide<neuron-compiler-cli-reference-guide>` for details.


What is the difference between  Data Types?
-------------------------------------------

The NeuronCore v2 support multiple data types (see :ref:`NeuronCore v2 Data Types<neuron-data-types-v2>`). Each data type provides benefits and drawbacks due to its dynamic range and numeric precision.

+------+-----------+----------+--------------------------------------------------------+---------------------------------------------------+
| Type | Minimum   | Maximum  | Strength                                               | Weakness                                          |
+======+===========+==========+========================================================+===================================================+
| FP16 | -65504    | 65504    |	Numeric Precision, High granularity, Mid-range numbers | Low range, medium precision                       |
+------+-----------+----------+--------------------------------------------------------+---------------------------------------------------+
| BF16 | -3.40E+38 | 3.40E+38 |	Dynamic Range, Extremely small/large numbers           | Low precision                                     |
+------+-----------+----------+--------------------------------------------------------+---------------------------------------------------+
| TF32 | -3.40E+38 | 3.40E+38 |	Dynamic Range, Extremely small/large numbers           | Medium precision                                  |
+------+-----------+----------+--------------------------------------------------------+---------------------------------------------------+
| FP32 | -3.40E+38 | 3.40E+38 | N/A                                                    | Larger model size, potentially slower computation |
+------+-----------+----------+--------------------------------------------------------+---------------------------------------------------+

* FP16 provides a high density of representable values that are neither extremely small or extremely large. The density of representable values within the range is approximately an order of magnitude greater than BF16.

  * Conversion from FP32 to FP16 will perform well when values are relatively small but non-extreme (either very small or very large).
  * Conversion from FP32 to FP16 will perform badly if the original FP32 values are outside of the range of FP16. This will produce inf/-inf values and may result in NaN depending on the operation.

* BF16 provides a wider range of representable values which includes both very small and very large values. However, the overall density of representable values is usually lower than FP16 for more non-extreme values. The range is nearly identical to the range of FP32 but because the number of bits is halved, this means the individual values are sparse.

  * Conversion from FP32 to BF16 will perform well when the values are well-distributed throughout the range. Since BF16 covers the entire FP32 range, this means each original value can map to a relatively close downcast value.
  * Conversion from FP32 to BF16 will perform badly when fine granularity is needed. Since BF16 granularity is sacrificed for greater range it will almost always map worse to values that are within the FP16 range.

Should I downcast operations to smaller Data Types?
---------------------------------------------------

This choice here is driven entirely by accuracy vs performance tradeoff. Casting operations to smaller 16-bit data types will provide a significant performance benefit but may end up sacrificing accuracy.

The compiler uses BF16 casting **by default** for matrix multiplication operations. The speedup from casting operations gives a significant performance boost and the range of representable values in BF16 allows for more safety compared to FP16 when the possible numeric range of input values is unknown.

The Neuron Compiler’s  :option:`--auto-cast` and :option:`--auto-cast-type` CLI options are used to direct the compiler to perform alternate casting operations. See the detailed list of the options in :ref:`Neuron v2 Compiler CLI Reference Guide<neuron-compiler-cli-reference-guide>`.

It is recommended that you start with compiling the model to achieve high performance (default), you can then test the accuracy of the application and, if needed, try the next higher precision casting option until the desired accuracy and performance are achieved.

The option combinations to consider in a typical flow are:


+---------------------------------------------------------+--------------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------+
| Compiler autocast                                       | Options    Effect                                                        | Performance                                         | Accuracy                                        |
+=========================================================+==========================================================================+=====================================================+=================================================+
| ``--auto-cast all --auto-cast-type bf16``               | Best performance at the expense of precision                             | Performance *decreases* as you move down the table  | Accuracy *increases* as you move down the table |
+---------------------------------------------------------+                                                                          +                                                     |                                                 |
| ``--auto-cast matmul --auto-cast-type bf16``  (default) |                                                                          |                                                     |                                                 |
+---------------------------------------------------------+--------------------------------------------------------------------------+                                                     |                                                 |
| ``--auto-cast all —-auto-cast-type fp16``                | Best performance at the expense of dynamic range                        |                                                     |                                                 |
+---------------------------------------------------------+                                                                          +                                                     |                                                 |
| ``--auto-cast matmul --auto-cast-type fp16``            |                                                                          |                                                     |                                                 |
+---------------------------------------------------------+--------------------------------------------------------------------------+                                                     |                                                 |
| ``--auto-cast all —-auto-cast-type tf32``                | Balance of performance, dynamic range, and precision                    |                                                     |                                                 |
+---------------------------------------------------------+                                                                          +                                                     |                                                 |
| ``--auto-cast matmult --auto-cast-type tf32``           |                                                                          |                                                     |                                                 |
+---------------------------------------------------------+--------------------------------------------------------------------------+                                                     |                                                 |
| ``--auto-cast none``                                    | Disables all auto-casting, using the data types defined within the model |                                                     |                                                 |
+---------------------------------------------------------+--------------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------+

Note that compiler has to preserve the input/output (i/o) tensor types requested by Framework, therefore no casting is done on the i/o tensors. Additional speedup can be obtained by casting them in the Framework prior to compilation.

To learn how to configure the compiler options from within your application’s framework, please see:

* :ref:`Developer Guide for Training with PyTorch Neuron <pytorch-neuronx-programming-guide>`
