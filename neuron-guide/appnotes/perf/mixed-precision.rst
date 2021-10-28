.. _mixed-precision:

Mixed precision and performance-accuracy tuning
===============================================

.. contents::
   :local:
   :depth: 2

The Neuron Compiler supports machine learning models with FP32,
FP16 and BF16 (Bfloat16) tensors and operators. The Neuron hardware supports a
mix of 32 and 16 bit datatypes.
The available auto-cast methods and their performance / accuracy trade-offs
are explained in this document.

Neuron Hardware
-------------------

The Neuron hardware supports matrix multiplication using FP16 or BF16 on its Matmult Engine, and
accumulations using FP32.
Similarly, operators such as activations or vector operations
are supported using FP16, BF16 and FP32.
Neuron supports tensor transpose in two ways - by fast matrix
multiplication in FP16/BF16 or by slower byte-by-byte data movements.


Performance-accuracy tradeoffs for models trained in FP32
---------------------------------------------------------

Models that are trained using FP32 data types can be deployed on Neuron
through ahead of time compilation using the :ref:`Neuron Compiler <neuron_cli>`.


**By default**, the Neuron Compiler will **cast all FP32 tensors, 
weights and operations to BF16**. Only partial sums are left in FP32. The default, casting will generate the highest
performance for a FP32 trained model.

Using the ``--fast-math`` CLI option, you can choose the right 
tradeoff between performance and accuracy. The tradeoff usually is between achieving high performance or optimal accuracy, and decision what settings to use will be application specific.

It is recommended that the you start with compiling the model to achieve the high performance (default), you can then 
test the accuracy of the application and, if needed, try the next higher precision casting option until the desired 
accuracy and performance are achieved. A typical flow can be:

1. You can compile without options (default) or with ``--fast-math all`` which will optimize for performance.

2. If accuracy is not sufficient you can try ``--fast-math fp32-cast-matmult``  

3. If accuracy is not sufficient you can try ``--fast-math fp32-cast-matmult no-fast-relayout``

4. If accuracy is not sufficient you can try ``--fast-math none`` which will optimize for accuracy .

 
Between step 2 and step 3, and between step 3 and step 4 you have additional options that can provide different level of accuracy and which are explained in the below section.

Note that compiler has to preserve the input/output (i/o) tensor types requested by Framework, therefore no casting is done on the i/o tensors. Additional speedup can be obtained by casting them in the Framework prior compilation.

To learn how to use compiler command line interface (CLI) options with your application's framework, please see :ref:`torch_neuron_trace_api`, :ref:`tensorflow-ref-neuron-compile-api` and :ref:`tensorflow-ref-neuron-tracing-api`.


Compiler casting options
------------------------

``--fast-math`` option
^^^^^^^^^^^^^^^^^^^^^^^^

The ``--fast-math`` option is intended to replace the ``--fp32-cast`` option. It is recommended to
to start using or migrating to ``--fast-math`` option. The ``--fast-math`` option provides the same level of functionality
as the ``--fp32-cast`` option in addition to the following:

* The ``--fast-math`` option introduces the ``no-fast-relayout`` option to enable lossless transpose operation. This was not possible with the ``--fp32-cast`` option.
* The ``--fast-math`` option provides finer control than the ``--fp32-cast`` option. The transpose operation and the cast operation are controlled independently:

    - ``no-fast-relayout`` and ``fast-relayout`` provide control for the transpose operation.
    - ``fp32-cast-*`` provide control for casting.

See the detailed list of the options in :ref:`/neuron-guide/neuron-cc/command-line-reference.rst`.
