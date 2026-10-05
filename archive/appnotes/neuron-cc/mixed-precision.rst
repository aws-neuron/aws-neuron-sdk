.. _neuron-cc-training-mixed-precision:

Mixed precision and performance-accuracy tuning (``neuron-cc``)
===============================================================

.. contents:: Table of contents
   :local:
   :depth: 2

The Neuron Compiler supports machine learning models with FP32, FP16, and BF16 (Bfloat16) tensors and operators. The Neuron hardware supports a mix of 32-bit and 16-bit datatypes.

Neuron hardware
---------------

The Neuron hardware performs matrix multiplication in FP16 or BF16 on its Matmult Engine, and accumulations in FP32. Operators such as activations and vector operations run in FP16, BF16, and FP32. Neuron transposes tensors in two ways: fast matrix multiplication in FP16/BF16, or slower byte-by-byte data movement.

Performance-accuracy tradeoffs for models trained in FP32
---------------------------------------------------------

You can deploy models trained in FP32 on Neuron by compiling them ahead of time with the :ref:`Neuron Compiler <neuron-cc-index>`.

.. important::
    By default, the Neuron Compiler applies (``--fast-math all``) and casts FP32 weights and operations to BF16, and keeps only partial sums (accumulations) in FP32. This default gives the highest performance for an FP32-trained model, but not the best accuracy. To optimize for accuracy instead, use ``--fast-math none``.

Use the ``--fast-math`` CLI option to choose the tradeoff between performance and accuracy. The right settings depend on your application.

We recommend that you start by compiling for high performance (the default), then test the accuracy of your application. If you need more accuracy, try the next higher-precision casting option until you reach the accuracy and performance you need. A typical flow:

1. Compile without options (default) or with ``--fast-math all``, which optimizes for performance.
2. If accuracy is not sufficient, try ``--fast-math fp32-cast-matmult``.
3. If accuracy is not sufficient, try ``--fast-math fp32-cast-matmult no-fast-relayout``.
4. If accuracy is not sufficient, try ``--fast-math none``, which optimizes for accuracy.

Between steps 2 and 3, and between steps 3 and 4, you have additional options that provide different levels of accuracy. The following section explains them.

The compiler must preserve the input/output (I/O) tensor types requested by the framework, so it does not cast the I/O tensors. You can gain additional speedup by casting them in the framework before compilation.

To learn how to use compiler command line interface (CLI) options with your application's framework, see :ref:`torch_neuron_trace_api`, :ref:`tensorflow-ref-neuron-compile-api`, and :ref:`tensorflow-ref-neuron-tracing-api`.

Compiler casting options
------------------------

``--fast-math`` option
^^^^^^^^^^^^^^^^^^^^^^^^

The ``--fast-math`` option is intended to replace the ``--fp32-cast`` option, and we recommend migrating to ``--fast-math``. It provides the same functionality as ``--fp32-cast``, plus the following:

* ``--fast-math`` adds the ``no-fast-relayout`` option for lossless transpose, which ``--fp32-cast`` does not support.
* ``--fast-math`` gives finer control than ``--fp32-cast``. You control the transpose operation and the cast operation independently:

    - ``no-fast-relayout`` and ``fast-relayout`` control the transpose operation.
    - ``fp32-cast-*`` control casting.

See the full list of options in :doc:`/compiler/neuron-cc/command-line-reference`.
