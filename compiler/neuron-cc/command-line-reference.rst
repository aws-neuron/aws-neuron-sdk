.. _neuron-compiler-cli-reference:

Neuron compiler CLI Reference Guide (``neuron-cc``)
===================================================

This document describes the command line interface of the Neuron
compiler. This reference is not relevant for applications that run
neuron-cc from within a machine learning framework (TensorFlow-Neuron
for example) since these options are passed from the framework directly
to neuron-cc.

Using neuron-cc on the command line may be desirable for applications
that do not use a framework, or customize existing frameworks. It is
also possible to supply CLI commands to the framework as options to be
passed through to the compiler.

Usage
--------

Optional parameters are shown in square brackets. See the individual
framework guides for the correct syntax.

.. _neuron_cli:

.. rubric:: Neuron Compiler CLI

.. program:: neuron-cc

.. option:: neuron-cc [options] <command> [parameters]

Common options for the Neuron CLI:

    - :option:`--verbose` (string) default=“WARN”:

        Valid values:

        -  :option:`DEBUG`
        -  :option:`INFO`
        -  :option:`WARN`
        -  :option:`ERROR`



Use :option:`neuron-cc <command> --help` for information on a specific command.

Available Commands:
~~~~~~~~~~~~~~~~~~~

-  :option:`compile`
-  :option:`list-operators`


.. option:: neuron-cc compile [parameters]

    Compile a model for use on the AWS Inferentia Machine Learning Accelerator.

    .. code-block::

        neuron-cc compile <file names> --framework <value> --io-config <value> [--neuroncore-pipeline-cores <value>] [--enable-saturate-infinity] [--enable-fast-loading-neuron-binaries] [--enable-fast-context-switch] [--fp32-cast cast-method] [--fast-math cast-method] [--output <value>]

    **Compile Parameters:**

    - :option:`<file names>`: Input containing model specification. The number
      of arguments required varies between frameworks:

        -  **TENSORFLOW**: A local filename or URI of a TensorFlow Frozen
           GraphDef (.pb); or the name of a local directory containing a
           TensorFlow SavedModel.

           See
           https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto
           for the associated .proto schema for TensorFlow Frozen GraphDefs. See
           https://www.tensorflow.org/guide/saved_model for more information on
           the SavedModel format.

        -  **MXNET**: List of local filenames or URIs where input architecture
           .json file and parameter .param file are stored. These contains
           information related to the architecture of your graph and associated
           parameters, respectively.


    - :option:`--framework` (string): Framework in which the model was trained.

      Valid values:

        - :option:`TENSORFLOW`
        - :option:`MXNET`
        - :option:`XLA`

    - :option:`--neuroncore-pipeline-cores` (int) (default=1): Number of neuron cores
      to be used in "NeuronCore Pipeline" mode. This is different from data
      parallel deployment (same model on multiple neuron cores). Refer to
      Runtime/Framework documentation for data parallel deployment options.

      Compile for the given number of
      neuron cores so as to leverage NeuronCore Pipeline mode.

      .. note::
        This is not used to define the number of Neuron Cores to be used in a data
        parallel deployment (ie the same model on multiple Neuron Cores). That
        is a runtime/framework configuration choice.

    - :option:`--output` (string) (default=“out.neff”): Filename where compilation
      output (NEFF archive) will be recorded.

    - :option:`--io-config` (string): Configuration containing the names and shapes
      of input and output tensors.

      The io-config can be specified as a local filename, a URI, or a string
      containing the io-config itself.

      The io-config must be formatted as a JSON object with two members
      “inputs” and “outputs”. “inputs” is an object mapping input tensor names
      to an array of shape and data type. “outputs” is an array of output
      tensor names. Consider the following example:

      .. code-block:: json

        {
         "inputs": {
            "input0:0": [[1,100,100,3], "float16"],
            "input1:0": [[1,100,100,3], "float16"]
         },
         "outputs": ["output:0"]
        }

    - :option:`--enable-saturate-infinity` : Convert +/- infinity values to MAX/MIN_FLOAT for certain computations that have a high risk of generating Not-a-Number (NaN) values. There is a potential performance impact during model execution when this conversion is enabled.


    - :option:`--enable-fast-loading-neuron-binaries` : Write the compilation
      output (NEFF archive) in uncompressed format which results
      in faster loading of the archive during inference.

    - :option:`--enable-fast-context-switch` : Optimize for faster model switching
      rather than inference latency. This results in overall faster system
      performance when your application switches between models frequently
      on the same neuron core (or set of cores). The optimization
      triggered by this option for example defers loading some weight
      constants until the start of inference.

    - :option:`--fast-math` : Controls tradeoff between performance and accuracy for fp32 operators. See more suggestions on how to use this option with the below arguments in :ref:`neuron-cc-training-mixed-precision`.


        - ``all`` (Default): enables all optimizations that improve performance. This option can potentially lower precision/accuracy.

        - ``none`` : Disables all optimizations that improve performance. This option will provide best precision/accuracy.

        - Tensor transpose options

            - ``fast-relayout``: Only enables fast relayout optimization to improve performance by using the matrix multiplier for tensor transpose. The data type used for the transpose is either FP16 or BF16, which is controlled by the ``fp32-cast-xxx`` keyword.

            - ``no-fast-relayout``: Disables fast relayout optimization which ensures that tensor transpose is bit-accurate (lossless) but slightly slower.


        - Casting options

            - ``fp32-cast-all`` (Default): Cast all FP32 operators to BF16 to achieve highest performance and preserve dynamic range. Same as setting ``--fp32-cast all``.

            - ``fp32-cast-all-fp16``: Cast all FP32 operators to FP16 to achieve speed up and increase precision versus BF16. Same setting as ``--fp32-cast all-fp16``.

            - ``fp32-cast-matmult``: Only cast FP32 operators that use Neuron Matmult engine to BF16 while using FP16 for matmult-based transpose to get better accuracy. Same as setting ``--fp32-cast matmult``.

            - ``fp32-cast-matmult-bf16``: Cast only FP32 operators that use Neuron Matmult engine (including matmult-based transpose) to BF16 to preserve dynamic range. Same as setting ``--fp32-cast matmult-bf16``.

            - ``fp32-cast-matmult-fp16``: Cast only FP32 operators that use Neuron Matmult engine (including matmult-based transpose) to fp16 to better preserve precision. Same as setting ``--fp32-cast matmult-fp16``.



        .. important ::

            * ``all`` and ``none`` are mutually exclusive

            * ``all`` is equivalent to using ``fp32-cast-all fast-relayout`` (best performance)

            * ``none`` is equivalent to using ``fp32-cast-matmult-bf16 no-fast-relayout`` (best accuracy)

            * ``fp32-cast-*`` options are mutually exclusive

            * ``fast-relayout`` and ``no-fast-relayout`` are mutually exclusive

            * The ``fp32-cast-*`` and ``*-fast-relayout`` options will overwrite the default behavior in ``all`` and ``none``.

            * For backward compatibility, the ``--fp32-cast`` option has higher priority over ``--fast-math``. It will overwrite the FP32 casting options in any of the ``--fast-math`` options if ``--fp32-cast`` option is present explicitly.


    - :option:`--fp32-cast` : Refine the automatic casting of fp32 tensors. This is being replaced by a newer --fast-math.

        .. important ::

            * ``--fp32-cast`` option is being deprecated and ``--fast-math`` will replace it in future releases.

            * ``--fast-math`` is introducing the ``no-fast-relayout`` option to enable lossless transpose operation.


        The ``--fp32-cast`` is an interface for controlling the performance and accuracy tradeoffs. Many of the ``--fast-math`` values invoke (override) it.

        - ``all`` (default): Cast all FP32 operators to BF16 to achieve speed up and preserve dynamic range.

        - ``matmult``: Cast only FP32 operators that use Neuron Matmult engine to BF16 while using fp16 for matmult-based transpose to get better accuracy.

        - ``matmult-fp16``: Cast only FP32 operators that use Neuron Matmult engine (including matmult-based transpose) to fp16 to better preserve precision.

        - ``matmult-bf16``: Cast only FP32 operators that use Neuron Matmult engine (including matmult-based transpose) to BF16 to preserve dynamic range.

        - ``all-fp16``: Cast all FP32 operators to FP16 to achieve speed up and better preserve precision.




    **Log Levels:**

        Logs at levels “trace”, “debug”, and “info” will be written to STDOUT.

        Logs at levels “warn”, “error”, and “fatal” will be written to STDERR.

    **Exit Status**

        **0** - Compilation succeeded

        **>0** - An error occurred during compilation.

    **Examples**


        Compiling a saved TensorFlow model:

        .. code-block:: shell

           neuron-cc compile test_graph_tfmatmul.pb --framework TENSORFLOW --io-config test_graph_tfmatmul.config

        Compiling a MXNet model:

        .. code-block:: shell

           neuron-cc compile lenet-symbol.json lenet-0001.params --framework MXNET --neuroncore-pipeline-cores 2 --output file.neff

        Compiling an XLA HLO:

        .. code-block:: shell

           neuron-cc compile bert-model.hlo --framework XLA  --output file.neff

.. _neuron-cc-list-operators:

.. option:: neuron-cc list-operators [parameters]

    .. _description-1:

        Returns a newline ('n') separated list of operators supported by the NeuronCore.

        -  **TENSORFLOW**: Operators will be formatted according to the value
           passed to the associated REGISTER_OP(“OperatorName”) macro.

           See https://www.tensorflow.org/guide/create_op#define_the_op_interface
           for more information regarding operator registration in TensorFlow.

        -  **MXNET**: Operator names will be formatted according to the value
           passed to the associated NNVM_REGISTER_OP(operator_name) macro.

        -  **XLA**: Operator names will be formatted according to the value used by XLA compiler in XlaBuilder.

           See https://www.tensorflow.org/xla/operation_semantics for more information regarding XLA operator semantics in XLA interface.

    .. code-block:: shell

        neuron-cc list-operators --framework <value>

    .. _options-1:

    - :option:`--framework` (string): Framework in which the operators were
      registered.

      Valid values:

        - :option:`TENSORFLOW`
        - :option:`MXNET`
        - :option:`XLA`

    **Exit Status**

    **0** - Call succeeded

    **>0** - An error occurred

    **Example**

    .. code-block:: shell

       $ neuron-cc list-operators --framework TENSORFLOW
       AddN
       AdjustContrastv2
       CheckNumbers
       ...
