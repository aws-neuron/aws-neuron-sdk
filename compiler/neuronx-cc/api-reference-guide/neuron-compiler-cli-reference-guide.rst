.. _neuron-compiler-cli-reference-guide:

Neuron Compiler CLI Reference Guide (``neuronx-cc``)
====================================================

This document describes the command line interface of the Neuron Compiler.

This reference is not relevant for applications that run the Neuron Compiler from within a machine learning framework (:ref:`PyTorch-Neuron <pytorch-neuronx-programming-guide>` for example) since these options are passed from the framework directly to the compiler. Using the compiler command line may be desirable for applications that do not use a framework or customize existing frameworks. It is also possible to specify compiler options within the framework which will forward these options to the compiler using :ref:`NEURON_CC_FLAGS <pytorch-neuronx-envvars>`.


Usage
-----

*Optional parameters are shown in square brackets.*

.. _neuron_cli:

.. rubric:: Neuron Compiler Command-Line Interface

.. program:: neuronx-cc

.. option:: neuronx-cc <command> [parameters]

Common parameters for the Neuron CLI:

- :option:`--verbose <level>`: Specify the level of output produced by the compiler. (Default: ``warning``)

  Valid values:

  - ``info``: Informational messages regarding the progress of model compilation (written to stdout).
  - ``warning``: Diagnostic messages that report model code that is not inherently erroneous but may be risky or suggest there may have been an error (written to stderr).
  - ``error``: The compiler detected a condition causing it not complete the compilation successfully (written to stderr).
  - ``critical``: The compiler encountered an unrecoverable error terminates immediately (written to stderr).
  - ``debug``: Extensive information regarding the compiler's internal execution phases (written to stdout).

- :option:`--help`: Display a usage message of compiler options.
    Use :option:`neuronx-cc <command> --help` for information on a specific command.

Available Commands:
~~~~~~~~~~~~~~~~~~~~~~~

-  :option:`compile`
-  :option:`list-operators`

.. _neuronx-cc-compile:

.. option:: neuronx-cc compile [parameters]

  .. _description-1:

  Compile a model for use on the AWS Machine Learning Accelerator.

  .. code-block:: shell

     neuronx-cc compile <model_files>
     --framework <framework_name>
     --target <instance_family>
     [--model-type <model>]
     [--auto-cast <cast_mode>]
     [--auto-cast-type <data_type>]
     [--distribution-strategy <distribution_type>]
     [--O <opt-level>]
     [--enable-saturate-infinity]
     [--enable-fast-context-switch>]
     [--enable-fast-loading-neuron-binaries]
     [--logfile <filename>]
     [--output <filename>]

  *Compile Parameters:*

  - :option:`<model_files>`: Input containing model specification.
      The number of arguments required varies between frameworks:

      - **XLA**: A local filename of a HLO file (hlo.pb) generated via XLA. See `hlo.proto <https://github.com/tensorflow/tensorflow/blob/73c8e20101ae93e9f5ff0b58f68be0b70eca44c5/tensorflow/compiler/xla/service/hlo.proto>`_ for the .proto description and `inspect-compiled-programs <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/g3doc/index.md#user-content-inspect-compiled-programs>`_ for more information on how to generate such files.

  - :option:`--framework <framework_name>`: Framework used to generate training model.

    Valid values:

    - ``XLA``

  - :option:`--target <instance_family>`: Name of the Neuron instance family on which the compiled model will be run.

    Valid values:

    - ``inf2``
    - ``trn1``
    - ``trn1n``

  - :option:`--model-type <model>`: Permit the compiler to attempt model-specific optimizations based upon type of model being compiled. (Default: ``generic``)

    Valid values:

    - ``generic``: Perform optimizations applicable to all types of inference and training models.
    - ``transformer``: Perform optimizations specific to `Transformer <https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>`_ models. 
    - ``unet-inference``: Perform optimizations specific to certain `U-Net <https://en.wikipedia.org/wiki/U-Net>`_ model architectures when performing inference. U-Net models often have certain structures that result in excessive performance-impacting data transfers; this option allows the compiler to apply additional memory optimizations to prevent these data transfers and also allows the compiler to map larger normalization operators which would otherwise not successfully execute.

  - :option:`--auto-cast <cast_mode>`: Controls how the compiler makes tradeoffs between performance and accuracy for FP32 operations. (Default: ``matmult``)

    Valid values:

    - ``matmult``: Only cast FP32 operations that use the Neuron matrix-multiplication engine.
    - ``all``: Cast all FP32 operations to achieve highest performance. This option can potentially lower precision/accuracy.
    - ``none``: Leave all data types as defined in the model. Do not apply auto-casting data type optimizations.

    A more complete discussion on how to use this option and its arguments is in :ref:`Mixed Precision and Performance-accuracy Tuning for Training <neuronx-cc-training-mixed-precision>`.

    .. note:: If the :option:`--auto-cast` option is specified, the :option:`--auto-cast-type` compiler flag can be optionally set to define which lower-precision data type the compiler should use.

  - :option:`--auto-cast-type <data_type>`: When auto-cast mode is enabled, cast the FP32 operators to the lower-precision data type specified by this option. (Default: ``bf16``)

    Valid values:

    - ``bf16``: Cast the FP32 operations selected via the :option:`--auto-cast` option to BF16 to achieve highest performance and preserve dynamic range.
    - ``fp16``: Cast the FP32 operations selected via the :option:`--auto-cast` option to FP16 to achieve improved performance relative to FP32 and increased precision relative to BF16.
    - ``tf32``: Cast the FP32 operations selected via the :option:`--auto-cast` option to TensorFloat-32.
    - ``fp8_e4m3``: Cast the FP32 operations selected via the :option:`--auto-cast` option to a signed 8-bit floating point represented as a 4-bit exponent and 3-bit mantissa. 


    .. note:: If multiple competing options are specified then the option right-most on the command line will supercede previous options.

  - :option:`--distribution-strategy <distribution_type>`: Permit the compiler to attempt model-specific optimizations based upon type of model being compiled. (Default: ``generic``)

    Valid values:

    - ``NEMO``: Enable the compiler to perform optimizations applicable to models that use the `NeMo <https://github.com/NVIDIA/NeMo>`_ APIs to shard parameters, gradients, and optimizer states across data-parallel workers.

  - :option:`--O <opt_level>`: Specify the level of optimization the compiler should perform. Possible numeric values are {1, 2, 3}. (Default: ``2``)

    Valid values:

    - ``1``: enables the core performance optimizations in the compiler, while also minimizing compile time.
    - ``2``: [default] provides the best balance between model performance and compile time.
    - ``3``: may provide additional model execution performance but may incur longer compile times and higher host memory usage during model compilation.

    .. note:: This option supercedes, and deprecates, the ``—enable-experimental-O1`` option introduced in an earlier release.

  - :option:`--enable-saturate-infinity`: Convert +/- infinity values to MAX/MIN_FLOAT for certain computations that have a high risk of generating Not-a-Number (NaN) values. There is a potential performance impact during model execution when this conversion is enabled.

  - :option:`--enable-fast-context-switch`: Optimize for faster model switching rather than execution latency.
      This option will defer loading some weight constants until the start of model execution. This results in overall faster system performance when your application switches between models frequently on the same Neuron Core (or set of cores).

  - :option:`--enable-fast-loading-neuron-binaries`: Save the compilation output file in an uncompressed format.
      This creates executable files which are larger in size but faster for the Neuron Runtime to load into memory during model execution.

  - :option:`--logfile <filename>`: Filename where compiler writes log messages. (Default: “log-neuron-cc.txt”).

  - :option:`--output <filename>`: Filename where compilation output (NEFF archive) will be recorded. (Default: "file.neff”)

  *Example*:
    Compiling an XLA HLO:

    .. code-block:: shell

      neuronx-cc compile bert-model.hlo —-framework XLA -—target trn1 —-model-type transformer —-output bert.neff


.. _neuronx-cc-list-operators:

.. option:: neuronx-cc list-operators [parameters]

  .. _description-1:

  Returns a newline (‘\\n’) separated list of operators supported by the Neuron Compiler.

  .. code-block:: shell

    neuronx-cc list-operators
    --framework <value>

  *List-Operators Parameters:*

  - :option:`--framework <framework_name>`: Framework in which the operators were registered.

    Valid values:

    - ``XLA``: Operator names will be formatted according to the value used by XLA compiler in XlaBuilder.


  *Example*:

  .. code-block:: shell

    neuronx-cc list-operators —framework XLA
    ...


*Exit Statuses*:

- **0**: Compilation succeeded
- **<>0**: An error occurred during compilation.
