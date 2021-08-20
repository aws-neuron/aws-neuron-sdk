.. _neuron-compiler-cli-reference:

Neuron compiler CLI Reference Guide
===================================

This document describes the command line interface of the Neuron
compiler. This reference is not relevant for applications that run
neuron-cc from within a machine learning framework (TensorFlow-Neuron
for example) since these options are passed from the framework directly
to neuron-cc.

Using neuron-cc on the command line may be desirable for applications
that do not use a framework, or customize existing frameworks. It is
also possible to supply CLI commands to the framework as options to be
passed through to the compiler.

The synopsis for each command shows its parameters and their usage.
Optional parameters are shown in square brackets. See the individual
framework guides for the correct syntax.

Synopsis
--------

::

   neuron-cc [options] <command> [parameters]

::

   neuron-cc <command> --help for information on a specific command.

Common Options
--------------

- ``--verbose`` (string) (default=“WARN”):

  Valid values:

    -  ``DEBUG``
    -  ``INFO``
    -  ``WARN``
    -  ``ERROR``

Available Commands
------------------

-  ``compile``
-  ``list-operators``

neuron-cc compile
-----------------

::

   neuron-cc compile <file names> --framework <value> --io-config <value> [--neuroncore-pipeline-cores <value>] [--enable-fast-loading-neuron-binaries] [--enable-fast-context-switch] [--fp32-cast cast-method] [--output <value>] 

Description
~~~~~~~~~~~

Compile a model for use on the AWS Inferentia Machine Learning
Accelerator.

Examples
~~~~~~~~

::

   neuron-cc compile test_graph_tfmatmul.pb --framework TENSORFLOW --io-config test_graph_tfmatmul.config

::

   neuron-cc compile lenet-symbol.json lenet-0001.params --framework MXNET --neuroncore-pipeline-cores 2 --output file.neff


   neuron-cc compile bert-model.hlo --framework XLA  --output file.neff

Options
~~~~~~~

- ``<file names>``: Input containing model specification. The number
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


- ``--framework`` (string): Framework in which the model was trained.

  Valid values:

    - ``TENSORFLOW``
    - ``MXNET``
    - ``XLA``

- ``--neuroncore-pipeline-cores`` (int) (default=1): Number of neuron cores
  to be used in "NeuronCore Pipeline" mode. This is different from data
  parallel deployment (same model on multiple neuron cores). Refer to
  Runtime/Framework documentation for data parallel deployment options.

  Compile for the given number of
  neuron cores so as to leverage NeuronCore Pipeline mode.

  .. note::
    This is not used to define the number of Neuron Cores to be used in a data
    parallel deployment (ie the same model on multiple Neuron Cores). That
    is a runtime/framework configuration choice.

- ``--output`` (string) (default=“out.neff”): Filename where compilation
  output (NEFF archive) will be recorded.

- ``--io-config`` (string): Configuration containing the names and shapes
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

- ``--enable-fast-loading-neuron-binaries`` : Write the compilation
  output (NEFF archive) in uncompressed format which results
  in faster loading of the archive during inference.

- ``--enable-fast-context-switch`` : Optimize for faster model switching 
  rather than inference latency. This results in overall faster system
  performance when your application switches between models frequently
  on the same neuron core (or set of cores). The optimization 
  triggered by this option for example defers loading some weight
  constants until the start of inference.

- ``--fp32-cast`` : Refine the automatic casting of fp32 tensors.
  See detailed description and trade offs in :ref:`/neuron-guide/perf/performance-tuning.rst`.

STDOUT
~~~~~~

Logs at levels “trace”, “debug”, and “info” will be written to STDOUT.

STDERR
~~~~~~

Logs at levels “warn”, “error”, and “fatal” will be written to STDERR.

EXIT STATUS
~~~~~~~~~~~

**0** - Compilation succeeded

**>0** - An error occurred during compilation.

.. _neuron-cc-list-operators:

neuron-cc list-operators
------------------------

::

   neuron-cc list-operators --framework <value>

.. _description-1:

Description
~~~~~~~~~~~

Returns a newline ('n') separated list of operators supported by the
NeuronCore.

-  **TENSORFLOW**: Operators will be formatted according to the value
   passed to the associated REGISTER_OP(“OperatorName”) macro.

   See https://www.tensorflow.org/guide/create_op#define_the_op_interface
   for more information regarding operator registration in TensorFlow.

-  **MXNET**: Operator names will be formatted according to the value
   passed to the associated NNVM_REGISTER_OP(operator_name) macro.

   See https://mxnet.apache.org/api/faq/new_op for more details
   regarding operator registration in MxNet.


Example
~~~~~~~

::

   neuron-cc list-operators --framework TENSORFLOW
   AddN
   AdjustContrastv2
   CheckNumbers
   ...

.. _options-1:

Options
~~~~~~~

- ``--framework`` (string): Framework in which the operators were
  registered.

  Valid values:

    - ``TENSORFLOW``
    - ``MXNET``

.. _stdout-1:

STDOUT
~~~~~~

Returns a newline (``'\n'``) separated list of operators supported by the
NeuronCore.

.. _exit-status-1:

EXIT STATUS
~~~~~~~~~~~

**0** - Call succeeded

**> 0** - An error occurred
