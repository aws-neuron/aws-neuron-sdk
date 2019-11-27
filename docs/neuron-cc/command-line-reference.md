# Reference Guide: Neuron compiler CLI 

This document describes the command line interface of the Neuron compiler. This reference is not relevant for applications that run neuron-cc from within a machine learning framework (Tensorflow-neuron for example) since these options are passed from the framework to neuron-cc. 

Using neuron-cc on the command line may be desirable for applications that do not use a framework, or customize existing frameworks. It is also possible to supply CLI commands to the framework as options to be passed through to the compiler. 

The synopsis for each command shows its parameters and their usage. Optional parameters are shown in square brackets. See the individual framework guides for the correct syntax. 


## Synopsis

```
neuron-cc [options] <command> [parameters] 
```

```
neuron-cc <command> --help for information on a specific command. 
```


## Common Options

```
--log-level (string) (default “INFO”) 
``` 

* DEBUG
* INFO
* WARN
* ERROR


## Available Commands

* compile
* list-operators


# neuron-cc compile

```
neuron-cc compile <file names> --framework <value> --io-config <value> [--num-neuroncores <value>] [--output <value>]
```

## Description

Compile a model for use on the AWS Inferentia Machine Learning Accelerator.

## Examples

```
neuron-cc compile test_graph_tfmatmul.pb --framework TENSORFLOW --io-config test_graph_tfmatmul.config
```

```
neuron-cc compile lenet-symbol.json lenet-0001.params --framework MXNET --num-neuroncores 2 --output out.infa —debug
```

## Options

**```<file names> ```**
Input containing model specification. The number of arguments required varies between frameworks:

* **TENSORFLOW** A local filename or URI of a Tensorflow Frozen GraphDef (.pb); or the name of a local directory containing a Tensorflow SavedModel.
    
    See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto for the associated .proto schema for Tensorflow Frozen GraphDefs. See https://www.tensorflow.org/guide/saved_model for more information on the SavedModel format.
* **MXNET** - List of local filenames or URIs where input architecture .json file and parameter .param file are stored. These contains information related to the architecture of your graph and associated parameters, respectively.
* **ONNX** - A local filename or URI for a ONNX model.

**--framework** (string)
Framework in which the model was trained. 

Valid values: TENSORFLOW | MXNET | ONNX

**--num-neuroncores** (int) (default 1)
Compile for the given number of neuron cores so as to leverage Neuron Core Pipeline mode.

**--output** (string) (default “out.neff”)
Filename where compilation output (NEFF archive) will be recorded.

**--io-config**(string) 
Configuration containing the names and shape of input and output tensors.

The io-config can be specified as a local filename, a URI, or a string containing the io-config itself.

The io-config must be formatted as a JSON object with two members “inputs” and “outputs”. “inputs” is an object mapping input tensor names to an array of shape and data type. “outputs” is an array of output tensor names. Consider the following example:


```
{
  "inputs": {
     "input0:0": [[1,100,100,3], "float16"],
     "input1:0": [[1,100,100,3], "float16"]
  },
  "outputs": ["output:0"]
}
```

## STDOUT

Logs at levels “trace”, “debug”, and “info” will be written to STDOUT.

## STDERR

Logs at levels “warn”, “error”, and “fatal” will be written to STDERR.

## EXIT STATUS

**0** - Compilation succeeded

**>0** - An error occurred during compilation.

# neuron-cc list-operators

neuron-cc list-operators --framework <value>

## Description

Returns a newline ('\n') separated list of operators supported by the AWS Inferentia Neural Network Accelerator.


* ** TENSORFLOW ** - Operators will be formatted according to the value passed to the associated REGISTER_OP(“OperatorName”) macro. 
    
    See https://www.tensorflow.org/guide/extend/op#define_the_ops_interface for more information regarding operator registration in TensorFlow.
    
* ** MXNET ** - Operator names will be formatted according to the value passed to the associated NNVM_REGISTER_OP(operator_name) macro. 
    
    See http://mxnet.incubator.apache.org/versions/master/faq/new_op.html for more details regarding operator registration in MxNet.
* ** ONNX ** - 

## Example

```
neuron-cc list-operators —framework TENSORFLOW
AddN
AdjustContrastv2
CheckNumbers
::::::
```

## Options

**--framework** (string)
Framework in which the operators were registered.  

Valid values: TENSORFLOW | MXNET | ONNX

## STDOUT

Returns a newline ('\n') separated list of operators supported by the AWS Inferentia Neural Network Accelerator.

## EXIT STATUS

**0** - Call succeeded

**> 0** - An error occurred


