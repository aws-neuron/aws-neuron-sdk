# MXNet-Neuron Compilation Python API

The MXNet-Neuron Compilation Python API provides user a method to compile model graph for execution on Inferentia. It is available as a method in a Neuron module in MXNet's contribution space.

## Method

`mx.contrib.neuron.compile(sym, args, aux, inputs, **compile_args)`

## Description

Within the graph or subgraph, the compile method selects and send Neuron-supported operations to Neuron-Compiler for compilation and saves the compiled artifacts in the graph.  More on Neuron-Compiler can be found here: [link].

The “`num-neuroncores`” option directs compiler to limit compiled graph to run on a specified number of NeuronCores. This number can be less than the total available NeuronCores on an N1 instance. See performance tuning application note [link] for more information.

Please note that compiling for more than the number of available NeuronCores will work during compilation but result in resource error during inference.

The compiled graph can be saved using the MXNet save_checkpoint and served using MXNet Model Serving.

## Arguments

* sym - Symbol object loaded from symbol.json file
* args - args/params dictionary loaded from params file
* aux - aux/params dictionary loaded from params file
* inputs - a dictionary with key/value mappings for input name to input numpy arrays
* compile_args (optional) - a dictionary with key/value mappings for inferentia-specific compile options:
    * "num-neuroncores" : integer value that specifies the number NeuronCores to hold a subgraph

## Returns

* sym  - new partitioned symbol
* args - modified args/params
* auxs - modified aux/params

## Example Usage

The following is an example usage of the compilation, with default compilation arguments:

```python
sym, args, aux = mx.contrib.neuron.compile(sym, args, aux, inputs={'data' : img})
```
