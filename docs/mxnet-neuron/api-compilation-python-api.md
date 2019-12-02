# Reference: MXNet-Neuron Compilation Python API

The MXNet-Neuron compilation Python API provides a method to compile model graph for execution on Inferentia.

## Method

`mx.contrib.neuron.compile(sym, args, aux, inputs, **compile_args)`

## Description

Within the graph or subgraph, the compile method selects and sends Neuron-supported operations to Neuron-Compiler for compilation and saves the compiled artifacts in the graph.  

The “`--num-neuroncores`” option directs compiler to limit compiled graph to run on a specified number of NeuronCores. This number can be less than the total available NeuronCores on an Inf1 instance.

Please note that compiling for more than the number of available NeuronCores will work during compilation but result in resource error during inference operation.

The compiled graph can be saved using the MXNet save_checkpoint and served using MXNet Model Serving.

## Arguments

* sym - Symbol object loaded from symbol.json file
* args - args/params dictionary loaded from params file
* aux - aux/params dictionary loaded from params file
* inputs - a dictionary with key/value mappings for input name to input numpy arrays
* compile_args (optional) - a dictionary with key/value mappings for inferentia-specific compile options:
    * "--num-neuroncores" : integer value that specifies the number NeuronCores to hold a subgraph

## Returns

* sym  - new partitioned symbol
* args - modified args/params
* auxs - modified aux/params

## Example Usage

The following is an example usage of the compilation, with default compilation arguments:

```python
sym, args, aux = mx.contrib.neuron.compile(sym, args, aux, inputs={'data' : img})
```
To extract operation counts, insert the following code after compile step (assume csym is the compiled MXNet symbol):

```python
import json
def sym_nodes(sym):
  return json.loads(sym.tojson())['nodes']
def count_ops(graph_nodes):
  return len([x['op'] for x in graph_nodes if x['op'] != 'null'])
def get_compile_stats(sym):
  cnt = count_ops(sym_nodes(sym))
  neuron_subgraph_cnt = 0
  neuron_compiled_cnt = 0
  for g in sym_nodes(sym):
    if g['op'] == '_neuron_subgraph_op':
      neuron_subgraph_cnt += 1
      for sg in g['subgraphs']:
        neuron_compiled_cnt += count_ops(sg['nodes'])
  return (cnt, neuron_subgraph_cnt, neuron_compiled_cnt)

original_cnt = count_ops(sym_nodes(sym))
post_compile_cnt, neuron_subgraph_cnt, neuron_compiled_cnt = get_compile_stats(csym)
print("INFO:mxnet: Number of operations in original model: ", original_cnt)
print("INFO:mxnet: Number of operations in compiled model: ", post_compile_cnt)
print("INFO:mxnet: Number of Neuron subgraphs in compiled model: ", neuron_subgraph_cnt)
print("INFO:mxnet: Number of operations placed on Neuron runtime: ", neuron_compiled_cnt)
```

```bash
INFO:mxnet: Number of operations in original model:  67
INFO:mxnet: Number of operations in compiled model:  4
INFO:mxnet: Number of Neuron subgraphs in compiled model:  2
INFO:mxnet: Number of operations placed on Neuron runtime:  65
```
