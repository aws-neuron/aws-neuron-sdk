# Reference: MXNet-Neuron Compilation Python API

The MXNet-Neuron compilation Python API provides a method to compile model graph for execution on Inferentia.

## Method

`import mxnet as mx`
`mx.contrib.neuron.compile(sym, args, aux, inputs, **compile_args)`

## Description

Within the graph or subgraph, the compile method selects and sends Neuron-supported operations to Neuron-Compiler for compilation and saves the compiled artifacts in the graph. Uncompilable operations are kept as original operations for framework execution.

The compiled graph can be saved using the MXNet save_checkpoint and served using MXNet Model Serving. Please see [MXNet-Neuron Model Serving](./tutorial-model-serving.md) for more information about exporting to saved model and serving using MXNet Model Serving.

Options can be passed to Neuron compiler via the compile function. For example, the “`--num-neuroncores`” option directs Neuron compiler to compile each subgraph to fit in the specified number of NeuronCores. This number can be less than the total available NeuronCores on an Inf1 instance. See [Neuron Compiler CLI](../neuron-cc/command-line-reference.md) for more information about compiler options.

## Arguments

* **sym** - Symbol object loaded from symbol.json file
* **args** - args/params dictionary loaded from params file
* **aux** - aux/params dictionary loaded from params file
* **inputs** - a dictionary with key/value mappings for input name to input numpy arrays
* **kwargs** (optional) - a dictionary with key/value mappings for Neuron compiler options. For example, use `compile_args={'--num-neuroncores' : 4}` to set number of NeuronCores per subgraph to 4.

## Returns

* **sym**  - new partitioned symbol
* **args** - modified args/params
* **auxs** - modified aux/params

## Example Usage: Compilation

The following is an example usage of the compilation, with default compilation arguments:

```python
import mxnet as mx
...
sym, args, aux = mx.contrib.neuron.compile(sym, args, aux, inputs={'data' : img})
```

## Example Usage: Extract Compilation Statistics

To extract operation counts, insert the following code after compile step (assume csym is the compiled MXNet symbol):

```python
import json

# Return list of nodes from MXNet symbol
def sym_nodes(sym):
  return json.loads(sym.tojson())['nodes']

# Return number of operations in node list  
def count_ops(graph_nodes):
  return len([x['op'] for x in graph_nodes if x['op'] != 'null'])

# Return triplet of compile statistics
# - count of operations in symbol database
# - number of Neuron subgraphs
# - number of operations compiled to Neuron runtime  
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
