.. _ref-mxnet-neuron-compilation-python-api:

Reference: MXNet-Neuron Compilation Python API
==============================================

The MXNet-Neuron compilation Python API provides a method to compile
model graph for execution on Inferentia.

Method
------

``import mxnet as mx``
``mx.contrib.neuron.compile(sym, args, aux, inputs, **compile_args)``

Description
-----------

Within the graph or subgraph, the compile method selects and sends
Neuron-supported operations to Neuron-Compiler for compilation and saves
the compiled artifacts in the graph. Uncompilable operations are kept as
original operations for framework execution.

The compiled graph can be saved using the MXNet save_checkpoint and
served using MXNet Model Serving. Please see
:ref:`mxnet-neuron-model-serving` for more information about exporting
to saved model and serving using MXNet Model Serving.

Options can be passed to Neuron compiler via the compile function. For
example, the “\ ``--neuroncore-pipeline-cores``\ ” option directs Neuron compiler
to compile each subgraph to fit in the specified number of NeuronCores.
This number can be less than the total available NeuronCores on an Inf1
instance. See :ref:`neuron-compiler-cli-reference` for more information
about compiler options.

For debugging compilation, use SUBGRAPH_INFO=1 environment setting before
calling the compilation script. The extract subgraphs are preserved as hidden
files in the run directory. For more information, see :ref:`neuron_gatherinfo`

Arguments
---------

-  **sym** - Symbol object loaded from symbol.json file
-  **args** - args/params dictionary loaded from params file
-  **aux** - aux/params dictionary loaded from params file
-  **inputs** - a dictionary with key/value mappings for input name to
   input numpy arrays
-  **kwargs** (optional) - a dictionary with key/value mappings for
   MXNet-Neuron compilation and Neuron Compiler options.

   -  For example, to limit the number of NeuronCores per subgraph, use
      ``compile_args={'--neuroncore-pipeline-cores' : N}`` where N is an integer
      representing the maximum number of NeuronCores per subgraph.
   -  Additional compiler flags can be passed using
      ``'flags' : [<flags>]`` where is a comma separated list of
      strings. See :ref:`neuron_gatherinfo` for example of passing debug
      flags to compiler.
   -  Advanced option to exclude node names:
      ``compile_args={'excl_node_names' : [<node names>]}`` where is a
      comma separated list of node name strings.

Returns
-------

-  **sym** - new partitioned symbol
-  **args** - modified args/params
-  **auxs** - modified aux/params

Example Usage: Compilation
--------------------------

The following is an example usage of the compilation, with default
compilation arguments:

.. code:: python

   import mxnet as mx
   ...
   sym, args, aux = mx.contrib.neuron.compile(sym, args, aux, inputs={'data' : img})

Example Usage: Extract Compilation Statistics
---------------------------------------------

To extract operation counts, insert the following code after compile
step (assume csym is the compiled MXNet symbol):

.. code:: python

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

.. code:: bash

   INFO:mxnet: Number of operations in original model:  67
   INFO:mxnet: Number of operations in compiled model:  4
   INFO:mxnet: Number of Neuron subgraphs in compiled model:  2
   INFO:mxnet: Number of operations placed on Neuron runtime:  65
