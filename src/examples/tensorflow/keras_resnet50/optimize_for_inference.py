""" Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: MIT-0
"""

import re
import copy
import argparse
import tensorflow as tf
import numpy as np
import string

from google.protobuf import text_format
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.tools.graph_transforms import TransformGraph

def clear_input(node):
  for i in range(len(node.input)):
    node.input.pop()

def replace_name(node, name):
  node.name = name
     
def replace_input(node, input_name, new_name):
  # node.input.replace(input_name, new_name)
  temp = []
  for i in node.input:
    temp.extend([new_name if i == input_name else i])
  clear_input(node)
  for i in temp:
    node.input.extend([i])

def swap_names(node1, node2):
  temp = node2.name
  node2.name = node1.name
  node1.name = temp

def get_const_node(const_node_name, const_by_name):
  name = re.sub("/read$", "", const_node_name)
  return const_by_name[name]

def get_const_ndarray(const_node_name, const_by_name):
  name = re.sub("/read$", "", const_node_name)
  node = const_by_name[name]
  return tf.make_ndarray(node.attr.get("value").tensor)

def adjust_bias_values(bias_node, fbn_node, const_by_name):
  bias_val = get_const_ndarray(bias_node.input[1], const_by_name)  
  gamma_val = get_const_ndarray(fbn_node.input[1], const_by_name)  
  mean_val = get_const_ndarray(fbn_node.input[3], const_by_name)  
  variance_val = get_const_ndarray(fbn_node.input[4], const_by_name) 
  new_bias = bias_val * gamma_val / np.sqrt(variance_val)
  new_tensor = tensor_util.make_tensor_proto(new_bias, new_bias.dtype, new_bias.shape)
  bias_const_node = get_const_node(bias_node.input[1], const_by_name)
  bias_const_node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=new_tensor))

def MoveBiasAddAfterFusedBatchNorm(graphdef):
  """fold_batch_norm function of TransformGraph is unable to fold Keras ResNet50
  because of BiasAdd between Conv2D and FusedBatchNorm (BiasAdd is not needed
  if FusedBatchNorm is used, but it exists in Keras ResNet50). Here, we 
  move BiasAdd to after FusedBatchNorm, and adjust bias value by gamma/sqrt(variance).
  """
  sess = tf.compat.v1.Session(graph=tf.import_graph_def(graphdef))
  output_graph_def = tf.compat.v1.GraphDef()
  node_by_name = {}
  const_by_name = {}
  for node in graphdef.node:
    # Hack: use FusedBatchNormV2 so fold_batch_norm can recognize
    if node.op == "FusedBatchNormV3":
      node.op = "FusedBatchNorm"
      del(node.attr["U"])
      #import pdb; pdb.set_trace()
    copied_node = node_def_pb2.NodeDef()
    copied_node.CopyFrom(node)
    node_by_name[node.name] = copied_node
    skip_add_node = False
    # Switch Mul/BiasAdd in Keras RN50 so fold_batch_norm transform would work
    if node.op == "Const":
      const_by_name[node.name] = copied_node  
    elif node.op.startswith("FusedBatchNorm"):
      inputs = node.input
      for i in inputs:
        input_node = node_by_name[i]
        if input_node.op == "BiasAdd":
          output_graph_def.node.remove(input_node)
          input_node_input0 = input_node.input[0]
          # Adjust bias values (multiply by scale/sqrt(variance))
          adjust_bias_values(input_node, node, const_by_name)
          # Hack: swap names to avoid changing input of activation
          swap_names(copied_node, input_node)
          # Fix inputs for these two ops
          replace_input(copied_node, i, input_node_input0)
          replace_input(input_node, input_node_input0, copied_node.name)
          # Fix order in node list
          output_graph_def.node.extend([copied_node])
          output_graph_def.node.extend([input_node])
          skip_add_node = True
    # Add maybe-modified nodes if not already done
    if not skip_add_node:
      output_graph_def.node.extend([copied_node])
  return output_graph_def

def FoldFusedBatchNorm(graph_def):
  """Optimize training graph for inference:
    - Remove Identity and CheckNumerics nodes
    - Fold FusedBatchNorm constants into previous Conv2D weights
    - Fold other constants
    - Strip unused nodes
    - Sort by execution order
  """
  transformed_graph_def = TransformGraph (
         graph_def,
         ['input_1'],
         ['probs/Softmax'],
         [
            'add_default_attributes',
            'remove_nodes(op=Identity, op=CheckNumerics)',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'fold_old_batch_norms',
            'strip_unused_nodes',
            'sort_by_execution_order',
         ])
  return transformed_graph_def

def load_graph(model_file):
  graph_def = tf.compat.v1.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  return graph_def

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", help="graph/model to be executed",
      required=True)
  parser.add_argument("--out_graph", help="graph/model to be generated",
      required=True)
  args = parser.parse_args()

  graph_orig = load_graph(args.graph)
  graph_mod = MoveBiasAddAfterFusedBatchNorm(graph_orig)
  graph_mod2 = FoldFusedBatchNorm(graph_mod)
  with tf.io.gfile.GFile(args.out_graph, "wb") as f:
    f.write(graph_mod2.SerializeToString())
  #with tf.io.gfile.GFile(args.out_graph + "txt", 'w') as f:
  #  f.write(text_format.MessageToString(graph_mod2))
