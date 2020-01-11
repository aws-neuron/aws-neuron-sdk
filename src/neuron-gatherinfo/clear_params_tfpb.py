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

def zero_const(node):
  val = tf.make_ndarray(node.attr.get("value").tensor)
  new_val = val * 0.0
  new_tensor = tensor_util.make_tensor_proto(new_val, new_val.dtype, new_val.shape)
  node.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=new_tensor))

def ZeroAllConst(graphdef):
  sess = tf.compat.v1.Session(graph=tf.import_graph_def(graphdef))
  const_by_name = {}
  node_by_name = {}
  for node in graphdef.node:
    node_by_name[node.name] = node  
    if node.op == "Const":
      const_by_name[node.name] = node  
    if node.op == "BiasAdd" or node.op == "MatMul" \
            or node.op.startswith("Conv") \
            or node.op.startswith("FusedBatchNorm"):
      for i in node.input:  
        i_node = node_by_name[i]
        if i_node.op == "Const":
          zero_const(i_node)
        if i_node.op == "Identity":
          x_node = node_by_name[i_node.input[0]]
          if x_node.op == "Const":
            zero_const(x_node)
  return graphdef

def load_graph(model_file):
  graph_def = tf.compat.v1.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  return graph_def

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Zero-out parameters of BiasAdd, MatMul, Conv*, and FusedBatchNorm of TensorFlow frozen graph.")
  parser.add_argument("--graph", help="File name of frozen graph to be converted",
      required=True)
  parser.add_argument("--out_graph", help="File name to save converted frozen graph",
      required=True)
  args = parser.parse_args()

  graph_orig = load_graph(args.graph)
  graph_mod = ZeroAllConst(graph_orig)
  with tf.io.gfile.GFile(args.out_graph, "wb") as f:
    f.write(graph_mod.SerializeToString())
  #with tf.io.gfile.GFile(args.out_graph + "txt", 'w') as f:
  #  f.write(text_format.MessageToString(graph_mod))
