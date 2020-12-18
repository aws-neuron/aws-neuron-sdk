import os
import json
import sys
import struct
import argparse
import subprocess
from collections import Counter
 
class neuron_parser:
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.parser.add_argument('model_path', type=str, help='a TensorFlow SavedModel directory (currently supporting TensorFlow v1 SaveModel only).')
    self.parser.add_argument('--show_names', action='store_true', help='list operation by name instead of summarizing by type (caution: this option will generate many lines of output for a large model).')
    self.parser.add_argument('--expand_subgraph', action='store_true', help='show subgraph operations.')
    self.parser_args = self.parser.parse_args()
    self.neuronop_info = {}
    self.total_pipeline_cores = 0
    self.min_required_pipeline_cores = 0
    path = self.parser_args.model_path
    if os.path.exists(path + '-symbol.json'):
      self.load_mxnet_model(path)
    elif os.path.isdir(path):
      self.load_tensorflow_model(path)
    else:
      raise RuntimeError('Cannot determine framework type from model path argument.')
    self.supported = self.get_neuron_supported()
    self.supported.extend(self.addl_support)
    for name, executable, (sg_nodetypes, sg_nodenames) in self.neuron_nodes:
      num_cores, requested_cores, _ = self.get_cores_from_executable(executable)
      self.neuronop_info[name] = (num_cores, requested_cores, sg_nodetypes, sg_nodenames)
      self.total_pipeline_cores += num_cores
      if num_cores > self.min_required_pipeline_cores:
          self.min_required_pipeline_cores = num_cores

  def get_neuron_supported(self):
    exec_cmd = ["neuron-cc", "list-operators", "--framework", self.framework]
    oplist = subprocess.check_output(' '.join(exec_cmd), shell=True)
    oplist = str(oplist, 'utf-8')
    oplist = oplist.split("\n")
    return oplist[:-1]  # Remove the last element which is ''
 
  def get_tf_subgraph_types_names(self, node):
    from tensorflow.core.framework import graph_pb2
    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(node.attr['graph_def'].s)
    sg_nodes = graph_def.node
    sg_nodes = [sg_node for sg_node in sg_nodes if sg_node.op not in self.excl_types]
    nodetypes = [sg_node.op for sg_node in sg_nodes]
    nodenames = [sg_node.name for sg_node in sg_nodes]
    return nodetypes, nodenames

  def load_tensorflow_model(self, path):
    import tensorflow as tf
    import tensorflow_hub as hub
    self.framework = 'TENSORFLOW'
    self.neuron_optype = "NeuronOp"
    self.excl_types = ['Placeholder', 'PlaceholderWithDefault', 'NoOp', 'Const', 'Identity', 'IdentityN', 'VarHandleOp', 'VarIsInitializedOp', 'AssignVariableOp', 'ReadVariableOp', 'StringJoin', 'ShardedFilename', 'SaveV2', 'MergeV2Checkpoints', 'RestoreV2']
    self.addl_support = ['FusedBatchNormV3', 'BatchMatMulV2', 'AddV2', 'StopGradient', self.neuron_optype]
    model = hub.load(path)
    graph_def = model.graph.as_graph_def()
    nodes = graph_def.node
    nodes = [node for node in nodes if node.op not in self.excl_types]
    self.nodetypes = [node.op for node in nodes]
    self.nodenames = [node.name for node in nodes]
    self.neuron_nodes = [(node.name, node.attr['executable'].s, self.get_tf_subgraph_types_names(node)) for node in nodes if node.op == self.neuron_optype]

  def get_mx_subgraph_types_names(self, node):
    nodetypes = []
    nodenames = []
    for sg in node['subgraphs']:
      filtered_nodes = [sg_node for sg_node in sg['nodes'] if sg_node['op'] not in self.excl_types]
      nodetypes.extend([sg_node['op'] for sg_node in filtered_nodes])
      nodenames.extend([sg_node['name'] for sg_node in filtered_nodes])
    return nodetypes, nodenames

  def load_mxnet_model(self, path):      
    import mxnet as mx
    if mx.__version__ != "1.5.1":
      try:
        import mxnetneuron as mxn
      except:
        raise "Please install mxnetneuron package."
    self.framework = 'MXNET'
    self.neuron_optype = "_neuron_subgraph_op"
    self.excl_types = ['null']
    self.addl_support = [self.neuron_optype]
    sym, args, auxs = mx.model.load_checkpoint(path, 0)
    nodes = json.loads(sym.tojson())["nodes"]
    nodes = [node for node in nodes if node['op'] not in self.excl_types]
    self.nodetypes = [node['op'] for node in nodes]
    self.nodenames = [node['name'] for node in nodes]
    neuron_nodes_tmp = [node for node in nodes if node['op'] == self.neuron_optype]
    self.neuron_nodes = [(node['name'], bytearray(args[node['name']+"_neuronbin"].asnumpy()), self.get_mx_subgraph_types_names(node)) for node in neuron_nodes_tmp]

  @staticmethod
  def get_cores_from_executable(executable):
    _NC_HEADER_SIZE = 544
    header = executable[:_NC_HEADER_SIZE]
    info = list(struct.unpack('168xI304xI64B', header))
    numCores = info.pop(0)
    numCoresRequested = info.pop(0)
    coresPerNode = info
    return  numCores, numCoresRequested, coresPerNode

  # Display table of operation type or name and whether supported or not
  def print_node_type_info(self):
    self.cnt_total = len(self.nodetypes)
    self.cnt_supported = 0
    if self.parser_args.show_names:
      widthn = max(max(map(len, self.nodenames)), 8)
      widtht = max(max(map(len, self.nodetypes)), 8)
      format_str = "{:<" + str(widthn) + "}  {:<" + str(widtht) + "}  {:<4}"
      pp = lambda x: print(format_str.format(*x))
      pp(['Op Name', 'Op Type', 'Neuron Supported ?'])
      pp(['-------', '-------', '------------------'])
      for idx, opname in enumerate(self.nodenames):
        optype = self.nodetypes[idx]
        if optype in self.supported:
          pp([opname, optype, 'Yes'])
          self.cnt_supported += 1
      for idx, opname in enumerate(self.nodenames):
        optype = self.nodetypes[idx]
        if optype not in self.supported:
          pp([opname, optype, 'No'])
    else:
      count = Counter(self.nodetypes)
      width = max(max(map(len, self.nodetypes)), 8)
      format_str = "{:<" + str(width) + "}  {:<14}  {:<4}"
      pp = lambda x: print(format_str.format(*x))
      pp(['Op Type', 'Num Instances', 'Neuron Supported ?'])
      pp(['-------', '-------------', '------------------'])
      for key in count:
        if key in self.supported:
          pp([key, count[key], 'Yes'])
          self.cnt_supported += count[key]
      for key in count:
        if key not in self.supported:
          pp([key, count[key], 'No'])
    print()

  def print_subgraph_ops(self, sg_nodetypes, sg_nodenames):
    if self.parser_args.show_names:
      widthn = max(max(map(len, sg_nodenames)), 8)
      widtht = max(max(map(len, sg_nodetypes)), 8)
      format_str = "{:<" + str(widthn) + "}  {:<" + str(widtht) + "}"
      pp = lambda x: print('    ', format_str.format(*x))
      pp(['Op Name', 'Op Type'])
      pp(['-------', '-------'])
      for idx, opname in enumerate(sg_nodenames):
        optype = sg_nodetypes[idx]
        pp([opname, optype])
    else:
      count = Counter(sg_nodetypes)
      width = max(max(map(len, sg_nodetypes)), 8)
      format_str = "{:<" + str(width) + "}  {:<14}"
      pp = lambda x: print('    ', format_str.format(*x))
      pp(['Op Type', 'Num Instances'])
      pp(['-------', '-------------'])
      for key in count:
        pp([key, count[key]])

  def print_neuron_node_info(self):
    idx = 0
    width = max(max(map(len, self.neuronop_info)), 14) 
    format_str = "{:<" + str(width) + "}  {:<14}"
    pp = lambda x: print(format_str.format(*x))
    pp(['Subgraph Name', 'Num Pipelined NeuronCores'])
    pp(['-------------', '-------------------------'])
    core_cnt_list = []
    for name, (num_cores, _, sg_nodetypes, sg_nodenames) in self.neuronop_info.items():
      pp([name, num_cores])
      core_cnt_list.append(num_cores)
      idx += 1
      if self.parser_args.expand_subgraph:
        self.print_subgraph_ops(sg_nodetypes, sg_nodenames)
    print()

  def print_neuron_support_stats(self):
    print("* Total inference operations: {}".format(self.cnt_total))
    print("* Total Neuron supported inference operations: {}".format(self.cnt_supported))
    if self.cnt_total > 0:
      perc = self.cnt_supported / self.cnt_total * 100
    else:
      perc = 0
    print("* Percent of total inference operations supported by Neuron: {:.1f}".format(perc))
    print()

  def print_common_desc(self):
    if self.parser_args.show_names:
      print("* Each line shows an operation name and whether the type of that operation is supported in Neuron.")
    else:
      print("* Each line shows an operation type, the number of instances of that type within model,\n" \
            "* and whether the type is supported in Neuron.")
    print("* Some operation types are excluded from table because they are no-operations or training-related operations:\n", \
            self.excl_types, "\n")

  def run(self):
    if len(self.neuronop_info) > 0:
      print("\n* Found {} Neuron subgraph(s) ({}(s)) in this compiled model.\n" \
            "* Use this tool on the original uncompiled model to see Neuron supported operations.\n" \
            "* The following table shows all operations, including Neuron subgraphs.".format(len(self.neuronop_info), self.neuron_optype))
      self.print_common_desc()
      self.print_node_type_info()
      print('* Please run this model on Inf1 instance with at least {} NeuronCore(s).'.format(self.min_required_pipeline_cores))
      print("* The following list show each Neuron subgraph with number of pipelined NeuronCores used by subgraph\n"\
            "* (and subgraph operations if --expand_subgraph is used):\n")
      self.print_neuron_node_info()
    else:
      print("\n* The following table shows the supported and unsupported operations within this uncompiled model.")
      self.print_common_desc()
      self.print_node_type_info()
      self.print_neuron_support_stats()
 
if __name__=='__main__':
  toolkit = neuron_parser()
  toolkit.run()
