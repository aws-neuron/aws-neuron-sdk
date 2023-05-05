import os, sys

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import ViewList

from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles


# directories
neuron1_dir = ['n1']
neuronx_dir = ['frameworks/torch/torch-neuronx/','frameworks/tensorflow/tensorflow-neuronx/','neuron-customops']
common_dir = ['tools','neuron-runtime','release-notes','containers','general','compiler','frameworks','src']
text_template = '*This document is relevant for*: '
add_inf1_tag = ['general/arch',
                'general/arch/index',
                'general/arch/neuron-hardware/neuron-hw-arch',
                'frameworks/mxnet-neuron',
                'frameworks/mxnet-neuron/index',
                'general/announcements/index',
                'frameworks/tensorflow/tensorflow-neuron/'
                ]
add_trn1_tag = ['frameworks/neuron-customops/','frameworks/torch/inference-torch-neuronx']
add_neuronx_tag = ['frameworks/torch/torch-neuronx/','frameworks/tensorflow/tensorflow-neuronx/','frameworks/torch/inference-torch-neuronx/']
clear_inf1_tag = ['general/arch/neuron-features/neuron-caching',
                'general/arch/neuron-features/eager-debug-mode',
                'general/arch/neuron-features/collective-communication-operations',
                'general/arch/neuron-features/dynamic-shapes',
                'general/arch/neuron-features/control-flow',
                'general/arch/neuron-features/custom-c++-operators',
                'tools/tutorials/tutorial-tensorboard-scalars-mnist',
                'general/arch/neuron-features/collective-communication',
                'general/appnotes/neuronx-cc/neuronx-cc-training-mixed-precision',
                'release-notes/neuron-cc/index',
                'general/arch/neuron-hardware/trn1-arch',
                'general/benchmarks/trn1-performance',
                'general/arch/neuron-features/rounding-modes',
                'tools/tutorials/tutorial-neuron-monitor-mnist',
                'tools/tensorboard/getting-started-tensorboard-neuronx-plugin',
                'release-notes/runtime/aws-neuronx-collectives/',
                'release-notes/torch/torch-neuronx/',
                'release-notes/tensorflow/tensorflow-neuronx/',
                'release-notes/compiler/neuronx-cc/',  
                'frameworks/torch/torch-neuronx/',
                'frameworks/tensorflow/tensorflow-neuronx/',
                'general/benchmarks/trn1/',
                'general/faq/training/',
                'general/devflows/training',
                'general/devflows/inference/byoc-hosting-devflow-inf2',
                'compiler/neuronx-cc/',
                'general/appnotes/perf/neuronx-cc/',
                'general/announcements/neuron2.x/dlami-pytorch-introduce',
                'general/announcements/neuron2.x/sm-training-trn1-introduce',
                'frameworks/torch/training',
                'frameworks/torch/inference-torch-neuronx',
                'frameworks/tensorflow/tensorflow-neuronx-inference',
                'general/arch/neuron-hardware/inf2-arch',
                'general/arch/neuron-hardware/inferentia2',
                'general/arch/neuron-hardware/trainium',
                'general/arch/neuron-hardware/neuron-core-v2',
                'frameworks/torch/torch-neuronx/transformers-neuronx/readme',
                'release-notes/torch/transformers-neuronx/index',
                'tools/neuron-sys-tools/nccom-test',
                'general/benchmarks/inf2/inf2-performance',
                'general/announcements/neuron2.x/sm-training-dlc-2.9.1'
                ]

clear_inf2_tag = ['frameworks/torch/torch-neuronx/training',
                  'frameworks/torch/training',
                  'frameworks/torch/inference-torch-neuron',
                  'frameworks/tensorflow/tensorflow-neuron-inference',
                  'general/arch/neuron-hardware/trn1-arch',
                  'general/arch/neuron-hardware/trainium',
                  'general/benchmarks/trn1/trn1-performance'
               ]

clear_trn1_tag = [ 'general/arch/neuron-hardware/inf2-arch', 
                    'general/arch/neuron-hardware/inferentia2',
                    'general/benchmarks/inf2/inf2-performance'
               ]

clear_nc_v2_tag = [
                'tools/tutorials/tutorial-neuron-check-model',
                'tools/tutorials/tutorial-neuron-gatherinfo',
                'tools/tutorials/getting-started-tensorboard-neuron-plugin',
                'general/appnotes/neuron-cc/mixed-precision',
                'general/arch/neuron-hardware/inf1-arch',
                'containers/dlc-then-ec2-devflow',
                'containers/dlc-then-ecs-devflow',
                'containers/dlc-then-eks-devflow',
                'containers/container-sm-hosting-devflow',
                'containers/rn',
                'general/announcements/neuron1.x/',
                'general/quick-start/mxnet-neuron',
                'tools/tensorboard/getting-started-tensorboard-neuron-plugin',
                'tools/helper-tools/tutorial-neuron-check-model',
                'tools/helper-tools/tutorial-neuron-gatherinfo',
                'containers/tutorials/k8s-neuron-scheduler',
                'general/arch/neuron-features/neuroncore-pipeline',
                'release-notes/mxnet-neuron/',
                'release-notes/torch/torch-neuron/',
                'release-notes/tensorflow/tensorflow-neuron/',
                'release-notes/compiler/neuron-cc/',
                'release-notes/neuron1/',
                'frameworks/torch/torch-neuron/',
                'frameworks/tensorflow/tensorflow-neuron/',   
                'frameworks/mxnet-neuron/',
                'general/benchmarks/inf1/',
                'general/faq/inference/',
                'compiler/neuron-cc/',
                'general/appnotes/perf/neuron-cc/',
                'general/appnotes/neuron1x/',
                'general/appnotes/torch-neuron/',
                'frameworks/torch/inference-torch-neuron',
                'frameworks/tensorflow/tensorflow-neuron-inference',
                'general/arch/neuron-hardware/inferentia',
                'general/arch/neuron-hardware/neuron-core-v1'
                ]

class NeuronTag(SphinxDirective):

    def run(self):

        return_text = ''

        # list of instances that will be applied to the page
        return_instances = []

        cur_file = self.env.docname # current file path

        


        path_split, path_len = splitall(cur_file)

        # see if it is a landing page or an index file.
        if path_split[0] == 'index': # The landing page does not need a tag.
            return_instances = []
        #elif path_split[path_len-1] == 'index': # An index file does not need a tag.
        #    return_instances = []

        # parse based on the top-level directory
        if path_split[0] in neuron1_dir: 
            return_instances = ['Inf1']
        elif path_split[0] in neuronx_dir:
            return_instances = ['Trn1','Trn1n','Inf2']
        elif path_split[0] in common_dir:
            return_instances = ['Trn1','Trn1n','Inf2','Inf1']

        # parse based on the directory where the file is.
        if path_split[path_len-2] == 'inference':
            return_instances = ['Inf1']
        elif path_split[path_len-2] == 'training':
            return_instances = ['Trn1','Trn1n']

        # add or clear tags based on file name and/or folder
        if in_list(cur_file, add_trn1_tag):
            if 'Trn1' not in return_instances:
                return_instances.append('Trn1')
                return_instances.append('Trn1n')
                return_instances.append('Inf2')

        if in_list(cur_file, add_neuronx_tag):
            if 'Trn1' not in return_instances:
                return_instances.append('Trn1')
                return_instances.append('Trn1n')
                return_instances.append('Inf2')

        if in_list(cur_file, add_inf1_tag):
            if 'Inf1' not in return_instances:
                return_instances.append('Inf1')

        if in_list(cur_file, clear_nc_v2_tag):
            if 'Trn1' in return_instances:
                return_instances.remove('Trn1')
            if 'Trn1n' in return_instances:
                return_instances.remove('Trn1n')
            if 'Inf2' in return_instances:
                return_instances.remove('Inf2')

        if in_list(cur_file, clear_trn1_tag):
            if 'Trn1' in return_instances:
                return_instances.remove('Trn1')
            if 'Trn1n' in return_instances:
                return_instances.remove('Trn1n')

        if in_list(cur_file, clear_inf1_tag):
            if 'Inf1' in return_instances:
                return_instances.remove('Inf1')
        
        if in_list(cur_file, clear_inf2_tag):
            if 'Inf2' in return_instances:
                return_instances.remove('Inf2')

        if cur_file=='general/arch/neuron-hardware/inferentia2':
             if 'Inf2' not in return_instances:
                return_instances.append('Inf2')
       
            
        # generate text from instances list if the list is not empty.
        return_instances = sorted(set(return_instances))
        if len(return_instances) > 0:
            return_text = text_template + ', '. join([str('``'+item+'``') for item in return_instances])

        rst = ViewList()
        # Add the content one line at a time.
        # Second argument is the filename to report in any warnings
        # or errors, third argument is the line number.            
        rst.append(return_text, "fakefile.rst", 10)

        # Create a node.
        node = nodes.section()
        node.document = self.state.document

        # Parse the rst.
        nested_parse_with_titles(self.state, rst, node)

        # And return the result.
        return node.children

def in_list(cur_file, file_list):

    result = any(file in cur_file for file in file_list)
    return result

def splitall(path):
    
    allparts = []

    while 1:
        parts = os.path.split(path)
        if parts[0] == path:
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    
    return allparts, len(allparts)

def setup(app):

    app.add_directive("neuron-tag", NeuronTag)
    
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
