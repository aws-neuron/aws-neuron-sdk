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
common_dir = ['tools','neuron-runtime','release-notes','containers','compiler','frameworks','src','about-neuron']
text_template = '**This document is relevant for**: '
add_inf1_tag = ['about-neuron/arch',
                'about-neuron/arch/index',
                'about-neuron/arch/neuron-hardware/neuron-hw-arch',
                'frameworks/mxnet-neuron',
                'frameworks/mxnet-neuron/index',
                'about-neuron/announcements/index',
                'frameworks/tensorflow/tensorflow-neuron/'
                ]
add_trn1_tag = ['frameworks/neuron-customops/','frameworks/torch/inference-torch-neuronx', 'libraries/nemo-megatron/', 'libraries/nxd-training/']
add_trn2_tag = ['libraries/nxd-training/', 'about-neuron/models/']
add_trn3_tag = ['about-neuron/arch/neuron-hardware/neuron-core-v4','about-neuron/arch/neuron-hardware/trn3-arch']
add_neuronx_tag = ['frameworks/torch/torch-neuronx/','frameworks/tensorflow/tensorflow-neuronx/','frameworks/torch/inference-torch-neuronx/','libraries/transformers-neuronx/','libraries/neuronx-distributed/','libraries/nxd-training', 'setup/tensorflow-neuronx']
clear_inf1_tag = ['about-neuron/arch/neuron-features/neuron-caching',
                'about-neuron/arch/neuron-features/eager-debug-mode',
                'about-neuron/arch/neuron-features/collective-communication-operations',
                'about-neuron/arch/neuron-features/dynamic-shapes',
                'about-neuron/arch/neuron-features/control-flow',
                'about-neuron/arch/neuron-features/custom-c++-operators',
                'tools/tutorials/tutorial-tensorboard-scalars-mnist',
                'about-neuron/arch/neuron-features/collective-communication',
                'about-neuron/appnotes/neuronx-cc/neuronx-cc-training-mixed-precision',
                'release-notes/neuron-cc/index',
                'about-neuron/arch/neuron-hardware/trn1-arch',
                'about-neuron/benchmarks/trn1-performance',
                'about-neuron/arch/neuron-features/rounding-modes',
                'tools/tutorials/tutorial-neuron-monitor-mnist',
                'tools/tensorboard/getting-started-tensorboard-neuronx-plugin',
                'release-notes/runtime/aws-neuronx-collectives/',
                'release-notes/torch/torch-neuronx/',
                'release-notes/tensorflow/tensorflow-neuronx/',
                'release-notes/compiler/neuronx-cc/',  
                'frameworks/torch/torch-neuronx/',
                'frameworks/tensorflow/tensorflow-neuronx/',
                'about-neuron/benchmarks/trn1/',
                'about-neuron/faq/training/',
                'devflows/training',
                'devflows/inference/byoc-hosting-devflow-inf2',
                'compiler/neuronx-cc/',
                'about-neuron/appnotes/perf/neuronx-cc/',
                'about-neuron/announcements/neuron2.x/dlami-pytorch-introduce',
                'about-neuron/announcements/neuron2.x/sm-training-trn1-introduce',
                'frameworks/torch/training',
                'frameworks/torch/inference-torch-neuronx',
                'frameworks/tensorflow/tensorflow-neuronx-inference',
                'about-neuron/arch/neuron-hardware/inf2-arch',
                'about-neuron/arch/neuron-hardware/inferentia2',
                'about-neuron/arch/neuron-hardware/trainium',
                'about-neuron/arch/neuron-hardware/neuron-core-v2',
                'frameworks/torch/torch-neuronx/transformers-neuronx/readme',
                'release-notes/torch/transformers-neuronx/index',
                'tools/neuron-sys-tools/nccom-test',
                'about-neuron/benchmarks/inf2/inf2-performance',
                'about-neuron/announcements/neuron2.x/sm-training-dlc-2.9.1',
                'about-neuron/appnotes/transformers-neuronx/generative-llm-inference-with-neuron',
                'about-neuron/calculator/neuron-calculator',
                 'about-neuron/appnotes/torch-neuronx/torch-neuronx-dataparallel-app-note',
                'setup/torch-neuronx',
                 'setup/tensorflow-neuronx',
                 'setup/neuron-setup/tensorflow/neuronx/',
                 'setup/neuron-setup/pytorch/neuronx/',
                 'about-neuron/models/inference-inf2-trn1-samples',
                 'about-neuron/models/training-trn1-samples',
                 'nki/',
                 'frameworks/jax/',
                 'about-neuron/models/training-inference-trn2-samples',
                 'about-neuron/arch/neuron-hardware/trn2-arch',
                 'about-neuron/arch/neuron-hardware/trn3-arch',
                 'about-neuron/arch/neuron-hardware/neuron-core-v3',
                 'about-neuron/arch/neuron-hardware/neuron-core-v4',
                 '/about-neuron/announcements/neuron2.x/announce-neuron-trn2',
                 '/about-neuron/arch/neuron-features/logical-neuroncore-config',
                 '/libraries/neuronx-distributed/context_parallelism_overview',
                 'libraries/nxd-training/',
                ]

clear_inf2_tag = ['frameworks/torch/torch-neuronx/training',
                  'frameworks/torch/training',
                  'frameworks/torch/inference-torch-neuron',
                  'frameworks/tensorflow/tensorflow-neuron-inference',
                  'about-neuron/arch/neuron-hardware/trn1-arch',
                  'about-neuron/arch/neuron-hardware/trainium',
                  'about-neuron/benchmarks/trn1/trn1-inference-performance',
                  'about-neuron/benchmarks/trn1/trn1-training-performance',
                  'neuronx-distributed/nxd-training',
                  'about-neuron/models/training-trn1-samples',
                  'about-neuron/models/training-inference-trn2-samples',
                  'about-neuron/arch/neuron-hardware/trn2-arch',
                  'about-neuron/arch/neuron-hardware/neuron-core-v3', 
                  'about-neuron/announcements/neuron2.x/announce-neuron-trn2',
                  'about-neuron/arch/neuron-features/logical-neuroncore-config',
                  'libraries/nxd-training/',
                  'about-neuron/arch/neuron-hardware/trn3-arch',
                  'about-neuron/arch/neuron-hardware/neuron-core-v4',
                  '/frameworks/tensorflow/'
               ]


clear_trn1_tag = [ 'about-neuron/arch/neuron-hardware/inf2-arch', 
                    'about-neuron/arch/neuron-hardware/inferentia2',
                    'about-neuron/benchmarks/inf2/inf2-performance',
                    'about-neuron/models/training-inference-trn2-samples',
                    'about-neuron/arch/neuron-hardware/trn2-arch',
                    'about-neuron/arch/neuron-hardware/trn3-arch',
                    'about-neuron/arch/neuron-hardware/trainium2',
                    'about-neuron/arch/neuron-hardware/neuron-core-v3',
                    'about-neuron/arch/neuron-hardware/neuron-core-v4',
                    '/about-neuron/announcements/neuron2.x/announce-neuron-trn2',
                    '/about-neuron/arch/neuron-features/logical-neuroncore-config'
               ]

clear_trn2_tag = [ 'frameworks/tensorflow/',
                  'libraries/transformers-neuronx/',
                  'arch/neuron-hardware/trn1-arch',
                  'arch/neuron-hardware/inf1/',
                  'about-neuron/benchmarks/',
                  'about-neuron/benchmarks/trn1/',
                  'about-neuron/benchmarks/inf2/inf2-performance',
                  'about-neuron/models/inference-inf2-trn1-samples',
                  'about-neuron/models/training-trn1-samples',
                  'about-neuron/arch/neuron-hardware/trainium',
                  'about-neuron/arch/neuron-hardware/neuron-core-v2',
                  'about-neuron/arch/neuron-hardware/neuron-core-v4',
                  'about-neuron/arch/neuron-hardware/trn3-arch',
                  '/libraries/neuronx-distributed/context_parallelism_overview',
                ]
clear_trn3_tag = [ 'frameworks/tensorflow/',
                  'libraries/transformers-neuronx/',
                  'arch/neuron-hardware/trn1-arch',
                  'arch/neuron-hardware/inf1/',
                  'about-neuron/benchmarks/',
                  'about-neuron/benchmarks/trn1/',
                  'about-neuron/benchmarks/inf2/inf2-performance',
                  'about-neuron/models/inference-inf2-trn1-samples',
                  'about-neuron/models/training-trn1-samples',
                  'about-neuron/arch/neuron-hardware/trainium',
                  'about-neuron/arch/neuron-hardware/neuron-core-v2',
                  'about-neuron/arch/neuron-hardware/neuron-core-v3',
                  'libraries/neuronx-distributed/context_parallelism_overview',
                  'about-neuron/appnotes/',
                  '/compiler/neuron-cc/'
                 ]

clear_nc_v2_tag = [
                'tools/tutorials/tutorial-neuron-check-model',
                'tools/tutorials/tutorial-neuron-gatherinfo',
                'tools/tutorials/getting-started-tensorboard-neuron-plugin',
                'about-neuron/appnotes/neuron-cc/mixed-precision',
                'about-neuron/arch/neuron-hardware/inf1-arch',
                'containers/dlc-then-ec2-devflow',
                'containers/dlc-then-ecs-devflow',
                'containers/dlc-then-eks-devflow',
                'containers/container-sm-hosting-devflow',
                'containers/rn',
                'about-neuron/announcements/neuron1.x/',
                'about-neuron/quick-start/mxnet-neuron',
                'tools/tensorboard/getting-started-tensorboard-neuron-plugin',
                'tools/helper-tools/tutorial-neuron-check-model',
                'tools/helper-tools/tutorial-neuron-gatherinfo',
                'containers/tutorials/k8s-neuron-scheduler',
                'about-neuron/arch/neuron-features/neuroncore-pipeline',
                'release-notes/mxnet-neuron/',
                'release-notes/torch/torch-neuron/',
                'release-notes/tensorflow/tensorflow-neuron/',
                'release-notes/compiler/neuron-cc/',
                'release-notes/neuron1/',
                'frameworks/torch/torch-neuron/',
                'frameworks/tensorflow/tensorflow-neuron/',   
                'frameworks/mxnet-neuron/',
                'about-neuron/benchmarks/inf1/',
                'about-neuron/faq/inference/',
                'compiler/neuron-cc/',
                'about-neuron/appnotes/perf/neuron-cc/',
                'about-neuron/appnotes/neuron1x/',
                'about-neuron/appnotes/torch-neuron/',
                'frameworks/torch/inference-torch-neuron',
                'frameworks/tensorflow/tensorflow-neuron-inference',
                'about-neuron/arch/neuron-hardware/inferentia',
                'about-neuron/arch/neuron-hardware/neuron-core-v1',
                'setup/tensorflow-neuron',
                'setup/neuron-setup/pytorch/neuron/',
                'setup/mxnet-neuron/',
                'setup/torch-neuron',
                'setup/mxnet-neuron',
                'setup/neuron-setup/mxnet/neuron/ubuntu/',
                'setup/neuron-setup/mxnet/neuron/amazon-linux/',
                'setup/neuron-setup/tensorflow/neuron/ubuntu/',
                'setup/neuron-setup/tensorflow/neuron/amazon-linux/',
                'about-neuron/models/inference-inf1-samples'
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
            return_instances = ['Trn1','Trn2','Trn3','Inf2']
        elif path_split[0] in common_dir:
            return_instances = ['Trn1','Trn2','Inf1','Inf2','Trn3']

        # parse based on the directory where the file is.
        if path_split[path_len-2] == 'inference':
            return_instances = ['Inf1']
        elif path_split[path_len-2] == 'training':
            return_instances = ['Trn1','Trn2','Trn3']

        # add or clear tags based on file name and/or folder
        if in_list(cur_file, add_trn1_tag):
            if 'Trn1' not in return_instances:
                return_instances.append('Trn1')
                return_instances.append('Trn2')
                return_instances.append('Trn3')
                return_instances.append('Inf2')

        if in_list(cur_file, add_trn2_tag):
            if 'Trn2' not in return_instances:
                return_instances.append('Trn2')
                return_instances.append('Trn3')

        if in_list(cur_file, add_trn3_tag):
            if 'Trn3' not in return_instances:
                return_instances.append('Trn3')

        if in_list(cur_file, add_neuronx_tag):
            if 'Trn1' not in return_instances:
                return_instances.append('Trn1')
                return_instances.append('Trn2')
                return_instances.append('Trn3')
                return_instances.append('Inf2')

        if in_list(cur_file, add_inf1_tag):
            if 'Inf1' not in return_instances:
                return_instances.append('Inf1')

        if in_list(cur_file, clear_nc_v2_tag):
            if 'Trn1' in return_instances:
                return_instances.remove('Trn1')
            if 'Trn2' in return_instances:
                return_instances.remove('Trn2')
            if 'Inf2' in return_instances:
                return_instances.remove('Inf2')

        if in_list(cur_file, clear_trn1_tag):
            if 'Trn1' in return_instances:
                return_instances.remove('Trn1')

        if in_list(cur_file, clear_trn2_tag):
            if 'Trn2' in return_instances:
                return_instances.remove('Trn2')
        
        if in_list(cur_file, clear_trn3_tag):
            if 'Trn3' in return_instances:
                return_instances.remove('Trn3')

        if in_list(cur_file, clear_inf1_tag):
            if 'Inf1' in return_instances:
                return_instances.remove('Inf1')
        
        if in_list(cur_file, clear_inf2_tag):
            if 'Inf2' in return_instances:
                return_instances.remove('Inf2')

        if cur_file=='about-neuron/arch/neuron-hardware/inferentia2':
            if 'Inf2' not in return_instances:
                return_instances.append('Inf2')

        if cur_file=='about-neuron/arch/neuron-hardware/trainium2':
            if 'Trn2' not in return_instances:
                return_instances.append('Trn2')

        if cur_file=='about-neuron/arch/neuron-hardware/trainium3':
            if 'Trn2' not in return_instances:
                return_instances.append('Trn3')
       
        if cur_file=='frameworks/torch/inference-torch-neuronx' or cur_file=='setup/torch-neuronx' or cur_file=='setup/tensorflow-neuronx':
            if 'Inf2' not in return_instances:
                return_instances.append('Inf2')
                return_instances.append('Trn1')
                return_instances.append('Trn2')
                return_instances.append('Trn2')

        if cur_file=='about-neuron/appnotes/neuronx-distributed/introducing-nxdt-training':
            return_instances = ['Trn1','Trn2','Trn3']
            
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
