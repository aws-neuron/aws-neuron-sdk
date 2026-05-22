import os

from docutils import nodes
from docutils.statemachine import ViewList

from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles


# =============================================================================
# Legacy add/clear lists (used only for files NOT handled by explicit overrides)
# =============================================================================

# These lists use substring matching via in_list(). They apply ONLY when no
# explicit_override was set. As more paths get explicit overrides, entries
# here become dead code. Kept for backward compatibility with paths not yet
# explicitly overridden.

add_inf1_tag = [
    'about-neuron/arch',
    'archive/mxnet-neuron',
    'about-neuron/announcements/index',
    'archive/tensorflow/tensorflow-neuron/',
]

add_trn1_tag = [
    'frameworks/neuron-customops/',
    'neuron-customops/',
    'frameworks/torch/inference-torch-neuronx',
    'libraries/nemo-megatron/',
    'libraries/nxd-training/',
]

add_trn2_tag = [
    'libraries/nxd-training/',
    'about-neuron/models/',
]

add_trn3_tag = [
    'about-neuron/arch/neuron-hardware/neuron-core-v4',
    'about-neuron/arch/neuron-hardware/trn3-arch',
]

add_neuronx_tag = [
    'frameworks/torch/torch-neuronx/',
    'archive/tensorflow/tensorflow-neuronx/',
    'frameworks/torch/inference-torch-neuronx/',
    'libraries/neuronx-distributed/',
    'libraries/nxd-training',
    'setup/tensorflow-neuronx',
]

clear_inf1_tag = [
    'about-neuron/arch/neuron-features/neuron-caching',
    'about-neuron/arch/neuron-features/eager-debug-mode',
    'about-neuron/arch/neuron-features/collective-communication-operations',
    'about-neuron/arch/neuron-features/dynamic-shapes',
    'about-neuron/arch/neuron-features/control-flow',
    'about-neuron/arch/neuron-features/custom-c++-operators',
    'about-neuron/arch/neuron-features/collective-communication',
    'about-neuron/arch/neuron-features/rounding-modes',
    'about-neuron/arch/neuron-hardware/trn1-arch',
    'about-neuron/arch/neuron-hardware/inf2-arch',
    'about-neuron/arch/neuron-hardware/inferentia2',
    'about-neuron/arch/neuron-hardware/trainium',
    'about-neuron/arch/neuron-hardware/neuron-core-v2',
    'about-neuron/arch/neuron-hardware/trn2-arch',
    'about-neuron/arch/neuron-hardware/trn3-arch',
    'about-neuron/arch/neuron-hardware/neuron-core-v3',
    'about-neuron/arch/neuron-hardware/neuron-core-v4',
    'about-neuron/benchmarks/trn1-performance',
    'about-neuron/benchmarks/trn1/',
    'about-neuron/benchmarks/inf2/inf2-performance',
    'about-neuron/faq/training/',
    'about-neuron/models/inference-inf2-trn1-samples',
    'about-neuron/models/training-trn1-samples',
    'about-neuron/models/training-inference-trn2-samples',
    'about-neuron/appnotes/neuronx-cc/neuronx-cc-training-mixed-precision',
    'about-neuron/appnotes/transformers-neuronx/generative-llm-inference-with-neuron',
    'about-neuron/appnotes/torch-neuronx/torch-neuronx-dataparallel-app-note',
    'about-neuron/calculator/neuron-calculator',
    'about-neuron/announcements/neuron2.x/dlami-pytorch-introduce',
    'about-neuron/announcements/neuron2.x/sm-training-trn1-introduce',
    'about-neuron/announcements/neuron2.x/sm-training-dlc-2.9.1',
    'devflows/training',
    'devflows/inference/byoc-hosting-devflow-inf2',
    'compiler/neuronx-cc/',
    'about-neuron/appnotes/perf/neuronx-cc/',
    'frameworks/torch/torch-neuronx/',
    'frameworks/torch/training',
    'frameworks/torch/inference-torch-neuronx',
    'archive/tensorflow/tensorflow-neuronx/',
    'archive/tensorflow/tensorflow-neuronx-inference',
    'frameworks/torch/torch-neuronx/transformers-neuronx/readme',
    'release-notes/neuron-cc/index',
    'release-notes/runtime/aws-neuronx-collectives/',
    'release-notes/torch/torch-neuronx/',
    'release-notes/torch/transformers-neuronx/index',
    'release-notes/tensorflow/tensorflow-neuronx/',
    'release-notes/compiler/neuronx-cc/',
    'archive/tensorboard/tutorial-tensorboard-scalars-mnist',
    'tools/tutorials/tutorial-neuron-monitor-mnist',
    'archive/tensorboard/getting-started-tensorboard-neuronx-plugin',
    'tools/neuron-sys-tools/nccom-test',
    'setup/torch-neuronx',
    'setup/tensorflow-neuronx',
    'setup/neuron-setup/tensorflow/neuronx/',
    'setup/neuron-setup/pytorch/neuronx/',
    'nki/',
    'frameworks/jax/',
    'libraries/nxd-training/',
    '/release-notes/components/nki',
    '/release-notes/components/nki-lib',
    '/release-notes/components/compiler'
]

clear_inf2_tag = [
    'frameworks/torch/torch-neuronx/training',
    'frameworks/torch/training',
    'archive/torch-neuron/inference-torch-neuron',
    'archive/tensorflow/tensorflow-neuron-inference',
    'frameworks/jax/',
    'about-neuron/arch/neuron-hardware/trn1-arch',
    'about-neuron/arch/neuron-hardware/trainium',
    'about-neuron/arch/neuron-hardware/trn2-arch',
    'about-neuron/arch/neuron-hardware/trn3-arch',
    'about-neuron/arch/neuron-hardware/neuron-core-v3',
    'about-neuron/arch/neuron-hardware/neuron-core-v4',
    'about-neuron/arch/neuron-features/logical-neuroncore-config',
    'about-neuron/benchmarks/trn1/trn1-inference-performance',
    'about-neuron/benchmarks/trn1/trn1-training-performance',
    'about-neuron/models/training-trn1-samples',
    'about-neuron/models/training-inference-trn2-samples',
    'about-neuron/announcements/neuron2.x/announce-neuron-trn2',
    'neuronx-distributed/nxd-training',
    'libraries/nxd-training/',
    'tools/neuron-sys-tools/nccom-test',
    'release-notes/runtime/aws-neuronx-collectives/',
]

clear_trn1_tag = [
    'about-neuron/arch/neuron-hardware/inf2-arch',
    'about-neuron/arch/neuron-hardware/inferentia2',
    'about-neuron/arch/neuron-hardware/trn2-arch',
    'about-neuron/arch/neuron-hardware/trn3-arch',
    'about-neuron/arch/neuron-hardware/trainium2',
    'about-neuron/arch/neuron-hardware/neuron-core-v3',
    'about-neuron/arch/neuron-hardware/neuron-core-v4',
    'about-neuron/benchmarks/inf2/inf2-performance',
    'about-neuron/models/training-inference-trn2-samples',
]

clear_trn2_tag = [
    'archive/tensorflow/',
    'libraries/transformers-neuronx/',
    'about-neuron/arch/neuron-hardware/trn1-arch',
    'about-neuron/arch/neuron-hardware/trainium',
    'about-neuron/arch/neuron-hardware/neuron-core-v2',
    'about-neuron/arch/neuron-hardware/neuron-core-v4',
    'about-neuron/arch/neuron-hardware/trn3-arch',
    'about-neuron/benchmarks/',
    'about-neuron/benchmarks/trn1/',
    'about-neuron/benchmarks/inf2/inf2-performance',
    'about-neuron/models/inference-inf2-trn1-samples',
    'about-neuron/models/training-trn1-samples',
    'neuron-customops/programming-guide/custom-c++-operators-devguide'
]

clear_trn3_tag = [
    'archive/tensorflow/',
    'libraries/transformers-neuronx/',
    'about-neuron/arch/neuron-hardware/trn1-arch',
    'about-neuron/arch/neuron-hardware/trainium',
    'about-neuron/arch/neuron-hardware/neuron-core-v2',
    'about-neuron/arch/neuron-hardware/neuron-core-v3',
    'about-neuron/benchmarks/',
    'about-neuron/benchmarks/trn1/',
    'about-neuron/benchmarks/inf2/inf2-performance',
    'about-neuron/models/inference-inf2-trn1-samples',
    'about-neuron/models/training-trn1-samples',
    'libraries/neuronx-distributed/context_parallelism_overview',
    'about-neuron/appnotes/',
    'neuron-customops/programming-guide/custom-c++-operators-devguide'
]

# Neuron 1.x / NeuronCore v1 era content — clear all non-Inf1 tags
clear_nc_v2_tag = [
    'tools/tutorials/tutorial-neuron-check-model',
    'tools/tutorials/tutorial-neuron-gatherinfo',
    'tools/tutorials/getting-started-tensorboard-neuron-plugin',
    'archive/tensorboard/getting-started-tensorboard-neuron-plugin',
    'tools/helper-tools/tutorial-neuron-check-model',
    'tools/helper-tools/tutorial-neuron-gatherinfo',
    'about-neuron/appnotes/neuron-cc/mixed-precision',
    'about-neuron/appnotes/perf/neuron-cc/',
    'about-neuron/appnotes/neuron1x/',
    'about-neuron/appnotes/torch-neuron/',
    'about-neuron/arch/neuron-hardware/inf1-arch',
    'about-neuron/arch/neuron-hardware/inferentia',
    'about-neuron/arch/neuron-hardware/neuron-core-v1',
    'about-neuron/arch/neuron-features/neuroncore-pipeline',
    'about-neuron/announcements/neuron1.x/',
    'about-neuron/quick-start/mxnet-neuron',
    'about-neuron/benchmarks/inf1/',
    'about-neuron/faq/inference/',
    'about-neuron/models/inference-inf1-samples',
    'containers/dlc-then-ec2-devflow',
    'containers/dlc-then-ecs-devflow',
    'containers/dlc-then-eks-devflow',
    'containers/container-sm-hosting-devflow',
    'containers/rn',
    'containers/tutorials/k8s-neuron-scheduler',
    'compiler/neuron-cc/',
    'release-notes/mxnet-neuron/',
    'release-notes/torch/torch-neuron/',
    'release-notes/tensorflow/tensorflow-neuron/',
    'release-notes/compiler/neuron-cc/',
    'release-notes/neuron1/',
    'archive/torch-neuron/',
    'archive/torch-neuron/inference-torch-neuron',
    'archive/tensorflow/tensorflow-neuron/',
    'archive/tensorflow/tensorflow-neuron-inference',
    'archive/mxnet-neuron/',
    'setup/tensorflow-neuron',
    'setup/torch-neuron',
    'setup/mxnet-neuron',
    'setup/neuron-setup/pytorch/neuron/',
    'setup/neuron-setup/mxnet/neuron/ubuntu/',
    'setup/neuron-setup/mxnet/neuron/amazon-linux/',
    'setup/neuron-setup/tensorflow/neuron/ubuntu/',
    'setup/neuron-setup/tensorflow/neuron/amazon-linux/',
]

# Top-level directories used for initial tag assignment
NEURON1_DIRS = ['n1']
COMMON_DIRS = [
    'tools', 'neuron-runtime', 'release-notes', 'containers', 'compiler',
    'frameworks', 'src', 'about-neuron', 'setup', 'devflows', 'dlami', 'libraries',
]

TEXT_TEMPLATE = '**This document is relevant for**: '


# =============================================================================
# Hardware architecture page map (exact docname → instance list)
# =============================================================================

HW_ARCH_MAP = {
    'about-neuron/arch/neuron-hardware/inf1-arch': ['Inf1'],
    'about-neuron/arch/neuron-hardware/inf2-arch': ['Inf2'],
    'about-neuron/arch/neuron-hardware/inferentia': ['Inf1'],
    'about-neuron/arch/neuron-hardware/inferentia2': ['Inf2'],
    'about-neuron/arch/neuron-hardware/neuron-core-v1': ['Inf1'],
    'about-neuron/arch/neuron-hardware/neuron-core-v2': ['Inf2', 'Trn1'],
    'about-neuron/arch/neuron-hardware/neuron-core-v3': ['Trn2'],
    'about-neuron/arch/neuron-hardware/neuron-core-v4': ['Trn3'],
    'about-neuron/arch/neuron-hardware/trainium': ['Trn1'],
    'about-neuron/arch/neuron-hardware/trainium2': ['Trn2'],
    'about-neuron/arch/neuron-hardware/trainium3': ['Trn3'],
    'about-neuron/arch/neuron-hardware/trn1-arch': ['Trn1'],
    'about-neuron/arch/neuron-hardware/trn2-arch': ['Trn2'],
    'about-neuron/arch/neuron-hardware/trn3-arch': ['Trn3'],
}

# NxD Core training-specific pages (no Inf2)
NXD_CORE_TRAINING_PAGES = [
    'libraries/neuronx-distributed/index-training',
    'libraries/neuronx-distributed/developer-guide-training',
    'libraries/neuronx-distributed/api-reference-guide-training',
    'libraries/neuronx-distributed/tp_developer_guide',
    'libraries/neuronx-distributed/pp_developer_guide',
    'libraries/neuronx-distributed/ptl_developer_guide',
    'libraries/neuronx-distributed/save_load_developer_guide',
    'libraries/neuronx-distributed/activation_memory_reduction',
    'libraries/neuronx-distributed/activation_memory_reduction_developer_guide',
    'libraries/neuronx-distributed/standard_mixed_precision',
    'libraries/neuronx-distributed/tensor_parallelism_overview',
    'libraries/neuronx-distributed/pipeline_parallelism_overview',
    'libraries/neuronx-distributed/lora_finetune_developer_guide',
    'libraries/neuronx-distributed/model_optimizer_wrapper_developer_guide',
    'libraries/neuronx-distributed/context_parallelism_overview',
]


def _in_list(cur_file, file_list):
    """Return True if any entry in file_list is a substring of cur_file."""
    return any(entry in cur_file for entry in file_list)


def _splitall(path):
    """Split a path into all its components."""
    parts = []
    while True:
        head, tail = os.path.split(path)
        if head == path:
            parts.insert(0, head)
            break
        elif tail == path:
            parts.insert(0, tail)
            break
        else:
            path = head
            parts.insert(0, tail)
    return parts, len(parts)


def _get_explicit_override(cur_file):
    """Return (instances, True) if cur_file has an explicit CSV-based override,
    or (None, False) otherwise.

    Rules are evaluated top-to-bottom. More specific paths must come AFTER
    broader paths so they can override them (last match wins).
    """

    # --- Libraries -----------------------------------------------------------

    # NxD Core = Inf2, Trn1, Trn2 (default for all neuronx-distributed pages)
    if cur_file.startswith('libraries/neuronx-distributed/'):
        result = ['Inf2', 'Trn1', 'Trn2']
        # Training-specific pages drop Inf2
        if cur_file in NXD_CORE_TRAINING_PAGES:
            result = ['Trn1', 'Trn2']
        if cur_file.startswith('libraries/neuronx-distributed/tutorials/training') or \
           cur_file.startswith('libraries/neuronx-distributed/tutorials/finetune'):
            result = ['Trn1', 'Trn2']
        return result, True

    if cur_file.startswith('libraries/transformers-neuronx/'):
        return ['Inf2', 'Trn1'], True

    if cur_file.startswith('libraries/nxd-training/'):
        return ['Trn1', 'Trn2'], True

    # vLLM must come before general nxd-inference
    if cur_file.startswith('libraries/nxd-inference/vllm/'):
        return ['Trn2', 'Trn3'], True

    if cur_file.startswith('libraries/nxd-inference/'):
        return ['Inf2', 'Trn1', 'Trn2'], True

    if cur_file.startswith('libraries/nemo-megatron/'):
        return ['Trn1', 'Trn2'], True

    # --- NKI -----------------------------------------------------------------

    if cur_file.startswith('nki/'):
        return ['Trn2', 'Trn3'], True

    # --- CustomOps -----------------------------------------------------------

    if cur_file.startswith('neuron-customops/'):
        return ['Inf2', 'Trn1'], True

    # --- Frameworks ----------------------------------------------------------

    if cur_file.startswith('frameworks/jax/'):
        return ['Trn2', 'Trn3'], True

    # TensorFlow NeuronX (must come before TensorFlow Neuron check)
    if 'tensorflow/tensorflow-neuronx' in cur_file:
        return ['Inf2', 'Trn1'], True

    # TensorFlow Neuron (Inf1)
    if 'tensorflow/tensorflow-neuron' in cur_file and 'neuronx' not in cur_file:
        return ['Inf1'], True

    # TorchNeuron native PyTorch (must come before torch-neuronx check)
    if 'torch/pytorch-native' in cur_file:
        return ['Trn2', 'Trn3'], True

    # PyTorch NeuronX (Torch/XLA)
    if 'torch/torch-neuronx' in cur_file:
        return ['Inf2', 'Trn1', 'Trn2'], True

    # PyTorch NeuronX top-level pages (not in torch-neuronx/ subdir)
    if cur_file in ['frameworks/torch/inference-torch-neuronx',
                     'frameworks/torch/training-torch-neuronx',
                     'frameworks/torch/training',
                     'frameworks/torch/inference']:
        return ['Inf2', 'Trn1', 'Trn2'], True

    # PyTorch Neuron (Inf1)
    if 'torch/torch-neuron' in cur_file and 'neuronx' not in cur_file:
        return ['Inf1'], True

    if cur_file == 'archive/torch-neuron/inference-torch-neuron':
        return ['Inf1'], True

    # MXNet
    if 'mxnet-neuron' in cur_file:
        return ['Inf1'], True

    # --- Neuron Runtime ------------------------------------------------------

    # Collectives (more specific, must come after general runtime)
    if cur_file.startswith('neuron-runtime/about/collectives') or \
       cur_file in ['neuron-runtime/explore/internode-collective-comm',
                     'neuron-runtime/explore/intranode-collective-comm',
                     'neuron-runtime/explore/compute-comm-overlap']:
        return ['Trn1', 'Trn2', 'Trn3'], True

    if cur_file.startswith('neuron-runtime/'):
        return ['Inf2', 'Trn1', 'Trn2', 'Trn3'], True

    # --- Compiler ------------------------------------------------------------

    if cur_file.startswith('compiler/error-codes/'):
        return ['Inf2', 'Trn1', 'Trn2', 'Trn3'], True

    if cur_file == 'compiler/neuron-cc' or cur_file.startswith('compiler/neuron-cc/'):
        return ['Inf1'], True

    if cur_file == 'compiler/neuronx-cc' or cur_file.startswith('compiler/neuronx-cc/'):
        return ['Inf2', 'Trn1', 'Trn2', 'Trn3'], True

    if cur_file == 'neuron-customops/programming-guide' or cur_file.startswith('neuron-customops/programming-guide'):
        return ['Inf2', 'Trn1'], True

    # --- Setup ---------------------------------------------------------------

    if cur_file.startswith('setup/install-templates/inf1/'):
        return ['Inf1'], True
    if cur_file.startswith('setup/install-templates/inf2/'):
        return ['Inf2'], True
    if cur_file.startswith('setup/install-templates/trn1/') or \
       cur_file == 'setup/install-templates/launch-trn1-dlami':
        return ['Trn1'], True

    if cur_file in ['setup/setup-neuron', 'setup/torch-neuron', 'setup/torch-neuron-ubuntu20']:
        return ['Inf1'], True

    if cur_file.startswith('setup/neuron-setup/pytorch/neuronx/'):
        return ['Inf2', 'Trn1', 'Trn2'], True
    if cur_file.startswith('setup/neuron-setup/tensorflow/neuronx/'):
        return ['Inf2', 'Trn1'], True
    if cur_file.startswith('setup/neuron-setup/pytorch/neuron/'):
        return ['Inf1'], True
    if cur_file.startswith('setup/neuron-setup/tensorflow/neuron/'):
        return ['Inf1'], True

    if cur_file == 'setup/jax-neuronx':
        return ['Trn2', 'Trn3'], True
    if cur_file == 'setup/torch-neuronx':
        return ['Inf2', 'Trn1', 'Trn2'], True
    if cur_file == 'setup/tensorflow-neuronx':
        return ['Inf2', 'Trn1'], True
    if cur_file == 'setup/tensorflow-neuron':
        return ['Inf1'], True

    return None, False


def _get_page_override(cur_file):
    """Return (instances, True) for page-specific overrides that don't fit
    neatly into _get_explicit_override (devflows, containers, tools, about-neuron, etc.).
    """

    # --- Devflows ------------------------------------------------------------

    if cur_file == 'devflows/inference/byoc-hosting-devflow-inf2':
        return ['Inf2'], True
    if cur_file == 'devflows/inference/ec2-then-ec2-devflow-inf2':
        return ['Inf2'], True
    if cur_file == 'devflows/parallelcluster-flows':
        return ['Trn1', 'Trn2'], True

    if cur_file.startswith('devflows/training/batch/') or \
       cur_file.startswith('devflows/training/ec2/') or \
       cur_file.startswith('devflows/training/parallelcluster/') or \
       cur_file.startswith('devflows/training/sm-devflow/'):
        return ['Trn1', 'Trn2', 'Trn3'], True

    if cur_file.startswith('devflows/plugins/npd'):
        return ['Inf2', 'Trn1', 'Trn2'], True

    # --- Containers ----------------------------------------------------------

    # OCI Hooks
    if 'tutorial-oci-hook' in cur_file:
        return ['Inf1', 'Inf2', 'Trn1', 'Trn2'], True

    # DRA
    if cur_file == 'containers/neuron-dra' or cur_file.startswith('containers/files/'):
        return ['Trn2', 'Trn3'], True

    if cur_file == 'containers/how-to/how-to-ultraserver':
        return ['Trn2', 'Trn3'], True

    # DLC quickstarts
    if cur_file == 'containers/get-started/quickstart-configure-deploy-dlc':
        return ['Trn2', 'Trn3'], True
    if cur_file == 'containers/get-started/quickstart-pytorch-inference-dlc':
        return ['Inf2', 'Trn1', 'Trn2', 'Trn3'], True

    # Inf1-era container content
    if cur_file == 'containers/tutorial-docker-runtime1.0':
        return ['Inf1'], True
    if cur_file == 'containers/container-deployment-flows' or \
       cur_file.startswith('containers/docker-example/inference/') or \
       cur_file.startswith('containers/docker-example/v1/') or \
       cur_file == 'containers/ec2-then-ec2-devflow' or \
       cur_file == 'containers/neo-then-hosting-devflow':
        return ['Inf1'], True

    # Container training/inference tutorials and docker examples
    if cur_file.startswith('containers/docker-example/training/'):
        return ['Trn1', 'Trn2', 'Trn3'], True
    if cur_file.startswith('containers/tutorials/inference/'):
        return ['Inf1'], True
    if cur_file.startswith('containers/tutorials/training/'):
        return ['Trn1', 'Trn2', 'Trn3'], True

    # Neuron Monitor Container
    if cur_file == 'containers/tutorials/k8s-neuron-monitor':
        return ['Inf2', 'Trn1', 'Trn2'], True

    # Node Problem Detector
    if cur_file.startswith('containers/tutorials/k8s-neuron-problem-detector'):
        return ['Inf2', 'Trn1', 'Trn2'], True

    # --- Tools ---------------------------------------------------------------

    # TensorBoard plugin (End Of Support)
    if cur_file == 'archive/tensorboard/getting-started-tensorboard-neuronx-plugin' or \
       cur_file == 'archive/tensorboard/tutorial-tensorboard-scalars-mnist' or \
       cur_file == 'archive/tensorboard/torch-neuronx-profiling-with-tb':
        return ['Inf2', 'Trn1'], True

    # --- Announcements -------------------------------------------------------

    if cur_file.startswith('about-neuron/announcements/'):
        return [], True

    # --- Hardware architecture -----------------------------------------------

    if cur_file in HW_ARCH_MAP:
        return HW_ARCH_MAP[cur_file], True

    # --- Arch features -------------------------------------------------------

    if cur_file == 'about-neuron/arch/neuron-features/custom-c++-operators':
        return ['Inf2', 'Trn1'], True
    if cur_file == 'about-neuron/arch/neuron-features/logical-neuroncore-config':
        return ['Trn2', 'Trn3'], True

    # --- Appnotes ------------------------------------------------------------

    if cur_file == 'about-neuron/appnotes/neuronx-distributed/introducing-nxd-inference':
        return ['Inf2', 'Trn1', 'Trn2'], True
    if cur_file == 'about-neuron/appnotes/neuronx-distributed/introducing-nxdt-training':
        return ['Trn1', 'Trn2'], True
    if cur_file.startswith('about-neuron/appnotes/torch-neuronx/'):
        return ['Inf2', 'Trn1', 'Trn2'], True
    if cur_file.startswith('about-neuron/appnotes/transformers-neuronx/'):
        return ['Inf2', 'Trn1'], True
    if cur_file == 'about-neuron/appnotes/neuronx-cc/neuronx-cc-training-mixed-precision':
        return ['Trn1', 'Trn2', 'Trn3'], True
    if cur_file.startswith('about-neuron/appnotes/neuron1x/'):
        return ['Inf1'], True

    # --- Benchmarks ----------------------------------------------------------

    if cur_file == 'about-neuron/benchmarks/index':
        return ['Inf1', 'Inf2', 'Trn1', 'Trn2', 'Trn3'], True

    # --- Quick-start ---------------------------------------------------------

    if cur_file == 'about-neuron/quick-start/tensorflow-neuron':
        return ['Inf1'], True
    if cur_file in ['about-neuron/quick-start/torch-neuron',
                     'about-neuron/quick-start/torch-neuron-tab-training']:
        return ['Inf1'], True

    if cur_file.startswith('about-neuron/quick-start/tab-inference-torch-neuronx'):
        return ['Inf2', 'Trn1', 'Trn2'], True
    if cur_file.startswith('about-neuron/quick-start/tab-inference-torch-neuron') and 'neuronx' not in cur_file:
        return ['Inf1'], True
    if cur_file.startswith('about-neuron/quick-start/tab-inference-tensorflow-neuronx'):
        return ['Inf2', 'Trn1'], True
    if cur_file.startswith('about-neuron/quick-start/tab-inference-tensorflow-neuron') and 'neuronx' not in cur_file:
        return ['Inf1'], True

    return None, False


class NeuronTag(SphinxDirective):

    def run(self):
        cur_file = self.env.docname
        path_split, path_len = _splitall(cur_file)

        # Landing page gets no tag
        if path_split[0] == 'index':
            return self._render('')

        # Step 1: Assign default instances based on top-level directory
        return_instances = []
        if path_split[0] in NEURON1_DIRS:
            return_instances = ['Inf1']
        elif path_split[0] in COMMON_DIRS:
            return_instances = ['Inf1', 'Inf2', 'Trn1', 'Trn2', 'Trn3']

        # Step 2: Check explicit overrides (CSV-based, highest priority)
        explicit_override = False

        result, matched = _get_explicit_override(cur_file)
        if matched:
            return_instances = result
            explicit_override = True

        if not explicit_override:
            result, matched = _get_page_override(cur_file)
            if matched:
                return_instances = result
                explicit_override = True

        # Step 3: Directory-based inference/training heuristic
        if not explicit_override:
            if path_len >= 2:
                parent_dir = path_split[path_len - 2]
                if parent_dir == 'inference':
                    return_instances = ['Inf1']
                elif parent_dir == 'training':
                    return_instances = ['Trn1', 'Trn2', 'Trn3']

        # Step 4: Legacy add/clear tag lists (only for non-overridden files)
        if not explicit_override:
            if _in_list(cur_file, add_trn1_tag):
                if 'Trn1' not in return_instances:
                    return_instances.extend(['Trn1', 'Trn2', 'Trn3', 'Inf2'])

            if _in_list(cur_file, add_trn2_tag):
                if 'Trn2' not in return_instances:
                    return_instances.extend(['Trn2', 'Trn3'])

            if _in_list(cur_file, add_trn3_tag):
                if 'Trn3' not in return_instances:
                    return_instances.append('Trn3')

            if _in_list(cur_file, add_neuronx_tag):
                if 'Trn1' not in return_instances:
                    return_instances.extend(['Trn1', 'Trn2', 'Trn3', 'Inf2'])

            if _in_list(cur_file, add_inf1_tag):
                if 'Inf1' not in return_instances:
                    return_instances.append('Inf1')

            if _in_list(cur_file, clear_nc_v2_tag):
                for tag in ['Trn1', 'Trn2', 'Trn3', 'Inf2']:
                    if tag in return_instances:
                        return_instances.remove(tag)

            if _in_list(cur_file, clear_trn1_tag):
                if 'Trn1' in return_instances:
                    return_instances.remove('Trn1')

            if _in_list(cur_file, clear_trn2_tag):
                if 'Trn2' in return_instances:
                    return_instances.remove('Trn2')

            if _in_list(cur_file, clear_trn3_tag):
                if 'Trn3' in return_instances:
                    return_instances.remove('Trn3')

            if _in_list(cur_file, clear_inf1_tag):
                if 'Inf1' in return_instances:
                    return_instances.remove('Inf1')

            if _in_list(cur_file, clear_inf2_tag):
                if 'Inf2' in return_instances:
                    return_instances.remove('Inf2')

        # Step 5: Generate output
        return_instances = sorted(set(return_instances))
        if return_instances:
            text = TEXT_TEMPLATE + ', '.join('``' + i + '``' for i in return_instances)
        else:
            text = ''

        return self._render(text)

    def _render(self, text):
        """Parse RST text and return docutils nodes."""
        rst = ViewList()
        rst.append(text, "neuron-tag", 1)
        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, rst, node)
        return node.children


def setup(app):
    app.add_directive("neuron-tag", NeuronTag)
    return {
        'version': '0.2',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
