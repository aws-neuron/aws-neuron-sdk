.. _torch-neuron-r-cnn-app-note:

Running R-CNNs on Inf1
======================

This application note demonstrates how to compile and run
`Detectron2 <https://github.com/facebookresearch/detectron2>`__-based
R-CNNs on Inf1. It also provides guidance on how to use profiling to
improve the performance of R-CNN models on Inf1.

.. contents:: Table of contents
   :local:


R-CNN Model Overview
--------------------

Region-based CNN (R-CNN) models are commonly used for object detection
and image segmentation tasks. A typical R-CNN architecture is composed
of the following components:

-  **Backbone:** The backbone extracts features from input images. In
   some models, the backbone is a Feature Pyramid Network (FPN), which
   uses a top-down architecture with lateral connections to build an
   in-network feature pyramid from a single-scale input. The backbone is
   commonly a ResNet or Vision Transformer based network.
-  **Region Proposal Network (RPN):** The RPN predicts region proposals
   with a wide range of scales and aspect ratios. RPNs are constructed
   using convolutional layers and anchor boxes that serve as references
   for multiple scales and aspect ratios.
-  **Region of Interest (RoI):** The RoI component is used to resize the
   extracted features that have varying size to the same size so that
   they can be consumed by a fully connected layer. RoI Align is
   typically used instead of RoI Pooling because RoI Align provides
   better alignment.

The `Detectron2 <https://github.com/facebookresearch/detectron2>`__
library provides many popular PyTorch R-CNN implementations, including
R-CNN, Fast R-CNN, Faster R-CNN, and Mask R-CNN. This application note will
focus on the Detectron2 R-CNN models.

R-CNN Limitations and Considerations on Inferentia (NeuronCore-v1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

R-CNN models can have a few limitations and considerations on Inferentia
(NeuronCore-v1). See the :ref:`Model Architecture Fit Guidelines
<rcnn_limitations_inf1>` for more information. These limitatins are not
applicable to NeuronCore-v2.

Requirements
------------

This application note is intended to be run on an ``inf1.2xlarge``. In practice,
R-CNN models can be run on any Inf1 instance size.

Verify that this Jupyter notebook is running the Python kernel
environment that was set up according to the `PyTorch Installation
Guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/setup/pytorch-install.html>`__.
You can select the kernel from the “Kernel -> Change Kernel” option on
the top of this Jupyter notebook page.

Installation
------------

This application note requires the following pip packages:

1. ``torch==1.11.0``
2. ``torch-neuron``
3. ``neuron-cc``
4. ``opencv-python``
5. ``pycocotools``
6. ``torchvision==0.12.0``
7. ``detectron2==0.6``

The following section builds ``torchvision`` from source and installs
the ``Detectron2`` package. It also reinstalls the Neuron packages to ensure
version compability.

The the ``torchvision`` ``roi_align_kernel.cpp`` kernel is modified to
use OMP threading for multithreaded inference on CPU. This significantly
improves the performance of RoI Align kernels on Inf1: OMP threading
leads to a 2 - 3x RoI Align latency reduction compared to the default
``roi_align_kernel.cpp`` kernel configuration.

.. code:: ipython3

    # Install python3.7-dev for pycocotools (a Detectron2 dependency)
    !sudo apt install python3.7-dev -y
    
    # Install Neuron packages
    !pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
    !pip uninstall -y torchvision
    !pip install --force-reinstall torch-neuron==1.11.0.* neuron-cc[tensorflow] "protobuf==3.20.1" ninja opencv-python
    
    # Change cuda to 10.2 for Detectron2
    !sudo rm /usr/local/cuda
    !sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
    
    # Install Torchvision 0.12.0 from source
    !git clone -b release/0.12 https://github.com/pytorch/vision.git
    
    # Update the RoI Align kernel to use OMP multithreading
    with open('vision/torchvision/csrc/ops/cpu/roi_align_kernel.cpp', 'r') as file:
        content = file.read()
    
    # Enable OMP Multithreading and set the number of threads to 4
    old = "// #pragma omp parallel for num_threads(32)"
    new = "#pragma omp parallel for num_threads(4)"
    content = content.replace(old, new)
    
    # Re-write the file
    with open('vision/torchvision/csrc/ops/cpu/roi_align_kernel.cpp', 'w') as file:
        file.write(content)
    
    # Build Torchvision with OMP threading
    !cd vision && CFLAGS="-fopenmp" python setup.py bdist_wheel
    %pip install vision/dist/*.whl
    
    # Install Detectron2 release v0.6
    !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

Compiling an R-CNN for Inf1
---------------------------

By default, R-CNN models are not compilable on Inf1 because they cannot
be traced with ``torch.jit.trace``, which is a requisite for inference
on Inf1. The following section demonstrates techniques for compiling a
Detectron2 R-CNN model for inference on Inf1.

Specifically, this section creates a standard Detectron2 R-CNN model
using a ResNet-101 backbone. It demonstrates how to use profiling to
identify the most compute intensive parts of the R-CNN that should be
compiled for accelerated inference on Inf1. It then explains how to
manually extract and compile the ResNet backbone (the dominant compute
component) and inject the compiled backbone back into the full model for
improved performance.

Create a Detectron2 R-CNN Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We create a Detectron2 R-CNN model using the
``COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml`` pretrained weights and
config file. We also download a sample image from the COCO dataset and
run an example inference.

.. code:: ipython3

    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    
    def get_model():
    
        # Configure the R-CNN model
        CONFIG_FILE = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        WEIGHTS_FILE = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(WEIGHTS_FILE)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = 'cpu'  # Send to CPU for Neuron Tracing
    
        # Create the R-CNN predictor wrapper
        predictor = DefaultPredictor(cfg)
        return predictor

.. code:: ipython3

    import os
    import urllib.request
    
    # Define a function to get a sample image
    def get_image():
        filename = 'input.jpg'
        if not os.path.exists(filename):
            url = "http://images.cocodataset.org/val2017/000000439715.jpg"
            urllib.request.urlretrieve(url, filename)
        return filename

.. code:: ipython3

    import time
    import cv2
    
    # Create an R-CNN model
    predictor = get_model()
    
    # Get a sample image from the COCO dataset
    image_filename = get_image()
    image = cv2.imread(image_filename)
    
    # Run inference and print inference latency
    start = time.time()
    outputs = predictor(image)
    print(f'Inference time: {(time.time() - start):0.3f} s')

Profile the Model
~~~~~~~~~~~~~~~~~

We can use the `PyTorch
Profiler <https://pytorch.org/docs/stable/profiler.html>`__ to identify
which operators contribute the most to the model’s runtime on CPU.
Ideally, we can compile these compute intensive operators onto Inf1 for
accelerated inference.

.. code:: ipython3

    import torch.autograd.profiler as profiler
    
    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            predictor(image)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

We see that convolution operators (``aten::convolution``) contribute the
most to the inference time. By compiling these convolution operators to
Inf1, we can improve performance of the R-CNN model. We can print the
R-CNN model architecture to see which layers contain the
``aten::convolution`` operators:

.. code:: ipython3

    print(predictor.model)

We observe that the ResNet FPN backbone
(`predictor.model.backbone <https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/backbone/fpn.py>`__ L17-L162)
contains the majority of convolution operators in the model. The RPN
(`predictor.model.proposal_generator <https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/proposal_generator/rpn.py>`__ L181-L533)
also contains a few convolutions. Based on this, we should try to
compile the ResNet backbone and RPN onto Inf1 to maximize performance.

Compiling the ResNet backbone to Inf1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this section we demonstrate how to compile the ResNet backbone to
Inf1 and use it for inference.

We “extract” the backbone by accessing it using
``predictor.model.backbone``. We compile the backbone using
``strict=False`` because the backbone outputs a dictionary. We use a
fixed input shape (``800 x 800``) for compilation. This will be the
input shape that all inputs will be resized to during inference. This
section also defines a basic preprocessing function (mostly derived from
the Detectron2 R-CNN
`DefaultPredictor <https://github.com/facebookresearch/detectron2/blob/45b3fcea6e76bf7a351e54e01c7d6e1a3a0100a5/detectron2/engine/defaults.py>`__
module L308-L318) that reshapes inputs to ``800 x 800``.

We also create a ``NeuronRCNN`` wrapper that we use to inject the
compiled backbone back into the model by dynamically replacing the
``predictor.model.backbone`` attribute with the compiled model.

.. code:: ipython3

    import torch
    import torch_neuron 
    
    example = torch.rand([1, 3, 800, 800])
    
    # Use `with torch.no_grad():` to avoid a jit tracing issue in the ResNet backbone
    with torch.no_grad():
        neuron_backbone = torch_neuron.trace(predictor.model.backbone, example, strict=False)
    
    backbone_filename = 'backbone.pt'
    torch.jit.save(neuron_backbone, backbone_filename)

.. code:: ipython3

    from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
    from torch.jit import ScriptModule

    class NeuronRCNN(torch.nn.Module):
        """
        Creates a `NeuronRCNN` wrapper that injects the compiled backbone into
        the R-CNN model. It also stores the `size_divisibility` attribute from
        the original backbone.
        """
    
        def __init__(self, model: GeneralizedRCNN, neuron_backbone: ScriptModule) -> None:
            super().__init__()
    
            # Keep track of the backbone variables
            size_divisibility = model.backbone.size_divisibility
    
            # Load and inject the compiled backbone
            model.backbone = neuron_backbone
    
            # Set backbone variables
            setattr(model.backbone, 'size_divisibility', size_divisibility)
    
            self.model = model
    
        def forward(self, x):
            return self.model(x)

.. code:: ipython3

    # Create the R-CNN with the compiled backbone
    neuron_rcnn = NeuronRCNN(predictor.model, neuron_backbone)
    neuron_rcnn.eval()

    # Print the R-CNN architecture to verify the backbone is now the
    # `neuron_backbone` (shows up as `RecursiveScriptModule`)
    print(neuron_rcnn)

.. code:: ipython3

    def preprocess(original_image, predictor):
        """
        A basic preprocessing function that sets the input height=800 and 
        input width=800. The function is derived from the preprocessing
        steps in the Detectron2 `DefaultPredictor` module.
        """
    
        height, width = original_image.shape[:2]
        resize_func = predictor.aug.get_transform(original_image)
        resize_func.new_h = 800 # Override height
        resize_func.new_w = 800 # Override width
        image = resize_func.apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return inputs

.. code:: ipython3

    # Get a resized input using the sample image
    inputs = preprocess(image, get_model())
    
    # Run inference and print inference latency
    start = time.time()
    for _ in range(10):
        outputs = neuron_rcnn([inputs])[0]
    print(f'Inference time: {((time.time() - start)/10):0.3f} s')

.. code:: ipython3

    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            neuron_rcnn([inputs])
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

By running the backbone on Inf1, the overall runtime is already
significantly improved. The count and runtime of ``aten::convolution``
operators is also decreased. We now see a ``neuron::forward_v2``
operator which is the compiled backbone.

Optimize the R-CNN model
------------------------

Compiling the RPN
~~~~~~~~~~~~~~~~~

Looking at the profiling, we see that there are still several
``aten::convolution``, ``aten::linear``, and ``aten::addmm`` operators
that significantly contribute to the model’s overall latency. By
inspecting the model architecture and code, we can determine that the
majority of these operators are contained in the RPN module
(`predictor.model.proposal_generator <https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/proposal_generator/rpn.py>`__ L181-L533).

To improve the model performance, we will extract the RPN Head and
compile it on Inf1 to increase the number of operators that are running
on Inf1. We only compile the RPN Head because the RPN Anchor Generator
contains objects that are not traceable with ``torch.jit.trace``.

The RPN Head contains five layers that run inference on multiple resized
inputs. In order to compile the RPN Head, we create a list of tensors
that contain the input (“``features``”) shapes that the RPN Head uses at
each layer. These tensor shapes can be determined by printing the input
shapes in the RPN Head ``forward`` function
(``predictor.model.proposal_generator.rpn_head.forward``).

We also create a new ``NeuronRCNN`` wrapper that injects both the
compiled backbone and RPN Head into the R-CNN model.

.. code:: ipython3

    import math
    
    input_shape = [1, 3, 800, 800] # Overall input shape at inference time
    
    # Create the list example of RPN inputs using the resizing logic from the RPN Head
    features = list()
    for i in [0, 1, 2, 3, 4]:
        ratio = 1 / (4 * 2**i)
        x_i_h = math.ceil(input_shape[2] * ratio)
        x_i_w = math.ceil(input_shape[3] * ratio)
        feature = torch.zeros(1, 256, x_i_h, x_i_w)
        features.append(feature)

.. code:: ipython3

    # Extract and compile the RPN Head
    neuron_rpn_head = torch_neuron.trace(predictor.model.proposal_generator.rpn_head, [features])
    rpn_head_filename = 'rpn_head.pt'
    torch.jit.save(neuron_rpn_head, rpn_head_filename)

.. code:: ipython3

    class NeuronRCNN(torch.nn.Module):
        """
        Creates a wrapper that injects the compiled backbone and RPN Head
        into the R-CNN model.
        """
    
        def __init__(self, model: GeneralizedRCNN, neuron_backbone: ScriptModule, neuron_rpn_head: ScriptModule) -> None:
            super().__init__()
    
            # Keep track of the backbone variables
            size_divisibility = model.backbone.size_divisibility
    
            # Inject the compiled backbone
            model.backbone = neuron_backbone
    
            # Set backbone variables
            setattr(model.backbone, 'size_divisibility', size_divisibility)
    
            # Inject the compiled RPN Head
            model.proposal_generator.rpn_head = neuron_rpn_head
    
            self.model = model
    
        def forward(self, x):
            return self.model(x)

.. code:: ipython3

    # Create the R-CNN with the compiled backbone and RPN Head
    predictor = get_model()
    neuron_rcnn = NeuronRCNN(predictor.model, neuron_backbone, neuron_rpn_head)
    neuron_rcnn.eval()

    # Print the R-CNN architecture to verify the compiled modules show up
    print(neuron_rcnn)

.. code:: ipython3

    # Run inference and print inference latency
    start = time.time()
    for _ in range(10):
        outputs = neuron_rcnn([inputs])[0]
    print(f'Inference time: {((time.time() - start)/10):0.3f} s')

.. code:: ipython3

    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            neuron_rcnn([inputs])
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

By running the compiled backbone and RPN Head on Inf1, the overall
runtime is improved. Once again, the number and runtime of
``aten::convolution`` operators is also decreased. We now see two
``neuron::forward_v2`` operators which correspond to the compiled
backbone and RPN Head.

Fusing the Backbone and RPN Head
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is typically preferable to compile fewer independent models
(“subgraphs”) on Inf1. Combining models and compiling them as a single
subgraph enables the Neuron compiler to perform additional optimizations
and reduces the I/O data transfer between CPU and NeuronCores between
each subgraph.

In this section, we “fuse” the ResNet backbone and RPN Head into a
single model that we compile on Inf1. We create the
``NeuronFusedBackboneRPNHead`` wrapper to create a compilable model that
contains both the ResNet backbone
(`predictor.model.backbone <https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/backbone/fpn.py>`__ L17-L162)
and RPN Head
(`predictor.model.proposal_generator <https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/proposal_generator/rpn.py>`__ L181-L533).
We also output the ``features`` because it is used downstream by the RoI
Heads. We compile this ``NeuronFusedBackboneRPNHead`` wrapper as
``neuron_backbone_rpn``. We then create a separate ``BackboneRPN``
wrapper that we use to inject the ``neuron_backbone_rpn`` in place of
the original backbone and RPN Head. We also copy the remainder of the
RPN ``forward`` code
(`predictor.model.proposal_generator.forward <https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/proposal_generator/rpn.py>`__ L431-L480)
to create a “fused” backbone + RPN module. Lastly, we re-write the
``NeuronRCNN`` wrapper to use the fused backbone + RPN module. The
``NeuronRCNN`` wrapper also uses the ``predictor.model`` ``forward``
code to re-write the rest of the R-CNN model forward function.

.. code:: ipython3

    class NeuronFusedBackboneRPNHead(torch.nn.Module):
        """
        Wrapper to compile the fused ResNet backbone and RPN Head.
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
            self.backbone = model.backbone
            self.rpn_head = model.proposal_generator.rpn_head
            self.in_features = model.proposal_generator.in_features
    
        def forward(self, x):
            features = self.backbone(x)
            features_ = [features[f] for f in self.in_features]
            return self.rpn_head(features_), features

.. code:: ipython3

    # Create the wrapper with the combined backbone and RPN Head
    predictor = get_model()
    backbone_rpn_wrapper = NeuronFusedBackboneRPNHead(predictor.model)
    backbone_rpn_wrapper.eval()
    
    # Compile the wrapper
    example = torch.rand([1, 3, 800, 800])
    
    with torch.no_grad():
        neuron_backbone_rpn_head = torch_neuron.trace(
            backbone_rpn_wrapper, example, strict=False)
    
    backbone_rpn_filename = 'backbone_rpn.pt'
    torch.jit.save(neuron_backbone_rpn_head, backbone_rpn_filename)

.. code:: ipython3

    class BackboneRPN(torch.nn.Module):
        """
        Wrapper that uses the compiled `neuron_backbone_rpn` instead
        of the original backbone and RPN Head. We copy the remainder
        of the RPN `forward` code (`predictor.model.proposal_generator.forward`)
        to create a "fused" backbone + RPN module.
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
            self.backbone_rpn_head = NeuronFusedBackboneRPNHead(model)
            self._rpn = model.proposal_generator
            self.in_features = model.proposal_generator.in_features
    
        def forward(self, images):
            preds, features = self.backbone_rpn_head(images.tensor)
            features_ = [features[f] for f in self.in_features]
            pred_objectness_logits, pred_anchor_deltas = preds
            anchors = self._rpn.anchor_generator(features_)
    
            # Transpose the Hi*Wi*A dimension to the middle:
            pred_objectness_logits = [
                # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                score.permute(0, 2, 3, 1).flatten(1)
                for score in pred_objectness_logits
            ]
            pred_anchor_deltas = [
                # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
                x.view(x.shape[0], -1, self._rpn.anchor_generator.box_dim,
                       x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
                for x in pred_anchor_deltas
            ]
    
            proposals = self._rpn.predict_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )
            return proposals, features

.. code:: ipython3

    class NeuronRCNN(torch.nn.Module):
        """
        Wrapper that uses the fused backbone + RPN module and re-writes
        the rest of the R-CNN `model` `forward` function.
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
    
            # Use the fused Backbone + RPN
            self.backbone_rpn = BackboneRPN(model)
    
            self.roi_heads = model.roi_heads
    
            self.preprocess_image = model.preprocess_image
            self._postprocess = model._postprocess
    
        def forward(self, batched_inputs):
            images = self.preprocess_image(batched_inputs)
            proposals, features = self.backbone_rpn(images)
            results, _ = self.roi_heads(images, features, proposals, None)
            return self._postprocess(results, batched_inputs, images.image_sizes)

.. code:: ipython3

    # Create the new NeuronRCNN wrapper with the combined backbone and RPN Head
    predictor = get_model()
    neuron_rcnn = NeuronRCNN(predictor.model)
    neuron_rcnn.eval()

    # Inject the Neuron compiled models
    neuron_rcnn.backbone_rpn.backbone_rpn_head = neuron_backbone_rpn_head

    # Print the R-CNN architecture to verify the compiled modules show up
    print(neuron_rcnn)

.. code:: ipython3

    # Run inference and print inference latency
    start = time.time()
    for _ in range(10):
        outputs = neuron_rcnn([inputs])[0]
    print(f'Inference time: {((time.time() - start)/10):0.3f} s')

.. code:: ipython3

    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            neuron_rcnn([inputs])
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

By running the fused backbone + RPN Head on Inf1, the overall runtime is
improved again. We now see a single ``neuron::forward_v2`` operator with
a lower runtime than the previous combined runtime of the two separate
``neuron::forward_v2`` operators.

Compiling the RoI Heads
~~~~~~~~~~~~~~~~~~~~~~~

In this section, we extract and compile part of RoI Heads module
(`predictor.model.roi_heads <https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/modeling/roi_heads/roi_heads.py>`__ L530-L778).
This will run most of the remaining ``aten::linear`` and ``aten::addmm``
operators on Inf1. We cannot extract the entire RoI Heads module because
it contains unsupported operators. Thus, we create a
``NeuronBoxHeadBoxPredictor`` wrapper that extracts specific parts of
the ``roi_heads`` for compilation. The example input for compilation is
the shape of the input into the ``self.roi_heads.box_head.forward``
function. We write another wrapper, ``ROIHead`` that combines the
compiled ``roi_heads`` into the rest of the RoI module. The
``_forward_box`` and ``forward`` functions are from the
``predictor.model.roi_heads`` module. We re-write the ``NeuronRCNN``
wrapper to use the optimized RoI Heads wrapper as well as the fused
backbone + RPN module.

.. code:: ipython3

    class NeuronBoxHeadBoxPredictor(torch.nn.Module):
        """
        Wrapper that extracts the RoI Box Head and Box Predictor
        for compilation.
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
            self.roi_heads = model.roi_heads
    
        def forward(self, box_features):
            box_features = self.roi_heads.box_head(box_features)
            predictions = self.roi_heads.box_predictor(box_features)
            return predictions

.. code:: ipython3

    # Create the NeuronBoxHeadBoxPredictor wrapper
    predictor = get_model()
    box_head_predictor = NeuronBoxHeadBoxPredictor(predictor.model)
    box_head_predictor.eval()

    # Compile the wrapper
    example = torch.rand([1000, 256, 7, 7])
    neuron_box_head_predictor = torch_neuron.trace(box_head_predictor, example)

    roi_head_filename = 'box_head_predictor.pt'
    torch.jit.save(neuron_box_head_predictor, roi_head_filename)

.. code:: ipython3

    class ROIHead(torch.nn.Module):
        """
        Wrapper that combines the compiled `roi_heads` into the
        rest of the RoI module. The `_forward_box` and `forward`
        functions are from the `predictor.model.roi_heads` module.
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
            self.roi_heads = model.roi_heads
            self.neuron_box_head_predictor = NeuronBoxHeadBoxPredictor(model)
    
        def _forward_box(self, features, proposals):
            features = [features[f] for f in self.roi_heads.box_in_features]
            box_features = self.roi_heads.box_pooler(
                features, [x.proposal_boxes for x in proposals])
            predictions = self.neuron_box_head_predictor(box_features)
            pred_instances, _ = self.roi_heads.box_predictor.inference(
                predictions, proposals)
            return pred_instances
    
        def forward(self, images, features, proposals, targets=None):
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.roi_heads.forward_with_given_boxes(
                features, pred_instances)
            return pred_instances, {}

.. code:: ipython3

    class NeuronRCNN(torch.nn.Module):
        """
        Wrapper that uses the fused backbone + RPN module and the optimized RoI
        Heads wrapper
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
    
            # Create fused Backbone + RPN
            self.backbone_rpn = BackboneRPN(model)
    
            # Create Neuron RoI Head
            self.roi_heads = ROIHead(model)
    
            # Define pre and post-processing functions
            self.preprocess_image = model.preprocess_image
            self._postprocess = model._postprocess
    
        def forward(self, batched_inputs):
            images = self.preprocess_image(batched_inputs)
            proposals, features = self.backbone_rpn(images)
            results, _ = self.roi_heads(images, features, proposals, None)
            return self._postprocess(results, batched_inputs, images.image_sizes)

.. code:: ipython3

    # Initialize an R-CNN on CPU
    predictor = get_model()

    # Create the Neuron R-CNN on CPU
    neuron_rcnn = NeuronRCNN(predictor.model)
    neuron_rcnn.eval()

    # Inject the Neuron compiled models
    neuron_rcnn.backbone_rpn.backbone_rpn_head = neuron_backbone_rpn_head
    neuron_rcnn.roi_heads.neuron_box_head_predictor = neuron_box_head_predictor

.. code:: ipython3

    # Run inference and print inference latency
    start = time.time()
    for _ in range(10):
        outputs = neuron_rcnn([inputs])[0]
    print(f'CPU Inference time: {((time.time() - start)/10):0.3f} s')

.. code:: ipython3

    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            neuron_rcnn([inputs])
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

Although the overall latency didn’t change significantly, running more
of the model on Inf1 instead of CPU will free up CPU resources when
multiple models are running in parallel.

End-to-end Compilation and Inference
------------------------------------

In this section we provide standalone code that compiles and runs an
optimized Detectron2 R-CNN on Inf1. Most of the code in this section is
from the previous sections in this application note and it’s
consolidated here for easy deployment. This section has the following
main componennts:

1. Preprocessing and compilation functions
2. Wrappers that extract the R-CNN ResNet backbone, RPN Head, and RoI
   Head for compilation on Inf1.
3. A ``NeuronRCNN`` wrapper that creates an optimized end-to-end
   Detectron2 R-CNN model for inference on Inf1
4. Benchmarking code that runs parallelized inference for optimized
   throughput on Inf1

Benchmarking
~~~~~~~~~~~~

In the benchmarkinng section, we load multiple optimized RCNN models and
run them in parallel to maximize throughput.

We use the beta NeuronCore placement API,
``torch_neuron.experimental.neuron_cores_context()``, to ensure all
compiled models in an optimized RCNN model are loaded onto the same
NeuronCore. Please note that the functionality and API of
``torch_neuron.experimental.neuron_cores_context()`` might change in
future releases.

We define a simple benchmark function that loads four optimized RCNN
models onto four separate NeuronCores, runs multithreaded inference, and
calculates the corresponding latency and throughput. We benchmark
various numbers of loaded models to show the impact of parallelism.

We observe that throughput increases (at the cost of latency) when more
models are run in parallel on Inf1. Increasing the number of worker
threads also improves throughput.

Other improvements
~~~~~~~~~~~~~~~~~~

There are many additional optimizations that can be applied to RCNN
models on Inf1 depending on the application:

For latency sensitive applications:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Each of the five layers in the RPN head can be parallelized to
   decrease the overall latency.
-  The number of OMP Threads can be increased in the ROI Align kernel.
   Both of these optimizations will improve latency at the cost of
   decreasing throughput.

For throughput sensitive applications:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  The input batch size can be increased to improve the NeuronCore
   utilization.

.. code:: ipython3

    import time
    import os
    import urllib.request
    from typing import Any, Union, Callable
    
    import cv2
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    
    import torch
    import torch_neuron
    
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
    
    
    # -----------------------------------------------------------------------------
    # Helper functions
    # -----------------------------------------------------------------------------
    
    def get_model():
    
        # Configure the R-CNN model
        CONFIG_FILE = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        WEIGHTS_FILE = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(WEIGHTS_FILE)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = 'cpu'  # Send to CPU for Neuron Tracing
    
        # Create the R-CNN predictor wrapper
        predictor = DefaultPredictor(cfg)
        return predictor
    
    
    def get_image():
    
        # Get a sample image
        filename = 'input.jpg'
        if not os.path.exists(filename):
            url = "http://images.cocodataset.org/val2017/000000439715.jpg"
            urllib.request.urlretrieve(url, filename)
        return filename
    
    
    def preprocess(original_image, predictor):
        """
        A basic preprocessing function that sets the input height=800 and 
        input width=800. The function is derived from the preprocessing
        steps in the Detectron2 `DefaultPredictor` module.
        """
    
        height, width = original_image.shape[:2]
        resize_func = predictor.aug.get_transform(original_image)
        resize_func.new_h = 800 # Override height
        resize_func.new_w = 800 # Override width
        image = resize_func.apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        return inputs
    
    
    # -----------------------------------------------------------------------------
    # Neuron modules
    # -----------------------------------------------------------------------------
    
    class NeuronFusedBackboneRPNHead(torch.nn.Module):
        """
        Wrapper to compile the fused ResNet backbone and RPN Head.
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
            self.backbone = model.backbone
            self.rpn_head = model.proposal_generator.rpn_head
            self.in_features = model.proposal_generator.in_features
    
        def forward(self, x):
            features = self.backbone(x)
            features_ = [features[f] for f in self.in_features]
            return self.rpn_head(features_), features
    
    
    class BackboneRPN(torch.nn.Module):
        """
        Wrapper that uses the compiled `neuron_backbone_rpn` instead
        of the original backbone and RPN Head. We copy the remainder
        of the RPN `forward` code (`predictor.model.proposal_generator.forward`)
        to create a "fused" backbone + RPN module.
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
            self.backbone_rpn_head = NeuronFusedBackboneRPNHead(model)
            self._rpn = model.proposal_generator
            self.in_features = model.proposal_generator.in_features
    
        def forward(self, images):
            preds, features = self.backbone_rpn_head(images.tensor)
            features_ = [features[f] for f in self.in_features]
            pred_objectness_logits, pred_anchor_deltas = preds
            anchors = self._rpn.anchor_generator(features_)
    
            # Transpose the Hi*Wi*A dimension to the middle:
            pred_objectness_logits = [
                # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                score.permute(0, 2, 3, 1).flatten(1)
                for score in pred_objectness_logits
            ]
            pred_anchor_deltas = [
                # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
                x.view(x.shape[0], -1, self._rpn.anchor_generator.box_dim,
                       x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
                for x in pred_anchor_deltas
            ]
    
            proposals = self._rpn.predict_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )
            return proposals, features
    
    
    class NeuronBoxHeadBoxPredictor(torch.nn.Module):
        """
        Wrapper that extracts the RoI Box Head and Box Predictor
        for compilation.
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
            self.roi_heads = model.roi_heads
    
        def forward(self, box_features):
            box_features = self.roi_heads.box_head(box_features)
            predictions = self.roi_heads.box_predictor(box_features)
            return predictions
    
    
    class ROIHead(torch.nn.Module):
        """
        Wrapper that combines the compiled `roi_heads` into the
        rest of the RoI module. The `_forward_box` and `forward`
        functions are from the `predictor.model.roi_heads` module.
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
            self.roi_heads = model.roi_heads
            self.neuron_box_head_predictor = NeuronBoxHeadBoxPredictor(model)
    
        def _forward_box(self, features, proposals):
            features = [features[f] for f in self.roi_heads.box_in_features]
            box_features = self.roi_heads.box_pooler(
                features, [x.proposal_boxes for x in proposals])
            predictions = self.neuron_box_head_predictor(box_features)
            pred_instances, _ = self.roi_heads.box_predictor.inference(
                predictions, proposals)
            return pred_instances
    
        def forward(self, images, features, proposals, targets=None):
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.roi_heads.forward_with_given_boxes(
                features, pred_instances)
            return pred_instances, {}
    
    
    class NeuronRCNN(torch.nn.Module):
        """
        Wrapper that uses the fused backbone + RPN module and the optimized RoI
        Heads wrapper
        """
    
        def __init__(self, model: GeneralizedRCNN) -> None:
            super().__init__()
    
            # Create fused Backbone + RPN
            self.backbone_rpn = BackboneRPN(model)
    
            # Create Neuron RoI Head
            self.roi_heads = ROIHead(model)
    
            # Define pre and post-processing functions
            self.preprocess_image = model.preprocess_image
            self._postprocess = model._postprocess
    
        def forward(self, batched_inputs):
            images = self.preprocess_image(batched_inputs)
            proposals, features = self.backbone_rpn(images)
            results, _ = self.roi_heads(images, features, proposals, None)
            return self._postprocess(results, batched_inputs, images.image_sizes)
    
    
    # -----------------------------------------------------------------------------
    # Compilation functions
    # -----------------------------------------------------------------------------
    
    def compile(
        model: Union[Callable, torch.nn.Module],
        example_inputs: Any,
        filename: str,
        **kwargs
    ) -> torch.nn.Module:
        """
        Compiles the model for Inf1 if it doesn't already exist and saves it as the provided filename. 
        
        model: A module or function which defines a torch model or computation.
        example_inputs: An example set of inputs which will be passed to the
            `model` during compilation.
        filename: Name of the compiled model
        kwargs: Extra `torch_neuron.trace` kwargs
        """
    
        if not os.path.exists(filename):
            with torch.no_grad():
                compiled_model = torch_neuron.trace(model, example_inputs, **kwargs)
            torch.jit.save(compiled_model, filename)
    
    
    # -----------------------------------------------------------------------------
    # Benchmarking function
    # -----------------------------------------------------------------------------
    
    def benchmark(backbone_rpn_filename, roi_head_filename, inputs, 
                  n_models=4, batch_size=1, n_threads=4, iterations=200):
        """
        A simple benchmarking function that loads `n_models` optimized
        models onto separate NeuronCores, runs multithreaded inference,
        and calculates the corresponding latency and throughput.
        """
    
        # Load models
        models = list()
        for i in range(n_models):
            with torch_neuron.experimental.neuron_cores_context(i):
                # Create the RCNN with the fused backbone + RPN Head and compiled RoI Heads
                # Initialize an R-CNN on CPU
                predictor = get_model()

                # Create the Neuron R-CNN on CPU
                neuron_rcnn = NeuronRCNN(predictor.model)
                neuron_rcnn.eval()

                # Inject the Neuron compiled models
                neuron_rcnn.backbone_rpn.backbone_rpn_head = torch.jit.load(backbone_rpn_filename)
                neuron_rcnn.roi_heads.neuron_box_head_predictor = torch.jit.load(roi_head_filename)

                models.append(neuron_rcnn)
    
        # Warmup
        for _ in range(8):
            for model in models:
                model([inputs])
    
        latencies = []
    
        # Thread task
        def task(i):
            start = time.time()
            models[i]([inputs])
            finish = time.time()
            latencies.append((finish - start) * 1000)
    
        begin = time.time()
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for i in range(iterations):
                pool.submit(task, i % n_models)
        end = time.time()
    
        # Compute metrics
        boundaries = [50, 95, 99]
        names = [f'Latency P{i} (ms)' for i in boundaries]
        percentiles = np.percentile(latencies, boundaries)
        duration = end - begin
    
        # Display metrics
        results = {
            'Samples': iterations,
            'Batch Size': batch_size,
            'Models': n_models,
            'Threads': n_threads,
            'Duration (s)': end - begin,
            'Throughput (inf/s)': (batch_size * iterations) / duration,
            **dict(zip(names, percentiles)),
        }
    
        print('-' * 80)
        pad = max(map(len, results))
        for key, value in results.items():
            if isinstance(value, float):
                print(f'{key + ":" :<{pad + 1}} {value:0.3f}')
            else:
                print(f'{key + ":" :<{pad + 1}} {value}')
        print()
    
    
    if __name__ == "__main__":
    
        # Create and compile the combined backbone and RPN Head wrapper
        backbone_rpn_filename = 'backbone_rpn.pt'
        predictor = get_model()
        backbone_rpn_wrapper = NeuronFusedBackboneRPNHead(predictor.model)
        backbone_rpn_wrapper.eval()
        example = torch.rand([1, 3, 800, 800])
        compile(backbone_rpn_wrapper, example, backbone_rpn_filename, strict=False)

        # Create and compile the RoI Head wrapper
        roi_head_filename = 'box_head_predictor.pt'
        predictor = get_model()
        box_head_predictor = NeuronBoxHeadBoxPredictor(predictor.model)
        box_head_predictor.eval()
        example = torch.rand([1000, 256, 7, 7])
        compile(box_head_predictor, example, roi_head_filename)

        # Download a sample image from the COCO dataset and read it
        image_filename = get_image()
        image = cv2.imread(image_filename)
        inputs = preprocess(image, get_model())
    
        # Benchmark the Neuron R-CNN model for various numbers of loaded models
        benchmark(backbone_rpn_filename, roi_head_filename, inputs, n_models=1, n_threads=1)
        benchmark(backbone_rpn_filename, roi_head_filename, inputs, n_models=1, n_threads=2)
        benchmark(backbone_rpn_filename, roi_head_filename, inputs, n_models=2, n_threads=2)
        benchmark(backbone_rpn_filename, roi_head_filename, inputs, n_models=2, n_threads=4)
        benchmark(backbone_rpn_filename, roi_head_filename, inputs, n_models=4, n_threads=4)
        benchmark(backbone_rpn_filename, roi_head_filename, inputs, n_models=4, n_threads=8)
