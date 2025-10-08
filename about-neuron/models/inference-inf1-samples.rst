.. _model_samples_inference_inf1:

Inference Samples/Tutorials (Inf1)
==================================

.. contents:: Table of contents
   :local:
   :depth: 1

   
.. _encoder_model_samples_inference_inf1:
 
Encoders 
--------

.. list-table::
   :widths: 20 15 45 
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - Model
     - Frameworks/Libraries
     - Samples and Tutorials

   * - bert-base-cased-finetuned-mrpc
     - torch-neuron
     - * HuggingFace pretrained BERT tutorial :ref:`[html] </src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert.ipynb>` :pytorch-neuron-src:`[notebook] <bert_tutorial/tutorial_pretrained_bert.ipynb>`
       * `BertBaseCased Inference on Inf1 instances <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/bertbasecased/BertBaseCased.ipynb>`_
       * Bert TorchServe tutorial :ref:`[html] <pytorch-tutorials-torchserve>`
       * Bring your own HuggingFace pretrained BERT container to Sagemaker Tutorial :ref:`[html] </src/examples/pytorch/byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb>` :pytorch-neuron-src:`[notebook] <byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb>`

   * - bert-base-uncased
     - torch-neuron
     - * NeuronCore Pipeline tutorial :ref:`[html] </src/examples/pytorch/pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb>` :pytorch-neuron-src:`[notebook] <pipeline_tutorial/neuroncore_pipeline_pytorch.ipynb>`

   * - bert-large-uncased
     - torch-neuron
     - * `BertLargeUncased Inference on Inf1 instances <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/bertlargeuncased/BertLargeUncased.ipynb>`_
   
   * - roberta-base
     - torch-neuron
     - * `Roberta-Base inference on Inf1 instances <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/robertabase/RobertaBase.ipynb>`_

   * - distilbert-base-uncased-finetuned-sst-2-english
     - tensorflow-neuron 
     - * Tensorflow 2.x - HuggingFace Pipelines distilBERT with Tensorflow2 Neuron :ref:`[html] </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>` :github:`[notebook] </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>`
    
   * - gluon bert
     - mxnet-neuron 
     - * MXNet 1.8: Using data parallel mode tutorial :ref:`[html] </src/examples/mxnet/data_parallel/data_parallel_tutorial.ipynb>` :mxnet-neuron-src:`[notebook] <data_parallel/data_parallel_tutorial.ipynb>`



.. _vision_transformer_model_samples_inference_inf1:

Vision Transformers  
-------------------

.. list-table::
   :widths: 20 15 45 
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   
   * - Model
     - Frameworks/Libraries
     - Samples and Tutorials

   * - ssd
     - torch-neuron
     - * `Inference of SSD model on inf1 instances <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/ssd/SSD300VGG16.ipynb>`_
 

   * - TrOCR
     - torch-neuron
     - * `TrOCR inference on Inf1 instances <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/trocr/TrOCR.ipynb>`_

    
   * - vgg
     - torch-neuron
     - * `VGG inference on Inf1 instances <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/vgg/VGG.ipynb>`_


   * - google/vit-base-patch16-224
     - torch-neuron
     - * `ViT model inference on Inf1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/vit/ViT.ipynb>`_



.. _cnn_model_samples_inference_inf1:

Convolutional Neural Networks(CNN)
----------------------------------


.. list-table::
   :widths: 20 15 45 
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - Model
     - Frameworks/Libraries
     - Samples and Tutorials

   * - EfficientNet
     - torch-neuron
     - * `EfficientNet model inference on Inf1 instances <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/efficientnet/EfficientNet.ipynb>`_

   * - GFL (MMDetection)
     - torch-neuron
     - * `GFL (MMDetection) inference on Inf1 instances <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/gfl_mmdet/GFL.ipynb>`_

   * - HRNet
     - torch-neuron
     - * `HRNET - Pose Estimation <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/hrnet/HRnet.ipynb>`_

   * - MarianMT
     - torch-neuron
     - * HuggingFace MarianMT tutorial :ref:`[html] </src/examples/pytorch/transformers-marianmt.ipynb>` :pytorch-neuron-src:`[notebook] <transformers-marianmt.ipynb>`
       * `Inference of Pre-trained MarianMT model on Inf1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/marianmt/MarianMT.ipynb>`_

   * - Detectron2 R-CNN 
     - torch-neuron
     - * `R-CNN inference on Inf1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/rcnn/Rcnn.ipynb>`_

   * - resnet
     - torch-neuron
     - * `Inference of Pre-trained Resnet model (18,34,50,101,152) on Inf1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/resnet/Resnet.ipynb>`_
       * ResNet-50 tutorial :ref:`[html] </src/examples/pytorch/resnet50.ipynb>` :pytorch-neuron-src:`[notebook] <resnet50.ipynb>`

   * - resnet
     - tensorflow-neuron
     - * Tensorflow 2.x - Using NEURON_RT_VISIBLE_CORES with TensorFlow Serving :ref:`[html] </src/examples/tensorflow/tensorflow_serving_tutorial.rst>`
   
   * - resnet
     - mxnet-neuron
     - * ResNet-50 tutorial :ref:`[html] </src/examples/mxnet/resnet50/resnet50.ipynb>` :mxnet-neuron-src:`[notebook] <resnet50/resnet50.ipynb>`
       * Getting started with Gluon tutorial :ref:`[html] </src/examples/mxnet/mxnet-gluon-tutorial.ipynb>` :github:`[notebook] </src/examples/mxnet/mxnet-gluon-tutorial.ipynb>`
       * NeuronCore Groups tutorial :ref:`[html] </src/examples/mxnet/resnet50_neuroncore_groups.ipynb>` :mxnet-neuron-src:`[notebook] <resnet50_neuroncore_groups.ipynb>`
    

   * - Resnext
     - torch-neuron
     - * `Inference of Resnext model on Inf1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/resnext/Resnext.ipynb>`_


   * - Yolov4
     - torch-neuron 
     - * PyTorch YOLOv4 tutorial :ref:`[html] </src/examples/pytorch/yolo_v4.ipynb>` :pytorch-neuron-src:`[notebook] <yolo_v4.ipynb>`

   * - Yolov5
     - torch-neuron
     - * `Inference of Yolov5 on Inf1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/yolov5/Yolov5.ipynb>`_


   * - Yolov6
     - torch-neuron 
     - * `Inference of Yolov6 on Inf1 instances <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/yolov6/Yolov6.ipynb>`_


   * - Yolov7
     - torch-neuron
     - * `Inference of Yolov7 model on Inf1 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuron/inference/yolov7>`_

   * - Yolof
     - torch-neuron
     - * `Inference of Yolof model on Inf1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuron/inference/yolof_detectron2/YoloF.ipynb>`_

   * - fairseq
     - torch-neuron
     - * `Inference of fairseq model on Inf1 <https://github.com/aws-neuron/aws-neuron-samples-staging/tree/master/torch-neuron/inference/fairseq>`_

   * - unet
     - tensorflow-neuron
     - * `Unet - Tensorflow 2.x tutorial <https://github.com/aws-neuron/aws-neuron-samples/blob/master/tensorflow-neuron/inference/unet>`_



.. _vision_model_samples_inference_inf1:

Vision
------

.. list-table::
   :widths: 20 15 45 
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - Model
     - Frameworks/Libraries
     - Samples and Tutorials

   * - craft-pytorch
     - torch-neuron
     - * `CRAFT model inference on Inf1 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuron/inference/craft>`_

   






 











