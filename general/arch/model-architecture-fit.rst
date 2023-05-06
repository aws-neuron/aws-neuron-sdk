.. _model_architecture_fit:

Model Architecture Fit Guidelines
=================================

.. contents:: Table of contents
   :local:
   :depth: 2

Introduction
$$$$$$$$$$$$

AWS Neuron SDK enables you to train and deploy a wide range of deep learning models on :ref:`EC2 Inf1 <aws-inf1-arch>`, :ref:`EC2 inf2 <aws-inf2-arch>` and :ref:`EC2 Trn1/Trn1n <aws-trn1-arch>` instances , which are powered by :ref:`Inferentia <inferentia-arch>`, :ref:`Inferentia2 <inferentia2-arch>` and :ref:`Trainium <trainium-arch>` devices. The below table provides details of the NeuronDevices and NeuronCores enabling each instance:


.. list-table::
    :widths: auto
    :header-rows: 1
    :stub-columns: 1    
    :align: left
    

    *   - Instance
        - NeuronDevices
        - NeuronCores
        - # NeuronCores in a NeuronDevice


    *   - :ref:`EC2 Trn1 <aws-trn1-arch>`
        - 16 x :ref:`Trainium <trainium-arch>`
        - 32 x :ref:`NeuronCore-v2 <neuroncores-v2-arch>`
        - 2

    *   - :ref:`EC2 Trn1n <aws-trn1-arch>`
        - 16 x :ref:`Trainium <trainium-arch>`
        - 32 x :ref:`NeuronCore-v2 <neuroncores-v2-arch>`
        - 2

    *   - :ref:`EC2 inf2 <aws-inf2-arch>`
        - 12 x :ref:`Inferentia2 <inferentia2-arch>`
        - 24 x :ref:`NeuronCore-v2 <neuroncores-v2-arch>`
        - 2

    *   - :ref:`EC2 Inf1 <aws-inf1-arch>`
        - 16 x :ref:`Inferentia <inferentia-arch>`
        - 64 x :ref:`NeuronCore-v1 <neuroncores-v1-arch>`
        - 4


This document describes what types of deep learning model architectures are a good fit for  :ref:`Inferentia <inferentia-arch>`, :ref:`Inferentia2 <inferentia2-arch>` and :ref:`Trainium <trainium-arch>` powered instances. 



Model Support Overview
$$$$$$$$$$$$$$$$$$$$$$

.. _model-architecture-fit-neuroncore-v2:

AWS Trainium and AWS Inferentia2 (NeuronCore-v2)
------------------------------------------------

*Last update* - 05/05/2023

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   

   *  - Model Family/  
        Neural Network Architecture
      - Category
      - Hardware Architecture
      - Training with PyTorch Neuron (``torch-neuronx``)
      - Inference with PyTorch Neuron (``torch-neuronx``)
      - Inference with TensorFlow Neuron (``tensorflow-neuronx``)


   *  - Transformer Encoders
      - NLP
      - Good Fit
      - Supported
      - Supported
      - Supported

   *  - Transformer Decoders
      - NLP
      - Good Fit
      - Supported
      - Supported
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Transformer Encoder-Decoder (Sequence-to-sequence)
      - NLP
      - Good Fit
      - Supported
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - LSTMs
      - NLP and Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Vision Transformer
      - Computer Vision
      - Good Fit
      - Supported
      - Supported
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Diffusion models
      - Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - Supported
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Convolutional Neural Network (CNN) models
      - Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - Supported
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - R-CNNs
      - Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`

.. note::

   Supported means that at least a single model of the model family or the neural-network architecture already enabled. 

.. _model-architecture-fit-neuroncore-v1:

AWS Inferentia (NeuronCore v1)
------------------------------

*Last update* - 05/05/2023

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   

   *  - Model Family/  
        Neural Network Architecture
   
      - Category
      - Hardware Architecture
      - PyTorch Neuron (``torch-neuron``)
      - TensorFlow Neuron (``tensorflow-neuron (TF 1.x)``)
      - TensorFlow Neuron (``tensorflow-neuron (TF 2.x)``)

   *  - Transformer Encoders
      - NLP
      - Good Fit
      - Supported
      - Supported
      - Supported

   *  - Transformer Decoders
      - NLP
      - Not a Good Fit
      - NA
      - NA
      - NA

   *  - Transformer Encoder-Decoder (Sequence-to-sequence)
      - NLP
      - Not a Good Fit
      - NA
      - NA
      - NA

   *  - LSTMs
      - NLP and Computer Vision
      - Good Fit
      - Supported
      - NA
      - NA

   *  - Vision Transformer
      - Computer Vision
      - Good Fit
      - Supported
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Diffusion models
      - Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - NA
      - NA

   *  - Convolutional Neural Network (CNN) models
      - Computer Vision
      - Good Fit
      - Supported
      - Supported
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - R-CNNs
      - Computer Vision
      - Supported with limitations
      - Supported with limitations
      - NA
      - NA

.. note::

   Supported means that at least a single model of the model family or the neural-network architecture already enabled. 


Clarifications on Inferentia (1st generation) Model Architecture
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Natural Language Processing (NLP) Models with Transformer
----------------------------------------------------------

Transformer Encoders
~~~~~~~~~~~~~~~~~~~~~

Autoencoding models use only the encoder part of the Transformer architecture. Representatives of this family include models like **BERT, distilBERT, XLM-BERT, Roberta, BioBert**, etc.  Since the encoding process in these models can be parallelized, you can expect these models to run well both on Inferentia and Trainium. 

- **Architecture Fit** - Autoencoding models are a good fit for Inferentia.
- **Neuron Support** - Neuron SDK support running Autoencoding models for inference on Inferentia. Please see :ref:`benchmark results <appnote-performance-benchmark>` of these models. To get started with NLP models you can refer to Neuron :ref:`PyTorch <pytorch-nlp>`, :ref:`TensorFlow <tensorflow-nlp>` and :ref:`MXNet <mxnet-nlp>` NLP tutorials.

Decoder models, or autoregressive models with Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autoregressive models keep only the decoder part of the Transformer architecture. Representatives of this family include models like **GPT-3, GPT-2**, etc.

- **Architecture Fit** - Autoregressive models are not a good fit for Inferentia. Usually the decoder part in these models is the most significant performance bottleneck since it must be executed once per output token, causing frequent access to the memory. Due to this these models typically experience the best performance only when the decoder maximum sequence length is short (e.g., 128).
- **Neuron Support** - Neuron SDK does not support Autoregressive models inference on Inferentia.

Encoder-decoder models, or sequence-to-sequence models with Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequence-to-sequence models use both of encoder and decoder of the Transformer architecture. Representatives of this family include models like **T5, Bart, Marian MT**, etc.

- **Architecture Fit** - Sequence-to-sequence models are not a good fit for Inferentia. Like decoder models explained above, usually the decoder part in these sequence-to-sequence models is the most significant performance bottleneck since it must be executed once per output token, causing frequent access to the memory. Due to this, even when you enabled the models to run on Inferentia with wrapping the decoder part, these models typically experience the best performance only when the decoder maximum sequence length is short (e.g., 128).
- **Neuron Support** - Neuron SDK does not support sequence-to-sequence models inference on Inferentia out of the box. However, you can run a model with defining wrappers around the encoder and decoder portions of it. For example, please refer to :ref:`MarianMT tutorial </src/examples/pytorch/Transformer-marianmt.ipynb>` on Inferentia for more details. 

Computer Vision Models
----------------------

Convolutional Neural Network (CNN) based models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CNN based models are used for applications in image classification and object detection. Representatives of this family include models like **ResNet, ResNext, VGG, YOLO, SSD**, etc.

- **Architecture Fit** - CNN based models are a good fit for Inferentia.
- **Neuron Support** - Neuron SDK supports CNN based models inference on Inferentia. Please see the :ref:`benchmark results <appnote-performance-benchmark>` of these models. To get started with these models you can refer to Neuron :ref:`PyTorch <pytorch-computervision>`, :ref:`TensorFlow <tensorflow-computervision>` and :ref:`MXNet <mxnet-computervision>` tutorials.

Region-based CNN (R-CNN) models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Region-based CNNs (R-CNNs) models are commonly used for object detection and image segmentation tasks. Popular variants of the the R-CNN model include R-CNN, Fast R-CNN, Faster R-CNN, and Mask R-CNN.


.. _rcnn_limitations_inf1:

- **Architecture Fit** - R-CNN models can have a few limitations and considerations on Inferentia: **RoI Align operators**: At this time, RoI Align operators typically cannot run efficiently on NeuronCore v1. As a result, RoI Align operators are mapped directly to CPU during compilation. R-CNN models that predict a low number of bounding boxes (<100) experience the best performance on Inferentia. **Large ResNet backbone**: R-CNNs that have a large ResNet backbone (such as ResNet-50 or ResNet-101) experience the greatest performance improvement on Inferentia because a larger portion of the R-CNN compute is accelerated.
- **Neuron Support** - Torch models must be traceable using :func:`torch.jit.trace` for compilation on Inferentia. Most `Detectron2 <https://github.com/facebookresearch/detectron2>`_-based R-CNNs are not jit traceable by default, so they cannot readily be compiled for optimized inference on Inferentia. The :ref:`torch-neuron-r-cnn-app-note` application note demonstrates how to compile and improve the performance of R-CNN models on Inferentia. It also provides an end-to-end example of running a Detectron2 R-CNN on Inferentia.

Models with Long Short-Term Memory (LSTM) networks
--------------------------------------------------

LSTMs use an internal state to process sequential data. LSTMs are commonly used to model temporal sequences of data in language processing and computer vision applications. 


- **Architecture Fit** - Models with LSTM cells are a good fit for Inferentia.
- **Neuron Support** - Models with LSTM networks are supported on Inferentia, please see :ref:`torch_neuron_lstm_support`.


Diffusion Models
----------------


- **Architecture Fit** - Diffusion models are a good fit for Inferentia.
- **Neuron Support** - Diffusion models are not supported on Inferentia as of the latest Neuron release. Please track the :ref:`Neuron Roadmap <neuron_roadmap>` for details.


Known Issues on Inferentia (NeuronCore v1)
------------------------------------------

Support of large models (impacts `torch-neuron` and `tensorflow-neuron` (TF1.x))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _2gb_protobuf_issue:

During compilation on Inferentia (NeuronCore v1), ``torch-neuron`` and ``tensorflow-neuron (TF1.x)`` export a protobuf that contains the model's graph structure and weights. This causes an issue when the total size of the model's weights exceeds the 2GB limitation of protobufs. As a result, customers who want to run large models such as **RegNet**, **Stable Diffusion**, and **t5-11b** might run into protobuf errors during compilation. 

This is a known issue related to the compilation process, not a hardware-dependent issue. Allowing large models like this to be compiled for inference on Inferentia (NeuronCore v1) is a feature that we intend to address in a future release. Please track the :ref:`Neuron Roadmap <neuron_roadmap>` for details.

.. note::

   Neuron release 2.5.0 added Experimental support for tracing models larger than 2GB `in `tensorflow-neuron (TF2.x)``, please see ``extract-weights`` flag in :ref:`tensorflow-ref-neuron-tracing-api` 

