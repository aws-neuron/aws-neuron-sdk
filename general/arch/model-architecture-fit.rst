.. _model_architecture_fit:

Model Architecture Fit Guidelines
=================================

.. contents:: Table of contents
   :local:
   :depth: 2

Introduction
------------

AWS Neuron SDK enables you to train and deploy a wide range of deep learning models on AWS Inferentia and Trainium. Inferentia is the first generation NeuronDevice optimized for inference, and Trainium is the second generation that adds support for model training. 

This document describes what types of deep learning model architectures are a good fit for Inferentia and Trainium. 


Model Support Overview
----------------------


AWS Trainium (NeuronCore v2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Last update* - 10/10/2022

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size
   

   *  - Model Family/  
        Neural Network Architecture
   
      - Category
      - Hardware Architecture
      - PyTorch Neuron (``torch-neuronx``)
      - TensorFlow Neuron (``tensorflow-neuronx``)


   *  - Transformers Encoders
      - NLP
      - Good Fit
      - Supported
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Transformers Decoders
      - NLP
      - Good Fit
      - Supported
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Transformers Encoder-Decoder (Sequence-to-sequence)
      - NLP
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Vision Transformers
      - Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Diffusion models
      - Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Convolutional Neural Network (CNN) models
      - Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - LSTMs
      - Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`


   *  - RNNs
      - Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`

.. note::

   Supported means that at least a single model of the model family or the neural-network architecture already enabled. 

AWS Inferentia (NeuronCore v1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Last update* - 10/10/2022

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

   *  - Transformers Encoders
      - NLP
      - Good Fit
      - Supported
      - Supported
      - Supported

   *  - Transformers Decoders
      - NLP
      - Not a Good Fit
      - NA
      - NA
      - NA

   *  - Transformers Encoder-Decoder (Sequence-to-sequence)
      - NLP
      - Not a Good Fit
      - NA
      - NA
      - NA

   *  - Vision Transformers
      - Computer Vision
      - Good Fit
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - Diffusion models
      - Computer Vision
      - Not a Good Fit
      - NA
      - NA
      - NA

   *  - Convolutional Neural Network (CNN) models
      - Computer Vision
      - Good Fit
      - Supported
      - Supported
      - :ref:`Roadmap Item <neuron_roadmap>`

   *  - LSTMs
      - Computer Vision
      - Not a Good Fit
      - NA
      - NA
      - NA


   *  - RNNs
      - Computer Vision
      - Not a Good Fit
      - NA
      - NA
      - NA

.. note::

   Supported means that at least a single model of the model family or the neural-network architecture already enabled. 




Natural Language Processing (NLP) Models with Transformers
----------------------------------------------------------

Encoder models, or autoencoding models with Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autoencoding models use only the encoder part of the Transformer architecture. Representatives of this family include models like **BERT, distilBERT, XLM-BERT, Roberta, BioBert**, etc.  Since the encoding process in these models can be parallelized, you can expect these models to run well both on Inferentia and Trainium. 

Trainium
^^^^^^^^

- **Architecture Fit** - Autoencoding models are a good fit for Trainium. 
- **Neuron Support** - Neuron SDK supports Autoencoding models training and evaluation on Trainium. Please refer to the Trainium tutorials for more details.

Inferentia
^^^^^^^^^^

- **Architecture Fit** - Autoencoding models are a good fit for Inferentia.
- **Neuron Support** - Neuron SDK support running Autoencoding models for inference on Inferentia. Please see :ref:`benchmark results <appnote-performance-benchmark>` of these models. To get started with NLP models you can refer to Neuron :ref:`PyTorch <pytorch-nlp>`, :ref:`TensorFlow <tensorflow-nlp>` and :ref:`MXNet <mxnet-nlp>` NLP tutorials.

Decoder models, or autoregressive models with Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Autoregressive models keep only the decoder part of the Transformer architecture. Representatives of this family include models like **GPT-3, GPT-2**, etc.

Trainium
^^^^^^^^

- **Architecture Fit** - Autoregressive models are a good fit for Trainium.  
- **Neuron Support** - Neuron SDK supports Autoregressive models training and evaluation on Trainium. Please refer to the Trainium tutorials for more details.

Inferentia
^^^^^^^^^^

- **Architecture Fit** - Autoregressive models are not a good fit for Inferentia. Usually the decoder part in these models is the most significant performance bottleneck since it must be executed once per output token, causing frequent access to the memory. Due to this these models typically experience the best performance only when the decoder maximum sequence length is short (e.g., 128).
- **Neuron Support** - Neuron SDK does not support Autoregressive models inference on Inferentia.

Encoder-decoder models, or sequence-to-sequence models with Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sequence-to-sequence models use both of encoder and decoder of the Transformer architecture. Representatives of this family include models like **T5, Bart, Marian MT**, etc.

Trainium
^^^^^^^^

- **Architecture Fit** - Sequence-to-sequence models are a good fit for Trainium.
- **Neuron Support** - Sequence-to-sequence models are not supported on Trainium as of the latest Neuron release, please track :ref:`Neuron Roadmap <neuron_roadmap>` for details.

Inferentia
^^^^^^^^^^

- **Architecture Fit** - Sequence-to-sequence models are not a good fit for Inferentia. Like decoder models explained above, usually the decoder part in these sequence-to-sequence models is the most significant performance bottleneck since it must be executed once per output token, causing frequent access to the memory. Due to this, even when you enabled the models to run on Inferentia with wrapping the decoder part, these models typically experience the best performance only when the decoder maximum sequence length is short (e.g., 128).
- **Neuron Support** - Neuron SDK does not support sequence-to-sequence models inference on Inferentia out of the box. However, you can run a model with defining wrappers around the encoder and decoder portions of it. For example, please refer to :ref:`MarianMT tutorial </src/examples/pytorch/transformers-marianmt.ipynb>` on Inferentia for more details. 

Computer Vision Models
----------------------

Convolutional Neural Network (CNN) based models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CNN based models are used for applications in image classification and object detection. Representatives of this family include models like **ResNet, ResNext, VGG, YOLO, SSD**, etc.

Trainium
^^^^^^^^

- **Architecture Fit** - CNN based models are a good fit for Trainium.
- **Neuron Support** - CNN based models are not supported on Trainium as of the latest Neuron release, please track :ref:`Neuron Roadmap <neuron_roadmap>` for details.

Inferentia
^^^^^^^^^^

- **Architecture Fit** - CNN based models are a good fit for Inferentia.
- **Neuron Support** - Neuron SDK supports CNN based models inference on Inferentia. Please see :ref:`benchmark results <appnote-performance-benchmark>` of these models. To get started with these models you can refer to Neuron :ref:`PyTorch <pytorch-computervision>`, :ref:`TensorFlow <tensorflow-computervision>` and :ref:`MXNet <mxnet-computervision>` tutorials.

Region-based CNN (R-CNN) models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

R-CNN models search limited number of regions on the image, while typical CNN models classify a huge number of regions, enabling running faster in object detection applications. The representatives of this family include models include **R-CNN, Fast R-CNN, Faster R-CNN**, etc.

Trainium
^^^^^^^^

- **Architecture Fit** - R-CNN models are a good fit for Trainium.
- **Neuron Support** -  R-CNN models are not supported on Trainium as of the latest Neuron release, please track :ref:`Neuron Roadmap <neuron_roadmap>` for details.

Inferentia
^^^^^^^^^^

- **Architecture Fit** - R-CNN models are not a good fit for Inferentia. R-CNN models can have a few limitations running on Inferentia for the following reasons: (1) While torch models must be traceable using ``torch.jit.trace`` for compilation on Inferentia, most R-CNNs are not jit traceable by default so they cannot readily be compiled for optimized inference on Inferentia. (2) At the moment, ROI Align operators typically cannot run efficiently on Neuron Cores. As a result, ROI Align operators are mapped directly to CPU during compilation, affecting the performance. R-CNN models that predict a low number of bounding boxes (e.g., <100) experience the best performance on Inferentia. (3) Since ResNet backbone part is fully utilizing Inferentia features such as parallelization, R-CNN models with a small ResNet backbone usually do not experience as much of performance booster on Inferentia. Consequently, R-CNNs that have a large ResNet backbone (such as ResNet-50 or ResNet-101) experience the greatest performance improvement on Inferentia.
- **Neuron Support** - While Neuron SDK does not support R-CNN based models inference on Inferentia as of the latest Neuron release, there will be R-CNN support in future releases. Please check the future releases and track :ref:`neuron_roadmap` for details.



Models with Long Short-Term Memory (LSTM) networks
--------------------------------------------------

Models with LSTM networks allow outputs from previous time steps taken as inputs for the current time step and are used in the applications such as time series prediction, speech recognition, etc.

Trainium
~~~~~~~~

- **Architecture Fit** - Models with LSTM networks are a good fit for Trainium.
- **Neuron Support** - Models with LSTM networks are not supported on Trainium as of the latest Neuron release, please track :ref:`Neuron Roadmap <neuron_roadmap>` for details.

Inferentia
~~~~~~~~~~

- **Architecture Fit** - Models with LSTM networks are not a good fit for Inferentia.  LSTM networks allow outputs from previous time steps taken as inputs for the current time step. This recurrent behavior is implemented using control flow. Since Inferentia does not support control flow, the model will have to be partitioned to execute partially on the CPU, which will affect the inference performance. The models should be unrolled to work with Neuron.
- **Neuron Support** - While the models with LSTM networks are not supported on Inferentia as of the latest Neuron release, there will be LSTM support in future releases. Please check the future releases and track :ref:`Neuron Roadmap <neuron_roadmap>` for details.


Diffusion Models
----------------

Trainium
~~~~~~~~

- **Architecture Fit** - Diffusion models are a good fit for Trainium.
- **Neuron Support** - Diffusion models are not supported on Trainium as of the latest Neuron release, please track :ref:`Neuron Roadmap <neuron_roadmap>` for details.

Inferentia
~~~~~~~~~~

- **Architecture Fit** - Diffusion models are not a good fit for Inferentia. Diffusion models utilize encoder and decoder that includes control flow, conditions mechanism and attention layers. Since Inferentia does not support control flow, the model will have to be partitioned to execute partially on the CPU, which will affect the size of the model compilation as well as the inference performance.
- **Neuron Support** - Diffusion models are not supported on Inferentia as of the latest Neuron release. Please track :ref:`Neuron Roadmap <neuron_roadmap>` for details.


Known Issues on Inferentia
--------------------------

Support of large models (impacts `torch-neuron` and `tensorflow-neuron`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Inferentia, the weights of the models are put into a buffer that has 2GB limitation at the moment. As a result, customers who want to run model inference with large weights such as **RegNet**, **Stable Diffusion**, and **t5-11b**, might run into an error related to the size of the model. 

This is a known issue related to the compilation process, not a hardware-dependent issue. Allowing large models like this to be compiled is a feature that we intend to address in a future release. Currently these models are not well-supported. Please check the future releases.
