.. _model_samples_inference_inf2_trn1:

Inference Samples/Tutorials (Inf2/Trn1)
=======================================

.. contents:: Table of contents
   :local:
   :depth: 1


.. _encoder_model_samples_inference_inf2_trn1:
 
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
     - torch-neuronx
     - * :ref:`BERT TorchServe tutorial <pytorch-tutorials-torchserve-neuronx>`
       * HuggingFace pretrained BERT tutorial :ref:`[html] </src/examples/pytorch/torch-neuronx/bert-base-cased-finetuned-mrpc-inference-on-trn1-tutorial.ipynb>` :pytorch-neuron-src:`[notebook] <torch-neuronx/bert-base-cased-finetuned-mrpc-inference-on-trn1-tutorial.ipynb>`
       * `LibTorch C++ Tutorial for HuggingFace Pretrained BERT <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuron/tutorials/tutorial-libtorch.html#pytorch-tutorials-libtorch>`_
       * `Compiling and Deploying HuggingFace Pretrained BERT on Inf2 on Amazon SageMaker <https://github.com/aws-neuron/aws-neuron-sagemaker-samples/blob/master/inference/inf2-bert-on-sagemaker/inf2_bert_sagemaker.ipynb>`_


   * - bert-base-cased-finetuned-mrpc
     - neuronx-distributed
     - * :ref:`tp_inference_tutorial`


   * - bert-base-uncased
     - torch-neuronx
     - * `HuggingFace Pretrained BERT Inference on Trn1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_bert_inference_on_trn1.ipynb>`_

   * - distilbert-base-uncased
     - torch-neuronx
     - * `HuggingFace Pretrained DistilBERT Inference on Trn1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_distilbert_Inference_on_trn1.ipynb>`_


   * - roberta-base
     - tensorflow-neuronx
     - * HuggingFace Roberta-Base :ref:`[html]</src/examples/tensorflow/tensorflow-neuronx/tfneuronx-roberta-base-tutorial.ipynb>` :github:`[notebook] </src/examples/tensorflow/tensorflow-neuronx/tfneuronx-roberta-base-tutorial.ipynb>`


   * - roberta-large
     - torch-neuronx
     - * `HuggingFace Pretrained RoBERTa Inference on Trn1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_roberta_inference_on_frn1.ipynb>`_



.. _decoder_model_samples_inference_inf2_trn1:

Decoders
--------

.. list-table::
   :widths: 20 15 45 
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - Model
     - Frameworks/Libraries
     - Samples and Tutorials

   * - gpt2
     - torch-neuronx
     - * `HuggingFace Pretrained GPT2 Feature Extraction on Trn1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_gpt2_feature_extraction_on_trn1.ipynb>`_
  
   * - meta-llama/Llama-3-8b
     - transformers-neuronx
     - * `Run Hugging Face meta-llama/Llama-3-8b autoregressive sampling on Inf2 & Trn1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-3-8b-sampling.ipynb>`_

   * - meta-llama/Llama-3-70b
     - transformers-neuronx
     - * `Run Hugging Face meta-llama/Llama-3-70b autoregressive sampling on Inf2 & Trn1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-3-70b-sampling.ipynb>`_


   * - meta-llama/Llama-2-13b
     - transformers-neuronx
     - * `Run Hugging Face meta-llama/Llama-2-13b autoregressive sampling on Inf2 & Trn1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb>`_


   * - meta-llama/Llama-2-70b
     - transformers-neuronx
     - * `Run Hugging Face meta-llama/Llama-2-70b autoregressive sampling on Inf2 & Trn1 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-70b-sampling.ipynb>`_
       *  `Run speculative sampling on Meta Llama models [Beta] <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/speculative_sampling.ipynb>`_

   * - meta-llama/Llama-2-7b
     - neuronx-distributed
     - * Run Hugging Face meta-llama/Llama-2-7b autoregressive sampling on Inf2 & Trn1 (:ref:`[html] </src/examples/pytorch/neuronx_distributed/llama/llama2_inference.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/llama/llama2_inference.ipynb>`)

   * - mistralai/Mistral-7B-Instruct-v0.1
     - transformers-neuronx
     - * :ref:`Run Mistral-7B-Instruct-v0.1 autoregressive sampling on Inf2 & Trn1 <mistral_gqa_code_sample>`

   * - mistralai/Mistral-7B-Instruct-v0.2
     - transformers-neuronx
     - * `Run Hugging Face mistralai/Mistral-7B-Instruct-v0.2 autoregressive sampling on Inf2 & Trn1 [Beta] <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/mistralai-Mistral-7b-Instruct-v0.2.ipynb>`_

   * - Mixtral-8x7B-v0.1
     - transformers-neuronx
     - * `Run Hugging Face mistralai/Mixtral-8x7B-v0.1 autoregressive sampling on Inf2 & Trn1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/mixtral-8x7b-sampling.ipynb>`_

   * - codellama/CodeLlama-13b-hf
     - transformers-neuronx
     - * `Run Hugging Face codellama/CodeLlama-13b-hf autoregressive sampling on Inf2 & Trn1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/transformers-neuronx/inference/codellama-13b-16k-sampling.ipynb>`_

.. _encoder_decoder_model_samples_inference_inf2_trn1:

Encoder-Decoders  
----------------


.. list-table::
   :widths: 20 15 45 
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - Model
     - Frameworks/Libraries
     - Samples and Tutorials

   * - t5-large
     - * torch-neuronx
       * optimum-neuron
     - * T5 inference tutorial :ref:`[html] </src/examples/pytorch/torch-neuronx/t5-inference-tutorial.ipynb>` :pytorch-neuron-src:`[notebook] <torch-neuronx/t5-inference-tutorial.ipynb>`

   * - t5-3b
     - neuronx-distributed
     - * T5 inference tutorial :ref:`[html] </src/examples/pytorch/neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`

   * - google/flan-t5-xl
     - neuronx-distributed
     - * flan-t5-xl inference tutorial :ref:`[html] </src/examples/pytorch/neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`



.. _vision_transformer_model_samples_inference_inf2_trn1:

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

   * - google/vit-base-patch16-224
     - torch-neuronx
     - * `HuggingFace Pretrained ViT Inference on Trn1 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_vit_inference_on_inf2.ipynb>`_


   * - clip-vit-base-patch32
     - torch-neuronx
     - * `HuggingFace Pretrained CLIP Base Inference on Inf2 <https://github.com/aws-neuron/aws-neuron-samples-staging/blob/master/torch-neuronx/inference/hf_pretrained_clip_base_inference_on_inf2.ipynb>`_


   * - clip-vit-large-patch14
     - torch-neuronx
     - * `HuggingFace Pretrained CLIP Large Inference on Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_clip_large_inference_on_inf2.ipynb>`_



.. _cnn_model_samples_inference_inf2_trn1:

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

   * - resnet50
     - torch-neuronx
     - * `Torchvision Pretrained ResNet50 Inference on Trn1 / Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/tv_pretrained_resnet50_inference_on_trn1.ipynb>`_
       *  Torchvision ResNet50 tutorial :ref:`[html] </src/examples/pytorch/torch-neuronx/resnet50-inference-on-trn1-tutorial.ipynb>` :pytorch-neuron-src:`[notebook] <torch-neuronx/resnet50-inference-on-trn1-tutorial.ipynb>`

   * - resnet50
     - tensorflow-neuronx
     - * :ref:`tensorflow-servingx-neuronrt-visible-cores`

   * - unet
     - torch-neuronx
     - * `Pretrained UNet Inference on Trn1 / Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/pretrained_unet_inference_on_trn1.ipynb>`_

   * - vgg
     - torch-neuronx
     - * `Torchvision Pretrained VGG Inference on Trn1 / Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/tv_pretrained_vgg_inference_on_trn1.ipynb>`_


.. _sd_model_samples_inference_inf2_trn1:

Stable Diffusion
----------------

.. list-table::
   :widths: 20 15 45 
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - Model
     - Frameworks/Libraries
     - Samples and Tutorials

   * - stable-diffusion-v1-5
     - torch-neuronx
     - * `HuggingFace Stable Diffusion 1.5 (512x512) Inference on Trn1 / Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sd15_512_inference.ipynb>`_

   * - stable-diffusion-2-1-base
     - torch-neuronx
     - * `HuggingFace Stable Diffusion 2.1 (512x512) Inference on Trn1 / Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sd2_512_inference.ipynb>`_

   * - stable-diffusion-2-1
     - torch-neuronx
     - * `HuggingFace Stable Diffusion 2.1 (768x768) Inference on Trn1 / Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sd2_768_inference.ipynb>`_
       * `Deploy & Run Stable Diffusion on SageMaker and Inferentia2 <https://github.com/aws-neuron/aws-neuron-sagemaker-samples/blob/master/inference/stable-diffusion/StableDiffusion2_1.ipynb>`_

   * - stable-diffusion-xl-base-1.0
     - torch-neuronx
     - * `HuggingFace Stable Diffusion XL 1.0 (1024x1024) Inference on Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sdxl_base_1024_inference.ipynb>`_
       * `HuggingFace Stable Diffusion XL 1.0 Base and Refiner (1024x1024) Inference on Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sdxl_base_and_refiner_1024_inference.ipynb>`_

   * - stable-diffusion-2-inpainting
     - torch-neuronx
     - * `stable-diffusion-2-inpainting model Inference on Trn1 / Inf2 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sd2_inpainting_936_624_inference.ipynb>`_



.. _multi_modal_model_samples_inference_inf2_trn1:

Multi Modal
-----------

.. list-table::
   :widths: 20 15 45 
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size


   * - Model
     - Frameworks/Libraries
     - Samples and Tutorials
       

   * - multimodal-perceiver
     - torch-neuronx
     - * `HuggingFace Multimodal Perceiver Inference on Trn1 / Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_perceiver_multimodal_inference.ipynb>`_


   * - language-perceiver
     - torch-neuronx
     - * `HF Pretrained Perceiver Language Inference on Trn1 / Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_perceiver_language_inference.ipynb>`_


   * - vision-perceiver-conv
     - torch-neuronx
     - * `HF Pretrained Perceiver Image Classification Inference on Trn1 / Inf2 <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_perceiver_vision_inference.ipynb>`_



 







 











