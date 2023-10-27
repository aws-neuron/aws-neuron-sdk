.. _model_samples_training_trn1:

Training Samples/Tutorials (Trn1/Trn1n)
=======================================

.. contents:: Table of contents
   :local:
   :depth: 1


.. _encoder_model_samples_training_trn1:
 
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

   * - bert-base-cased
     - torch-neuronx
     - * `Fine-tune a "bert-base-cased" PyTorch model for Text Classification  <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/BertBaseCased.ipynb>`_
       * `How to fine-tune a "bert base cased" PyTorch model with AWS Trainium (Trn1 instances) for Sentiment Analysis <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_sentiment_analysis/01-hf-single-neuron.ipynb>`_
    
   * - bert-base-uncased
     - torch-neuronx
     - * `Fine-tune a "bert-base-uncased" PyTorch model <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/BertBaseUncased.ipynb>`_
       * `Fine tuning BERT base model from HuggingFace on Amazon SageMaker <https://github.com/aws-neuron/aws-neuron-sagemaker-samples/blob/master/training/trn1-bert-fine-tuning-on-sagemaker/bert-base-uncased-amazon-polarity.ipynb>`_
   
   * - bert-large-cased
     - torch-neuronx
     - * `Fine-tune a "bert-large-cased" PyTorch model  <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/BertLargeCased.ipynb>`_
    
   * - bert-large-uncased
     - torch-neuronx
     - * :ref:`hf-bert-pretraining-tutorial`
       * `Launch Bert Large Phase 1 pretraining job on Parallel Cluster <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/dp-bert-launch-job.md>`_
       * `Launch a Multi-Node PyTorch Neuron Training Job on Trainium Using TorchX and EKS <https://github.com/aws-neuron/aws-neuron-eks-samples/tree/master/dp_bert_hf_pretrain#tutorial-launch-a-multi-node-pytorch-neuron-training-job-on-trainium-using-torchx-and-eks>`_
       * :ref:`torch-hf-bert-finetune`
       * `Fine-tune a "bert-large-uncased" PyTorch model <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/BertLargeCased.ipynb>`_
       

   * - roberta-base
     - tensorflow-neuronx
     - * `Fine-tune a "roberta-base" PyTorch model <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/RobertaBase.ipynb>`_


   * - roberta-large
     - torch-neuronx
     - * `Fine-tune a "roberta-large" PyTorch model <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/RobertaLarge.ipynb>`_

  
   * - xlm-roberta-base
     - torch-neuronx
     - * `Fine-tune a "xlm-roberta-base" PyTorch model <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/XlmRobertaBase.ipynb>`_


   * - alberta-base-v2
     - torch-neuronx
     - * `Fine-tune a "alberta-base-v2" PyTorch model <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/AlbertBase.ipynb>`_


   * - distilbert-base-uncased
     - torch-neuronx
     - * `Fine-tune a "distilbert-base-uncased" PyTorch model <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/DistilbertBaseUncased.ipynb>`_


   * - camembert-base
     - torch-neuronx
     - * `Fine-tune a "camembert-base PyTorch model <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/CamembertBase.ipynb>`_

   * - cl-tohoku/bert-base-japanese-whole-word-masking
     - torch-neuronx
     - * `Fine-tuning & Deployment Hugging Face BERT Japanese model	<https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_bert_jp/bert-jp-tutorial.ipynb>`_




.. _decoder_model_samples_training_trn1:


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

   * - gpt-2
     - torch-neuronx
     - * `How to run training jobs for "gpt2" PyTorch model with AWS Trainium <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_language_modeling/gpt2/gpt2.ipynb>`_
       * :ref:`zero1-gpt2-pretraining-tutorial`
   
   
   * - gpt-3
     - neuronx-nemo-megatron
     - * `Launch a GPT-3 23B pretraining job using neuronx-nemo-megatron <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md>`_
       * `Launch a GPT-3 46B pretraining job using neuronx-nemo-megatron <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md>`_
       * `Launch a GPT-3 175B pretraining job using neuronx-nemo-megatron <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md>`_
    

   * - GPT-NEOX-20B
     - neuronx-distributed
     - * :ref:`gpt_neox_20b_tp_zero1_tutorial`
       * `Training GPT-NEOX 20B model using neuronx-distributed	 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain>`_

   
   * - GPT-NEOX-6.9B
     - neuronx-distributed
     - * :ref:`gpt_neox_tp_zero1_tutorial`
       * `Training GPT-NEOX 6.9B model using neuronx-distributed		 <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_6.9b_hf_pretrain>`_


   * - meta-llama/Llama-2-7b
     - neuronx-distributed
     - * :ref:`llama2_7b_tp_zero1_tutorial`

   * - meta-llama/Llama-2-70b
     - neuronx-distributed
     - * :ref:`llama2_70b_tp_pp_tutorial`

   * - meta-llama/Llama-2
     - neuronx-nemo-megatron
     - * `Launch a Llama-2-7B pretraining job using neuronx-nemo-megatron <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`_
       * `Launch a Llama-2-13B pretraining job using neuronx-nemo-megatron <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`_
       * `Launch a Llama-2-70B pretraining job using neuronx-nemo-megatron <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`_



.. _encoder_decoder_model_samples_training_trn1:

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

   * - t5-small
     - * torch-neuronx
       * optimum-neuron
     - * :ref:`torch-hf-t5-finetune`



.. _vision_transformer_model_samples_training_trn1:

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

   * - google/vit-base-patch16-224-in21k
     - torch-neuronx
     - * `Fine-tune a pretrained HuggingFace vision transformer PyTorch model  <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_image_classification/vit.ipynb>`_

    
   * - openai/clip-vit-base-patch32
     - torch-neuronx
     - * `Fine-tune a pretrained HuggingFace CLIP-base PyTorch model with AWS Trainium  <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_contrastive_image_text/CLIPBase.ipynb>`_


   * - openai/clip-vit-large-patch14
     - torch-neuronx
     - * `Fine-tune a pretrained HuggingFace CLIP-large PyTorch model with AWS Trainium <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_contrastive_image_text/CLIPLarge.ipynb>`_




.. _multi_modal_model_samples_training_trn1:

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
       

   * - language-perceiver
     - torch-neuronx
     - * `How to fine-tune a "language perceiver" PyTorch model with AWS Trainium (trn1 instances) <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/LanguagePerceiver.ipynb>`_


   * - vision-perceiver-conv
     - torch-neuronx
     - * `How to fine-tune a pretrained HuggingFace Vision Perceiver Conv <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_image_classification/VisionPerceiverConv.ipynb>`_



.. _cnn_model_samples_training_trn1:

Convolutional Neural Networks(CNN)
----------------------------------

.. list-table::
   :widths: 20 15 45 
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size


   * - resnet50
     - torch-neuronx
     - * `How to fine-tune a pretrained ResNet50 Pytorch model with AWS Trainium (trn1 instances) using NeuronSDK <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/resnet50>`_

   * - milesial/Pytorch-UNet
     - torch-neuronx
     - * `This notebook shows how to fine-tune a pretrained UNET PyTorch model with AWS Trainium (trn1 instances) using NeuronSDK. <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/unet_image_segmentation>`_

