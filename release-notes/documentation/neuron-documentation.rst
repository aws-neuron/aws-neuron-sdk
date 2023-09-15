.. _neuron-documentation-rn:

Neuron Documentation Release Notes
==================================

.. contents:: Table of contents
   :local:
   :depth: 1


Neuron 2.14.0
-------------
Date: 09/15/2023

- Neuron Calculator now supports multiple model configurations for Tensor Parallel Degree computation. See :ref:`neuron_calculator`
- Announcement to deprecate ``--model-type=transformer-inference`` flag. See :ref:`announce-deprecation-transformer-flag`
- Updated HF ViT benchmarking script to use ``--model-type=transformer`` flag. See :ref:`[script] <src/benchmark/pytorch/hf-google-vit_benchmark.py>`
- Updated ``torch_neuronx.analyze`` API documentation. See :ref:`torch_neuronx_analyze_api`
- Updated Performance benchmarking numbers for models on Inf1,Inf2 and Trn1 instances with 2.14 release bits. See :ref:`_benchmark`
- New tutorial for Training Llama2 7B with Tensor Parallelism and ZeRO-1 Optimizer using ``neuronx-distributed``  :ref:`llama2_7b_tp_zero1_tutorial`
- New tutorial for ``T5-3B`` model inference using ``neuronx-distributed``  (:pytorch-neuron-src:`tutorial <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>`)
- Updated ``Neuron Persistent Cache`` documentation regarding clarification of flags parsed by ``neuron_cc_wrapper`` tool which is a wrapper over ``Neuron Compiler CLI``. See :ref:`neuron-caching`
- Added ``tokenizers_parallelism=true`` in various notebook scripts to supress tokenizer warnings making errors easier to detect
- Updated Neuron device plugin and scheduler YAMLs to point to latest images.  See `yaml configs <https://github.com/aws-neuron/aws-neuron-sdk/tree/master/src/k8>`_
- Added notebook script to fine-tune ``deepmind/language-perceiver`` model using ``torch-neuronx``. See `sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_text_classification/LanguagePerceiver.ipynb>`_
- Added notebook script to fine-tune ``clip-large`` model using ``torch-neuronx``. See `sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_contrastive_image_text/CLIPLarge.ipynb>`_
- Added ``SD XL Base+Refiner`` inference sample script using ``torch-neuronx``. See `sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sdxl_base_and_refiner_1024_inference.ipynb>`_
- Upgraded default ``diffusers`` library from 0.14.0 to latest 0.20.2 in ``Stable Diffusion 1.5`` and ``Stable Diffusion 2.1`` inference scripts. See `sample scripts <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference>`_
- Added ``Llama-2-13B`` model training script using ``neuronx-nemo-megatron`` ( `tutorial <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`_ )




Neuron 2.13.0
-------------
Date: 08/28/2023


- Added tutorials for GPT-NEOX 6.9B and 20B models training using neuronx-distributed. See more at :ref:`tp_tutorials`
- Added TensorFlow 2.x (``tensorflow-neuronx``) analyze_model API section. See more at :ref:`tensorflow-ref-neuron-analyze_model-api`
- Updated setup instructions to fix path of existing virtual environments in DLAMIs. See more at :ref:`setup guide <setup-guide-index>`
- Updated setup instructions to fix pinned versions in upgrade instructions of setup guide. See more at :ref:`setup guide <setup-guide-index>`
- Updated tensorflow-neuron HF distilbert tutorial to improve performance by removing HF pipeline. See more at :ref:`[html] </src/examples/tensorflow/huggingface_bert/huggingface_bert.html>` :github:`[notebook] </src/examples/tensorflow/huggingface_bert/huggingface_bert.ipynb>`
- Updated training troubleshooting guide in torch-neuronx to describe network Connectivity Issue on trn1/trn1n 32xlarge with Ubuntu. See more at :ref:`pytorch-neuron-traning-troubleshooting`
- Added "Unsupported Hardware Operator Code" section to Neuron Runtime Troubleshooting page. See more at :ref:`nrt-troubleshooting`
- Removed 'Experimental' tag from ``neuronx-distributed`` section for training. ``neuronx-distributed`` Training is now considered stable and ``neuronx-distributed`` inference is considered as experimental.
- Added FLOP count(``flop_count``) and connected Neuron Device ids (``connected_devices``) to sysfs userguide. See :ref:`neuron-sysfs-ug`
- Added tutorial for ``T5`` model inference.  See more at :pytorch-neuron-src:`[notebook] <torch-neuronx/t5-inference-tutorial.ipynb>`
- Updated neuronx-distributed api guide and inference tutorial. See more at :ref:`api_guide` and :ref:`tp_inference_tutorial`
- Announcing End of support for ``AWS Neuron reference for Megatron-LM`` starting Neuron 2.13. See more at :ref:`announce-eol-megatronlm`
- Announcing end of support for ``torch-neuron`` version 1.9 starting Neuron 2.14. See more at :ref:`announce-eol-pytorch19`
- Upgraded ``numpy`` version to ``1.21.6`` in various training scripts for `Text Classification <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training>`_
- Added license for Nemo Megatron to SDK Maintenance Policy. See more at :ref:`sdk-maintenance-policy`
- Updated ``bert-japanese`` training Script to use ``multilingual-sentiments`` dataset. See `hf-bert-jp <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_bert_jp> `_
- Added sample script for LLaMA V2 13B model inference using transformers-neuronx. See `neuron samples repo <https://github.com/aws-neuron/aws-neuron-samples/>`_
- Added samples for training GPT-NEOX 20B and 6.9B models using neuronx-distributed. See `neuron samples repo <https://github.com/aws-neuron/aws-neuron-samples/>`_
- Added sample scripts for CLIP and Stable Diffusion XL inference using torch-neuronx. See `neuron samples repo <https://github.com/aws-neuron/aws-neuron-samples/>`_
- Added sample scripts for vision and language Perceiver models inference using torch-neuronx. See `neuron samples repo <https://github.com/aws-neuron/aws-neuron-samples/>`_
- Added camembert training/finetuning example for Trn1 under hf_text_classification in torch-neuronx. See `neuron samples repo <https://github.com/aws-neuron/aws-neuron-samples/>`_
- Updated Fine-tuning Hugging Face BERT Japanese model sample in torch-neuronx. See `neuron samples repo <https://github.com/aws-neuron/aws-neuron-samples/>`_
- See more neuron samples changes in `neuron samples release notes <https://github.com/aws-neuron/aws-neuron-samples/blob/master/releasenotes.md>`_
- Added samples for pre-training GPT-3 23B, 46B and 175B models using neuronx-nemo-megatron library. See `aws-neuron-parallelcluster-samples <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples>`_
- Announced End of Support for GPT-3 training using aws-neuron-reference-for-megatron-lm library. See `aws-neuron-parallelcluster-samples <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples>`_
- Updated bert-fine-tuning SageMaker sample by replacing amazon_reviews_multi dataset with amazon_polarity dataset. See `aws-neuron-sagemaker-samples <https://github.com/aws-neuron/aws-neuron-sagemaker-samples>`_


Neuron 2.12.0
-------------
Date: 07/19/2023

- Added best practices user guide for benchmarking performance of Neuron Devices `Benchmarking Guide and Helper scripts <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/microbenchmark>`_
- Announcing end of support for Ubuntu 18. See more at :ref:`announce-eol-ubuntu18`
- Improved sidebar navigation in Documentation.
- Removed support for Distributed Data Parallel(DDP) Tutorial.
  

Neuron 2.11.0
-------------

Date: 06/14/2023

- New :ref:`neuron_calculator` Documentation section to help determine number of Neuron Cores needed for LLM Inference.
- Added App Note :ref:`neuron_llm_inference`
- New ``ML Libraries`` Documentation section to have :ref:`neuronx-distributed-index` and :ref:`transformers_neuronx_readme`
- Improved Installation and Setup Guides for the different platforms supported. See more at :ref:`setup-guide-index`
- Added Tutorial :ref:`setup-trn1-multi-node-execution`
