.. _neuron-documentation-rn:

Neuron Documentation Release Notes
==================================

.. contents:: Table of contents
   :local:
   :depth: 1


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
