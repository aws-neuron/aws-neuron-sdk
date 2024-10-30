.. _neuron-documentation-rn:

Neuron Documentation Release Notes
==================================

.. contents:: Table of contents
   :local:
   :depth: 1

Neuron 2.20.0
---------------
Date: 09/16/2024

Neuron Compiler

- Added Getting Started with NKI guide for implementing a simple “Hello World” style NKI kernel and running it on a Neuron Device (Trainium/Inferentia2). See :ref:`nki_getting_started`
- Added NKI Programming Model guide for explaining the three main stages of the NKI programming model. See :ref:`nki_programming_model`
- Added NKI Kernel as a Framework Custom Operator guide for explaining how to insert a NKI kernel as a custom operator into a PyTorch or JAX model using simple code examples. See :ref:`nki_framework_custom_op`
- Added NKI Tutorials for the following kernels: Tensor addition, Transpose2D, AveragePool2D, Matrix multiplication, RMSNorm, Fused Self Attention, LayerNorm, and Fused Mamba. See :ref:`nki_kernels`
- Added NKI Kernels guide for optimized kernel examples. See :ref:`nki_kernels`
- Added Trainium/Inferentia2 Architecture Guide for NKI. See :ref:`trainium_inferentia2_arch`
- Added Profiling NKI kernels with Neuron Profile. See :ref:`neuron_profile_for_nki`
- Added NKI Performance Guide for explaining a recipe to find performance bottlenecks of NKI kernels and apply common software optimizations to address such bottlenecks. See :ref:`nki_perf_guide`
- Added NKI API Reference Manual with nki framework and types, nki.language, nki.isa, NKI API Common Fields, and NKI API Errors. See :ref:`nki_api_reference`
- Added NKI FAQ. See :ref:`nki_faq`
- Added NKI Known Issues. See :ref:`nki_known_issues`
- Updated Neuron Glossary with NKI terms. See :ref:`neuron_hw_glossary`
- Added new :ref:`NKI samples repository <https://github.com/aws-neuron/nki-samples>`
- Added average_pool2d, fused_mamba, layernorm, matrix_multiplication, rms_norm, sd_attention, tensor_addition, and transpose_2d kernel tutorials to the NKI samples respository. See :ref:`NKI samples repository <https://github.com/aws-neuron/nki-samples>`
- Added unit and integration tests for each kernel. See `NKI samples repository <https://github.com/aws-neuron/nki-samples>`_
- Updated Custom Operators API Reference Guide with updated terminology (HBM). See :ref:`custom-ops-api-ref-guide`

NeuronX Distributing Training (NxDT)

- Added NxDT (Beta) Developer Guide. See :ref:`nxdt_developer_guide`
- Added NxDT Developer Guide for Migrating from NeMo to Neuronx Distributed Training. See :ref:`nxdt_developer_guide_migration_nemo_nxdt`
- Added NxDT Developer Guide for Migrating from Neuron-NeMo-Megatron to Neuronx Distributed Training. See :ref:`nxdt_developer_guide_migration_nnm_nxdt`
- Added NxDT Developer Guide for Integrating a new dataset/dataloader. See :ref:`nxdt_developer_guide_integrate_new_dataloader`
- Added NxDT Developer Guide for Integrating a new model. See :ref:`nxdt_developer_guide_integrate_new_model`
- Added NxDT Developer Guide for Registering an optimizer and LR scheduler. See :ref:`Registering an optimizer and LR scheduler`
- Added NxDT YAML Configuration Overview. See :ref:`nxdt_config_overview`
- Added Neuronx Distributed Training Library Features documentation. See :ref:`nxdt_features`
- Added Installation instructions for NxDT. See :ref:`nxdt_installation_guide`
- Added Known Issues and Workarounds for NxDT. See :ref:`nxdt_known_issues`

NeuronX Distributed Core (NxD Core)

- Updated Developer guide for save/load checkpoint (neuronx-distributed ) with ZeRO-1 Optimizer State Offline Conversion. See :ref:`save_load_developer_guide`
- Added Developer guide for Standard Mixed Precision with NeuronX Distributed. See :ref:`standard_mixed_precision`
- Updated NeuronX Distributed API Guide LoRA finetuning support. See :ref:`api_guide`
- Added Developer guide for LoRA finetuning with NeuronX Distributed. See :ref:`lora_finetune_developer_guide`
- Updated CodeLlama tutorial with latest package versions. See `tutorial <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/neuronx_distributed/llama/codellama_16k_inference.html>`_
- Added tutorial for Fine-tuning Llama3 8B with tensor parallelism and LoRA using Neuron PyTorch-Lightning with NeuronX Distributed. See :ref:`llama3_8b_tp_ptl_lora_finetune_tutorial`
- Updated links in Llama2 NxD Finetuning tutorial. See :ref:`llama2_7b_tp_zero1_ptl_finetune_tutorial`
- Updated tokenizer download command in tutorials. See :ref:`llama2_7b_tp_zero1_tutorial`, :ref:`llama2_tp_pp_tutorial`, and :ref:`codegen25_7b_tp_zero1_tutorial`

JAX Neuron

- Added JAX Neuron Main page. See :ref:`jax-neuron-main`
- Added JAX Neuron plugin instructions. See :ref:`jax-neuronx-setup`
- Added JAX Neuron setup instructions. See :ref:`setup-jax-neuronx`

PyTorch NeuronX

- Updated Developer Guide for Training with PyTorch NeuronX with support for convolution in AMP. See :ref:`pytorch-neuronx-programming-guide`.
- Added inference samples for Wav2Vec2 conformer models with Relative Position Embeddings and Rotary Position Embedding. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_wav2vec2_conformer_relpos_inference_on_inf2.ipynb>`_ and `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_wav2vec2_conformer_rope_inference_on_inf2.ipynb>`_.
- Updated the ViT sample with updated accelerate version. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_image_classification/vit.ipynb>`_
- Updated PyTorch NeuronX Environment Variables with ``NEURON_TRANSFER_WITH_STATIC_RING_OPS``. See :ref:`pytorch-neuronx-envvars`
- Added inference samples for Pixart Alpha and PixArt Sigma models. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_pixart_alpha_inference_on_inf2.ipynb>`_ and `sample <torch-neuronx/inference/hf_pretrained_pixart_sigma_inference_on_inf2.ipynb>`_
- Added benchmarking scripts for PixArt alpha. See `benchmarking script <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/benchmark/pytorch/pixart_alpha_benchmark.py>`_

Transformers NeuronX

- Updated Transformers NeuronX Developer Guide with Multi-node inference support (TP/PP). See :ref:`transformers_neuronx_developer_guide`
- Updated Transformers NeuronX Developer Guide with BDH layout support. See :ref:`transformers_neuronx_developer_guide`
- Updated Transformers NeuronX Developer Guide with Flash Decoding to support long sequence lengths up to 128k. See :ref:`transformers_neuronx_developer_guide`
- Updated Transformers NeuronX Developer Guide with presharded weights support. See :ref:`transformers_neuronx_developer_guide`
- Added Llama 3.1 405b sample with 16k sequence length. See `tutorial <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-405b-multinode-16k-sampling.ipynb>`_
- Added Llama 3.1 70b 64k tutorial. See `tutorial <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-70b-64k-sampling.ipynb>`_
- Added Llama 3.1 8b 128k tutorial. See `tutorial <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-8b-128k-sampling.ipynb>`_
- Removed the sample llama-3-8b-32k-sampling.ipynb and replaced it with Llama-3.1-8B model sample llama-3.1-8b-32k-sampling.ipynb. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-3.1-8b-32k-sampling.ipynb>`_

Neuron Runtime

- Updated Neuron Runtime Troubleshooting guide with the latest hardware error codes and logs and with Neuron Runtime execution fails at out-of-bound access. See :ref:`nrt-troubleshooting`
- Updated Neuron Sysfs User Guide with new sysfs entries and device reset instructions. See :ref:`neuron-sysfs-ug`
- Added Neuron Runtime Input Dump on Trn1 documentation. See :ref:`nrt-input-dumps`

Containers

- Added Neuron Helm Chart repository to help streamline the deployment of AWS Neuron components on Amazon EKS. See `repo <https://github.com/aws-neuron/neuron-helm-charts>`_
- Updated Kubernetes container deployment process with Neuron Helm Chart documentation. See :ref:`k8s-neuron-helm-chart`
- Added guide for Deploying Neuron Container on Elastic Container Service (ECS). See :ref:`training-dlc-then-ecs-devflow`
- Added documentation for Neuron Plugins for Containerized Environments. See :ref:`neuron-container-plugins`
- Updated guide for locating DLC images. See :ref:`locate-neuron-dlc-image`

Neuron Tools

- Updated Neuron Profiler User Guide with Alternative output formats. See :ref:`neuron-profile-ug`

Software Maintenance and Misc

- Updated the Neuron Software Maintenance Policy. See :ref:`sdk-maintenance-policy`
- Added announcement and updated documentation for end of support start for Tensorflow-Neuron 1.x. See :ref:`announce-tfx-no-support`
- Added announcement and updated documentation for end of support start for 'neuron-device-version' field. See :ref:`eos-neuron-device-version`
- Added announcement and updated documentation for end of support start for ‘neurondevice’ resource name. See :ref:`eos-neurondevice`
- Added announcement and updated documentation for end of support start for AL2. See :ref:`eos-al2`
- Added announcement for maintenance mode for torch-neuron versions 1.9 and 1.10. See :ref:`announce-torch-neuron-eos`
- Added supported Protobuf versions to the Neuron Release Artifacts. See :ref:`latest-neuron-release-artifacts`
- Updated Neuron Github Roadmap. See :ref:`neuron_roadmap`

Neuron 2.19.0
-------------
Date: 07/03/2024


- Updated Transformers NeuronX Developer guide with support for inference for longer sequence lengths with Flash Attention kernel. See :ref:`Developer Guide <transformers_neuronx_developer_guide>`.
- Updated Transformers NeuronX developer guide with QKV Weight Fusion support. See :ref:`Developer Guide <transformers_neuronx_developer_guide>`.
- Updated Transformers NeuronX continuous batching developer guide with updated vLLM instructions and models supported. See :ref:`Developer Guide <transformers_neuronx_developer_guide_for_cb>`.
- Updated Neuronx Distributed User guide with interleaved pipeline support. See :ref:`api_guide`
- Added Codellama 13b 16k tutorial with NeuronX Distributed Inference library. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/codellama-13b-16k-sampling.ipynb>`_ 
- Updated PyTorch NeuronX Environment variables with custom SILU enabled via NEURON_CUSTOM_SILU. See :ref:`pytorch-neuronx-envvars`
- Updated ZeRO1 support to have FP32 master weights support and BF16 all-gather. See :ref:`zero1-gpt2-pretraining-tutorial`.
- Updated PyTorch 2.1 Appplication note with workaround for slower loss convergence for NxD LLaMA-3 70B pretraining using ZeRO1 tutorial. See :ref:`introduce-pytorch-2-1`.
- Updated Neuron DLAMI guide with support for new 2.19 DLAMIs. See :ref:`neuron-dlami-overview`.
- Updated HF-BERT pre-training documentation for port forwarding. See :ref:`hf-bert-pretraining-tutorial`
- Updated T5 inference tutorial with transformer flag. See  `sample <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/torch-neuronx/t5-inference-tutorial.html>`_ 
- Added support for Llama3 model training. See :ref:`llama3_tp_pp_tutorial` and :ref:`llama2_7b_tp_zero1_tutorial`
- Added support for Flash Attention kernel for training longer sequences in NeuronX Distributed. See :ref:`llama2_7b_tp_zero1_tutorial` and :ref:`api_guide`
- Updated Llama2 inference tutorial using NxD Inference library. See `sample <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/neuronx_distributed/llama/llama2_inference.html>`_ 
- Added new guide for Neuron node problem detection and recovery tool. See :ref:`configuration < k8s-neuron-problem-detector-and-recovery-irsa>` and :ref:`tutorial <k8s-neuron-problem-detector-and-recovery>`.
- Added new guide for Neuron Monitor container to enable easy monitoring of Neuron metrics in Kubernetes. Supports monitoring with Prometheus and Grafana. See :ref:`tutorial <k8s-neuron-monitor>`
- Updated Neuron scheduler extension documentation about enforcing allocation of contiguous Neuron Devices for the pods based on the Neuron instance type. See :ref:`tutorial <neuron_scheduler>`
- Updated Neuron Profiler User Guide with various UI enhancements. See :ref:`neuron-profile-ug`
- Added NeuronPerf support in Llama2 inference tutorial in NeuronX Distributed. See `sample <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/neuronx_distributed/llama/llama2_inference.html>`_ 
- Added announcement for maintenance mode of MxNet. See :ref:`announce-mxnet-maintenance`
- Added announcement for end of support of Neuron TensorFlow 1.x (Inf1). See :ref:`announce-tfx-eos`
- Added announcement for end of support of AL2. See :ref:`announce-eos-al2`
- Added announcement for end of support of 'neuron-device-version' field in neuron-monitor. See :ref:`announce-eos-neuron-device-version`
- Added announcement for end of support of 'neurondevice' resource name in Neuron Device K8s plugin. See :ref:`announce-eos-neurondevice`
- Added announcement for end of support for Probuf versions <= 3.19 for PyTorch NeuronX. See :ref:`announce-eos-probuf319`

Neuron 2.18.0
-------------
Date: 04/01/2024


- Updated PyTorch NeuronX developer guide with Snapshotting support. See :ref:`torch-neuronx-snapshotting`.
- Updated :ref:`api_guide` and :ref:`pp_developer_guide` with support for ``auto_partition`` API.
- Updated :ref:`api_guide` with enhanced checkpointing support with ``load`` API and ``async_save`` API.
- Updated documentation for ``PyTorch Lightning``  to train models using ``pipeline parallelism`` . See :ref:`API guide <api_guide>` and :ref:`Developer Guide <ptl_developer_guide>`.
- Updated NeuronX Distributed developer guide with support for :ref:`Autobucketing <nxd-inference-devguide-autobucketing>`
- Added PyTorch NeuronX developer guide for :ref:`Autobucketing <torch-neuronx-autobucketing-devguide>`.
- Updated :ref:`api_guide` and :ref:`llama2_tp_pp_tutorial` with support for asynchronous checkpointing.
- Updated Transformers NeuronX Developer guide with support for streamer and stopping criteria APIs. See :ref:`Developer Guide <transformers_neuronx_developer_guide>`.
- Updated Transformers NeuronX Developer guide with instructions for ``Repeating N-Gram Filtering``. See :ref:`Developer Guide <transformers_neuronx_developer_guide>`.
- Updated Transformers NeuronX developer guide with Top-K on-device sampling support [Beta]. See :ref:`Developer Guide <transformers_neuronx_developer_guide>`.
- Updated Transformers NeuronX developer guide with Checkpointing support and automatic model selection. See :ref:`Developer Guide <transformers_neuronx_developer_guide>`.
- Updated Transformers NeuronX Developer guide with support for speculative sampling [Beta]. See :ref:`Developer Guide <transformers_neuronx_developer_guide>`.
- Added sample for training CodeGen2.5 7B with Tensor Parallelism and ZeRO-1 Optimizer with ``neuronx-distributed``. See :ref:`codegen25_7b_tp_zero1_tutorial`.
- Added Tutorial for codellama/CodeLlama-13b-hf model inference with 16K seq length using Transformers Neuronx. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/codellama-13b-16k-sampling.ipynb>`_.
- Added Mixtral-8x7B Inference Sample/Notebook using TNx. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/mixtral-8x7b-sampling.ipynb>`_.
- Added Mistral-7B-Instruct-v0.2 Inference inference sample using TNx. See `sample <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/mistralai-Mistral-7b-Instruct-v0.2.ipynb>`_.
- Added announcement for Maintenance mode of TensorFlow 1.x. See :ref:`announce-tfx-maintenance`.
- Updated PyTorch 2.1 documentation to reflect stable (out of beta) support. See :ref:`introduce-pytorch-2-1`.
- Updated PyTorch NeuronX environment variables to reflect stable (out of beta) support. See :ref:`pytorch-neuronx-envvars`.
- Updated :ref:`latest-neuron-release-artifacts` with supported HuggingFace Transformers versions.
- Added user guide instructions for ``Neuron DLAMI``. See :ref:`neuron-dlami-overview`.
- Updated :ref:`torch-hf-bert-finetune` tutorial with latest Hugging Face Trainer API.
- Updated Neuron Runtime API guide with support for ``nr_tensor_allocate``. See :ref:`nrt-api-guide`.
- Updated :ref:`neuron-sysfs-ug` with support for ``serial_number`` unique identifier.
- Updated :ref:`custom-ops-api-ref-guide` limitations and fixed nested sublists. See :ref:`feature-custom-operators-devguide`.
- Fixed issue in :ref:`zero1-gpt2-pretraining-tutorial`.
- Fixed potential hang during synchronization step in ``nccom-test``. See :ref:`nccom-test`.
- Updated troubleshooting guide with an additional hardware error messaging. See :ref:`nrt-troubleshooting`.
- Updated DLC documentation. See :ref:`containers-dlc-then-customize-devflow` and :ref:`dlc-then-ec2-devflow`.


Neuron 2.16.0
-------------
Date: 12/21/2023

- Added setup guide instructions for ``AL2023`` OS. See :ref:`setup-guide-index`
- Added announcement for name change of Neuron Components. See :ref:`announce-component-name-change`
- Added announcement for End of Support for ``PyTorch 1.10`` . See :ref:`announce-eos_pytorch110`
- Added announcement for End of Support for ``PyTorch 2.0`` Beta. See :ref:`announce-eos_pytorch2`
- Added announcement for moving NeuronX Distributed sample model implementations. See :ref:`announce-moving-samples`
- Updated Transformers NeuronX developer guide with support for Grouped Query Attention(GQA). See :ref:`developer guide <transformers_neuronx_developer_guide>` 
- Added sample for ``Llama-2-70b`` model inference. See `tutorial <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/transformers-neuronx/inference/llama-70b-sampling.ipynb>`_ 
- Added documentation for ``PyTorch Lightning``  to train models using ``tensor parallelism`` and ``data parallelism`` . See :ref:`api guide <api_guide>` , :ref:`developer guide <ptl_developer_guide>` and :ref:`tutorial <llama2_7b_tp_zero1_ptl_tutorial>`
- Added documentation for Model and Optimizer Wrapper training API that handles the parallelization. See :ref:`api guide <api_guide>` and :ref:`model_optimizer_wrapper_developer_guide`
- Added documentation for New ``save_checkpoint``  and ``load_checkpoint`` APIs to save/load checkpoints during distributed training. See :ref:`save_load_developer_guide`
- Added documentation for a new ``Query-Key-Value(QKV)`` module in NeuronX Distributed for Training. See :ref:`api guide <api_guide>` and :ref:`tutorial <llama2_tp_pp_tutorial>`
- Added new developer guide for Inference using NeuronX Distributed. :ref:`developer guide<nxd_inference_developer_guide>`
- Added ``Llama-2-7B`` model inference script (:ref:`[html] </src/examples/pytorch/neuronx_distributed/llama/llama2_inference.ipynb>` :pytorch-neuron-src:`[notebook] <neuronx_distributed/llama/llama2_inference.ipynb>`)
- Added App note on Support for ``PyTorch 2.1`` (Beta) . See :ref:`introduce-pytorch-2-1`
- Added developer guide for ``replace_weights`` API to replace the separated weights. See :ref:`torch_neuronx_replace_weights_api` 
- Added [Beta] script for training ``stabilityai/stable-diffusion-2-1-base`` and  ``runwayml/stable-diffusion-v1-5`` models . See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/stable_diffusion/>`_ 
- Added [Beta] script for training ``facebook/bart-large`` model. See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/hf_summarization/BartLarge.ipynb>`_ 
- Added [Beta] script for ``stabilityai/stable-diffusion-2-inpainting`` model inference.  See `script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/inference/hf_pretrained_sd2_inpainting_936_624_inference.ipynb>`_ 
- Added documentation for new ``Neuron Distributed Event Tracing (NDET) tool`` to help visualize execution trace logs and diagnose errors in multi-node workloads. See :ref:`neuron-det-ug` 
- Updated Neuron Profile User guide with support for multi-worker jobs. See :ref:`neuron-profile-ug`
- Minor updates to Custom Ops API reference guide.See :ref:`custom-ops-api-ref-guide`




Neuron 2.15.0
--------------
Date: 10/26/2023

- New :ref:`introduce-pytorch-2-0` application note with ``torch-neuronx``
- New :ref:`llama2_70b_tp_pp_tutorial` and (`sample script <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/tp_pp_llama2_70b_hf_pretrain>`_) using ``neuronx-distributed``
- New :ref:`model_samples_tutorials` documentation for a consolidated list of code samples and tutorials published by AWS Neuron.
- New :ref:`sdk-classification` documentation for alpha, beta, and stable Neuron SDK definitions and updated documentation references.
- New :ref:`pipeline_parallelism_overview` and :ref:`pp_developer_guide` documentation in ``neuronx-distributed``
- Updated :ref:`Neuron Distributed API Guide <api_guide>` regarding pipeline-parallelism support and checkpointing
- New :ref:`activation_memory_reduction` application note and :ref:`activation_memory_reduction_developer_guide` in ``neuronx-distributed``
- New ``Weight Sharing (Deduplication)`` `notebook script <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/bert_tutorial/tutorial_pretrained_bert_shared_weights.ipynb>`_
- Added Finetuning script for `google/electra-small-discriminator <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/ElectraSmall.ipynb>`_ with ``torch-neuronx``
- Added `ResNet50 training (Beta) <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/resnet50/resnet50.ipynb>`_ tutorial and scripts with ``torch-neuronx``
- Added `Vision Perceiver training sample <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_image_classification/VisionPerceiverConv.ipynb>`_ with ``torch-neuronx``
- Added ``flan-t5-xl`` model inference :pytorch-neuron-src:`tutorial <neuronx_distributed/t5-inference/t5-inference-tutorial.ipynb>` using ``neuronx-distributed`` 
- Added ``HuggingFace Stable Diffusion 4X Upscaler model Inference on Trn1 / Inf2`` `sample script <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sd_x4_upscaler_inference.ipynb>`_ with ``torch-neuronx``
- Updated `GPT-NeoX 6.9B and 20B model scripts <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain>`_ to include selective checkpointing.
- Added serialization support and removed ``-O1`` flag constraint to ``Llama-2-13B`` model inference script `tutorial <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/transformers-neuronx/inference/meta-llama-2-13b-sampling.ipynb>`_ with ``transformers-neuronx``
- Updated ``BERT`` script and ``Llama-2-7B`` script with Pytorch 2.0 support
- Added option-argument ``llm-training`` to the existing ``--distribution_strategy`` compiler option to make specific optimizations related to training distributed models in :ref:`neuron-compiler-cli-reference-guide`
- Updated :ref:`neuron-sysfs-ug` to include mem_ecc_uncorrected and sram_ecc_uncorrected hardware statistics.
- Updated :ref:`torch_neuronx_trace_api` to include io alias documentation
- Updated :ref:`transformers_neuronx_developer_guide` with serialization support.
- Upgraded ``numpy`` version to ``1.22.2`` for various scripts
- Updated ``LanguagePerceiver`` fine-tuning `script <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/hf_text_classification/LanguagePerceiver.ipynb>`_ to ``stable``
- Announcing :ref:`End of Support for OPT <announce-intent-eos-opt>`  example in ``transformers-neuronx``
- Announcing :ref:`End of Support for "nemo" option-argument <announce-intent-deprecate-nemo-arg>`  

Known Issues and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Following tutorials are currently not working. These tutorials will be updated once there is a fix.

- `Zero1-gpt2-pretraining-tutorial <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/zero1_gpt2.html#zero1-gpt2-pretraining-tutorial>`_
- `Finetune t5 tutorial <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/finetune_t5.html#torch-hf-t5-finetune>`_

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
- Removed 'beta' tag from ``neuronx-distributed`` section for training. ``neuronx-distributed`` Training is now considered stable and ``neuronx-distributed`` inference is considered as beta.
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
