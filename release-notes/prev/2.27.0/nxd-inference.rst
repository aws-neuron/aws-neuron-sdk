.. _neuron-2-27-0-nxd-inference:

.. meta::
   :description: The official release notes for the AWS Neuron SDK Transformers for Inference component, version 2.27.0. Release date: 12/19/2025.

AWS Neuron SDK 2.27.0: NxD Inference release notes
==================================================

**Date of release**: December 19, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.27.0 release notes home <neuron-2-27-0-whatsnew>`

What's New
----------

**Trn3 Platform Support** — Added support for running NxD Inference on Trn3 instances.

**vLLM V1 support** - This release adds support for vLLM V1 through  `vllm-neuron <https://github.com/vllm-project/vllm-neuron>`_ plugin. You can use the vLLM V1 by using the new vLLM V1 based Neuron DLC or using the vLLM virtual environment in Neuron DLAMIs. See :ref:`vLLM V1 guide <nxdi-vllm-user-guide-v1>` for more information.

**Qwen3 MoE Model Support (Beta)** — NxD Inference supports Qwen3 MoE language model which supports multilingual text inputs. You can use HuggingFace checkpoint. For more information about how to run Qwen3 MoE inference, see :doc:`Tutorial: Qwen3 MoE Inference </libraries/nxd-inference/tutorials/qwen3-moe-tutorial>`.

Compatible models include:

* `Qwen3-235B-A22B <https://huggingface.co/Qwen/Qwen3-235B-A22B>`_

**Pixtral Model Support (Beta)** — NxD Inference supports Pixtral image understanding model which processes text and image inputs. You can use HuggingFace checkpoint. For more information about how to run Pixtral inference, see :doc:`Tutorial: Deploy Pixtral Large on Trn2 instances </libraries/nxd-inference/tutorials/pixtral-tutorial>`.

Compatible models include:

* `Pixtral-Large-Instruct-2411 <https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411>`_

Known Issues
------------

* Pixtral deployment is supported up to batch size 32 and sequence length 10240 with vLLM v0. vLLM v1 deployment supports up to batch size 4 and sequence length 10240.
* The performance of Qwen3 MoE and Pixtral on Trn2 is not fully optimized. We will address the issues in the future release.
* The vllm-neuron plugin source code in github is currently not compatible with 2.27 SDK. Customers are advised to use inference DLAMI and DLC published with 2.27.0 SDK for vLLN V1 support. vllm-neuron github repo source code
  will be updated soon to be compatible with 2.27 release SDK.

