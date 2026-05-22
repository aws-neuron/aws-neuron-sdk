.. meta::
   :description: Archived AWS Neuron SDK documentation
   :keywords: AWS Neuron SDK, archived tutorials, legacy documentation
   :date-modified: 2026-05-11

=====================================
Archived AWS Neuron SDK documentation
=====================================

.. note::

    This page contains archived tutorials and other documentation for older versions of the AWS Neuron SDK.
    These pages are no longer actively maintained and may reference unsupported features or deprecated APIs. They are provided as-is and may not reflect the current state of the AWS Neuron SDK.

Overview
--------

The following content has been archived for reference purposes. For the latest documentation and guides, visit the `AWS Neuron SDK documentation <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/>`_.

Archived feature docs
---------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Last release supported
     - Date archived
   * - :doc:`profiler/index`
     - Neuron 2.28.0
     - Archived on: 5/11/2026
   * - :doc:`tensorboard/index`
     - Neuron 2.28.0 (Trn1 plugin); Neuron 2.27.0 (Inf1 plugin)
     - Archived on: 5/11/2026
   * - :doc:`neuronperf/index`
     - Neuron 2.27.0
     - Archived on: 12/2/2025
   * - :doc:`helper-tools/index`
     - Neuron 2.27.0
     - Archived on: 12/2/2025
   * - :doc:`transformers-neuronx/index`
     - Neuron 2.25.0
     - Archived on: 9/15/2025
   * - :doc:`MXNet Neuron Setup Guides <mxnet-neuron/index>`
     - Neuron 2.27.0
     - Archived on: 3/30/2026
   * - :doc:`mxnet-neuron/index`
     - Neuron 2.16.0
     - Archived on: 3/11/2026
   * - :doc:`tensorflow/index`
     - Neuron 2.22.0
     - Archived on: 3/11/2026
   * - :doc:`torch-neuron/index`
     - Neuron 2.22.0
     - Archived on: 3/11/2026

Archived deployment flows
--------------------------

The following deployment flow pages were archived during the consolidation of ``/containers/``, ``/devflows/``, and ``/dlami/`` into ``/deploy/``. These pages reference legacy patterns (Inf1, Neo compilation, BYOC hosting) that have been superseded by current deployment guides.

.. list-table::
   :header-rows: 1

   * - Page
     - Reason archived
     - Date archived
   * - :doc:`devflows/inference/byoc-hosting-devflow`
     - Legacy BYOC hosting flow
     - Archived on: 4/20/2026
   * - :doc:`devflows/inference/byoc-hosting-devflow-inf2`
     - Legacy BYOC hosting flow
     - Archived on: 4/20/2026
   * - :doc:`devflows/inference/container-sm-hosting-devflow`
     - Legacy SageMaker hosting flow
     - Archived on: 4/20/2026
   * - :doc:`devflows/inference/neo-then-hosting-devflow`
     - Legacy Neo compilation flow
     - Archived on: 4/20/2026
   * - :doc:`devflows/inference/dlc-then-k8s-devflow`
     - Superseded by EKS-specific guides
     - Archived on: 4/20/2026
   * - :doc:`containers/tutorial-docker-runtime1.0`
     - Legacy runtime 1.0
     - Archived on: 4/20/2026


Archived setup guides
---------------------

.. list-table::
   :header-rows: 1

   * - Page
     - Reason archived
     - Date archived
   * - :doc:`setup/setup-rocky-linux-9`
     - Rocky Linux 9 is no longer a supported install target
     - Archived on: 5/15/2026

Archived tutorials
------------------

.. list-table::
   :header-rows: 1

   * - Tutorial
     - Last release supported
     - Date archived
   * - :doc:`tutorials/finetune_t5`
     - Neuron 2.24.0
     - Archived on: 7/31/2025
   * - :doc:`tutorials/ssd300_demo/ssd300_demo`
     - Neuron 2.24.0
     - Archived on: 7/31/2025
   * - :doc:`tutorials/megatron_gpt_pretraining`
     - Neuron 2.25.0
     - Archived on: 7/31/2025
   * - :doc:`tutorials/finetuning_llama2_7b_ptl`
     - Neuron 2.26.0
     - Archived on: 8/25/2025
   * - :doc:`tutorials/training_llama2_tp_pp_ptl`
     - Neuron 2.26.0
     - Archived on: 8/25/2025
   * - :doc:`tutorials/training_codegen25_7b`
     - Neuron 2.26.0
     - Archived on: 8/25/2025
   * - :doc:`tutorials/gpt3_neuronx_nemo_megatron_pretraining`
     - Neuron 2.26.0
     - Archived on: 8/25/2025
   * - :doc:`tutorials/multinode-training-model-profiling`
     - Neuron 2.29.0
     - Archived on: 3/30/2026
   * - :doc:`tensorboard/torch-neuronx-profiling-with-tb`
     - Neuron 2.28.0
     - Archived on: 5/11/2026
   * - :doc:`tensorboard/tutorial-tensorboard-scalars-mnist`
     - Neuron 2.28.0
     - Archived on: 5/11/2026

.. toctree::
    :maxdepth: 1
    :hidden:

    tutorials/finetune_t5
    tutorials/ssd300_demo/ssd300_demo
    tutorials/megatron_gpt_pretraining
    tutorials/training-gpt-neox-20b
    tutorials/finetuning_llama2_7b_ptl
    tutorials/training_llama2_tp_pp_ptl
    tutorials/training_codegen25_7b
    tutorials/multinode-training-model-profiling
    tutorials/training-gpt-neox
    tensorboard/getting-started-tensorboard-neuron-plugin
    tensorboard/index
    neuronperf/index
    helper-tools/index
    transformers-neuronx/index
    mxnet-neuron/index
    tensorflow/index
    torch-neuron/index
    devflows/inference/byoc-hosting-devflow
    devflows/inference/byoc-hosting-devflow-inf2
    devflows/inference/container-sm-hosting-devflow
    devflows/inference/neo-then-hosting-devflow
    devflows/inference/dlc-then-k8s-devflow
    containers/tutorial-docker-runtime1.0
    setup/setup-rocky-linux-9

Accessing Archived Content
--------------------------

Each tutorial listed above corresponds to a specific version or feature set of the Neuron SDK that has since been superseded. Use these resources for historical context or migration guidance.

.. warning::

    Archived tutorials may not be compatible with current Neuron SDK releases. Exercise caution when following instructions from these documents.
