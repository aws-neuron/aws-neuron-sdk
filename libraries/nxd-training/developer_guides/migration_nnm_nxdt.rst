.. _nxdt_developer_guide_migration_nnm_nxdt:

Migrating from Neuron-NeMo-Megatron to Neuronx Distributed Training
====================================================================

In this section, we go over the changes one would have to make if they are migrating their
training workload from
`Neuronx-NeMo-Megatron (NNM) <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nemo-megatron/index.html>`_
to Neuronx Distributed Training (NxDT) framework.

.. contents:: Table of contents
   :local:
   :depth: 2

Config migration
----------------

NxDT is a framework built on top of `NeMo <https://github.com/NVIDIA/NeMo>`_ and
`NeuronxDistributed (NxD) <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/index.html>`_
and supports megatron-style model. The megatron model implementation is ported over from NNM.
Hence, most of the config YAMLs from NNM can be migrated to use NxDT.

When building NxDT for the sake of modularity, we grouped certain parameters together, eg.
:ref:`distributed_strategy<nxdt_config_distributed_strategy>` has all the configuration for model parallelism,
:ref:`data<nxdt_config_data>` config now holds all the parameters required to configure
the dataset.

At a high level, there are some differences with the NNM config, which are highlighted below:

#.
    The overall config structure has changed. For simplicity and ease of understanding, the config parameters
    are grouped according to their high level use case. For example, previously all the distributed config parameters
    used to reside inside ``model`` config, now it’s been moved to a ``distributed_config`` of its own. Similarly data
    config is moved out to have clear separation between model and data.

#.
    Environment variables like ``neuron_cc_flags``  and ``neuron_compile_cache_url`` can be set from the config
    itself. There is no need to set the environment variables. The rationale is to avoid having to configure training
    scripts from multiple places.

#.
    ``Activation Checkpointing:`` NxDT only supports selective and full activation checkpointing. The ``selective``
    checkpointing is done only for the ``CoreAttention`` block (in case of llama3-8K we recompute the ``MLP``
    block, too) and ``full`` activation checkpointing is done only at a layer boundary. NxDT doesn’t support
    config parameters like ``activations_checkpoint_method``, ``activations_checkpoint_num_layers``,
    ``num_micro_batches_with_partial_activation_checkpoints``, ``activations_checkpoint_layers_per_pipeline``,
    ``disable_layer_norm_checkpointing``. Please remove these parameters from your config.yaml file.

.. note::

    If you plan to add more modules that need to be recomputed, one would have to override the checkpointing config inside
    ``ModelModule`` (refer to ``build_model`` API at :ref:`nxdt_developer_guide_integrate_new_model_build_module`)
    and add the modules that need to be recomputed.

4.
    ``Tokenizer:`` The tokenizer which used to reside under ``model`` is now moved to ``data``. This is done so that all
    data related configuration can reside at one place.

#.
    ``accumulate_grad_batches:`` This param is removed since it should always be 1. Gradient accumulation is handled by
    setting the global_batch_size and micro_batch_size along with data-parallel degree.

#.
    ``pre_process and post_process:``: These two parameters were added to the model to decide if the embedding lookup
    needs to be added at the start and if a ``pooler`` layer needs to be added at the end. This has been set by default
    for all decoder models and hence the config param is no longer exposed.

#.
    ``Mixed precision config:`` NxDT no longer exposes NeMo mixed precision parameters: ``native_amp_init_scale``,
    ``native_amp_growth_interval``, ``hysteresis``, ``fp32_residual_connection``, ``fp16_lm_cross_entropy``. All these
    parameters are specific to the GPU mixed precision strategy, which Neuron doesn’t support, or they are not
    applicable. Neuron has a different way to enable mixed precision training through ``master_weights`` and
    ``fp32_grad_accumulation``.


#.
    ``megatron_amp_o2:`` This parameter is not supported.

#.
    ``Fusions:`` Neuron doesn’t support fusion parameters like ``grad_div_ar_fusion``, ``gradient_accumulation_fusion``,
    ``bias_activation_fusion``, ``bias_dropout_add_fusion``, ``masked_softmax_fusion``. All of these fusions are built
    for GPU and require CUDA kernels which cannot run on Trn1. Neuron would have its own set of kernels and when we
    support them, we would enable those parameters from the config.

.. note::

    If there is a need to support these configs, please create a feature request with exact needs and we shall work on it.

For detailed mapping, please check the :ref:`nxdt_nnm_nxdt_config_mapping`.

Model code
----------

There are the following differences in the model code:

#.
    NNM used `Apex <https://github.com/NVIDIA/apex/tree/master>`_ to get all the distributed parallel layers and schedules.
    Since NxDT uses NxD as the base library, all the
    `parallel layers/parallel state <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#parallel-layers>`_
    are coming from NxD. Eg. `apex.parallel_state <https://github.com/NVIDIA/apex/blob/master/apex/transformer/parallel_state.py>`_
    is replaced with
    `nxd.parallel_layers.parallel_state <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#parallel-model-state>`_.

#.
    NNM explicitly creates a module for each pipeline-parallel (PP) rank, however, NxDT uses NxD which does the
    partitioning under the hood. Hence, users no longer have to worry about creating a rank specific module.
    They can create one single model and
    `NxD’s PP wrapper <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#neuron-distributed-pipeline-model>`_
    takes care of sharding for each PP rank. Hence, all the code related to pipeline parallelism inside model
    code is removed. The model code assumes there is no PP and just uses TP layers from NxD.

.. note::
    For the tracer to work efficiently, we configure the pipeline parallel config inside the ``BaseModelModule`` class inside
    ``lightning_modules/model``.

3.
    In NNM, megatron module had to explicitly handle gradient reduction for shared weights across PP ranks. In NxDT,
    since we are using NxD’s PP wrapper, all that is handled for the user.

#.
    For activation checkpointing, NNM had explicit recompute functions which handled the
    `custom forward API <https://github.com/aws-neuron/neuronx-nemo-megatron/blob/main/nemo/nemo/collections/nlp/modules/common/megatron/transformer.py>`_.
    With NxDT, `NxD’s Activation Checkpoint wrapper <https://github.com/aws-neuron/neuronx-distributed/blob/main/src/neuronx_distributed/utils/activation_checkpoint.py>`_
    handles the recompute of the modules. Users just have to configure the ``activation_checkpoint_config`` inside
    ``nxd_config``
    `here <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#initialize-nxd-config>`__.


Checkpointing Save/Load
-----------------------

NxDT supports all the checkpointing features which NNM supports. This includes async checkpointing, auto-resume, etc.
There are some differences in the format of the checkpoint. This is because NxDT uses
`NxD’s checkpoint api <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#save-checkpoint>`_.
The key differences are listed below:

#.
    NNM combines the model weights, optimizers and other state_dicts into a single ``state_dict`` and dump a file
    of the format: ``tp_rank_0*_pp_rank_00*/model_optim_rng.ckpt``. However, with NxDT, we save the model ``state_dict``
    and the optimizer separately. The model ``statedict`` is saved in a folder of the form:
    ``model/dp_rank_00_tp_rank_00_pp_rank_00.pt`` and the optimizer is saved into a separate folder as:
    ``optim/dp_rank_00_tp_rank_00_pp_rank_00.pt``. This is mainly done so that when we use
    `zero1 <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html?highlight=zero1#neuron-zero1-optimizer>`_,
    each DP rank can save its own optimizer shard.

#.
    In NNM, if we are using pipeline parallelism, each pipeline stage creates an independent model. So lets say we have
    a model with 32 layers and we use PP=4, then NNM would create 4 chunks with layers 0-7. So each PP rank would have
    ``model_state_dict`` with keys going from layer-0-7. However, in NxDT, the model is created as a whole and then
    sharded. So the layer numbers are preserved.

#.
    There are checkpoint conversion scripts provided under ``examples/`` of NxDT repository to convert the existing NNM
    checkpoints to NxDT format in case of migrating in the middle of training.

.. code-block:: shell

    python nnm_nxdt_ckpt_converter.py --tp 8 --pp 4 --n_layers 32 --nnm_ckpt_path {path_to_ckpt}/ckpt/nnm --nxdt_ckpt_path {path_to_ckpt}/nnm-converted-nxdt-ckpt/ --enable_parallel_processing True --num_parallel_processes 8

.. _nxdt_nnm_nxdt_config_mapping:

Config Mapping
--------------

Here is a detailed mapping for all the parameters in the config file. For the below mapping, we chose the
Llama-7B example across NNM and NxDT frameworks. The same mapping is also true for other models.

.. csv-table::
   :file: nnm_nxdt_mapping.csv
   :header-rows: 1
   :widths: 20, 20, 40

.. note::

    For parameters that are not supported by NxDT, please create a feature request with specific use-case
    for the parameter, if needed.
