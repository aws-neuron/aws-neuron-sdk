.. _nxdt_config_overview:

YAML Configuration Settings
===========================

The library allows configuring a bunch of parameters in the YAML file to run large scale training.
The important categories and parameters are highlighted below. At the top level, we have the following
keys:

.. code-block:: yaml

    name:
        # Name of the experiment
    model_source:
        # Model source code, could be megatron or hf
    seed:
        # Random seed to be used for the entire experiment
    trainer:
        # Settings to configure the PyTorch-Lightning trainer
    exp_manager:
        # Settings to configure logging/checkpointing
    distributed_strategy:
        # Settings to configure how the model is to be distributed across devices
    data:
        # Settings to configure the dataset/dataloader
    model:
        # Settings to configure the model architecture and the optimizer
    precision:
        # Settings to configure the model precision
    compiler_flags:
        # Neuron compiler flags to be used
    compiler_cache_url:
        # Cache to be used to save the compiled artifacts
    aync_exec_max_inflight_requests:
        # Used to configure the runtime queue
    bucket_size_collectives:
        # Collectives are batched into tensors of this size (in MBs)
    neuron_rt_exec_timeout:
        # Runtime timeout
    neuron_experimental_compress_rg:
        # To use compress replica group


.. _nxdt_config_trainer:

Trainer
-------

Neuronx Distributed Trainer framework is built on top of `PyTorch-Lightning <https://lightning.ai/docs/pytorch/stable/>`_
and this key allows users to configure the ``trainer``.

.. code-block:: yaml

    devices: 32
    num_nodes: 1
    max_epochs: -1
    max_steps: 20000
    log_every_n_steps: 1
    val_check_interval: 20000
    check_val_every_n_epoch: null
    num_sanity_val_steps: 0
    limit_val_batches: 1
    limit_test_batches: 1
    gradient_clip_val: 1.0
    lnc: 2

.. note::

    All the above trainer parameters follow the exact same definition of the PyTorch-Lightning Trainer.
    More information about each of them can be found
    `here <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`__.

**devices**

Number of devices to be used for training. If using torchrun, this is equal to ``nproc_per_node * num_nodes``.

    * **Type**: integer
    * **Required**: True

**lnc**

Neuron-specific setting that specifies the logical-to-physical Neuron Core mapping ratio.
This parameter determines the number of physical Neuron cores used for each logical Neuron Core.

Values:

- lnc: 1 - Each node exposes 128 logical devices, with a 1:1 mapping between logical and physical Neuron Cores.
- lnc: 2 - Implements a 2:1 mapping between logical and physical Neuron Cores.

    * **Type**: integer
    * **Required**: False
    * **Default**: None (must be explicitly set)

**num_nodes**

Number of nodes to be used for training

    * **Type**: integer
    * **Required**: True

**max_epochs**

Maximum number of epochs to run. A value of ``-1`` means that the number of training steps would be inferred
from ``max_steps``

    * **Type**: integer
    * **Required**: True

**log_every_n_steps**

How often to log loss values

    * **Default value**: 1
    * **Type**: integer
    * **Required**: True

**val_check_interval**

How often to run validation step. Using this parameter one can run validation step after ``X`` training steps.

    * **Type**: integer
    * **Required**: True

**check_val_every_n_epoch**

Another parameter that controls the frequency of validation step. Using this parameter, one can run valiation
step after ``X`` epochs.

    * **Type**: integer
    * **Required**: True

**num_sanity_val_steps**

How many sanity validation steps to run. Keeping it to ``0`` would not run validation step at the start of
training.

    * **Type**: integer
    * **Required**: True


**limit_val_batches**

Number of batches to run validation step on.

    * **Type**: integer
    * **Required**: True


**gradient_clip_val**

Float value to clip gradients at.

    * **Type**: float
    * **Required**: True


.. _nxdt_config_exptm:

Experiment Manager
------------------

This setting is mainly for configuring different aspects of experiment management like checkpointing,
experiment logging directory, which parameters to log and how often to log, etc.


.. code-block:: yaml

    log_local_rank_0_only: True
    create_tensorboard_logger: True
    explicit_log_dir: null
    exp_dir: null
    name: megatron_llama
    resume_if_exists: True
    resume_ignore_no_checkpoint: True
    create_checkpoint_callback: True
    checkpoint_callback_params:
        monitor: step
        save_top_k: 1
        mode: max
        save_last: False
        filename: 'megatron_llama--{step}-{consumed_samples}'
        every_n_train_steps: 200
    log_parameter_norm: True
    log_gradient_norm: True
    enable_recovery_time_instrumentation: False
    save_xser: True
    load_xser: True
    async_checkpointing: False
    resume_from_checkpoint: null

**log_local_rank_0_only**

Log only on rank 0. The recommended setting should be ``True``

    * **Type**: bool
    * **Default**: False
    * **Required**: False

**create_tensorboard_logger**

Setting this ``True`` would log the loss and other parameters to tensorboard.

    * **Type**: bool
    * **Default**: False
    * **Required**: False

**exp_log_dir**

Explicitly specify the logging directory. Otherwise, the framework would save to current directory as default.

    * **Type**: str
    * **Default**: null
    * **Required**: False

**resume_if_exists**

Set this to ``True`` to resume from an existing checkpoint. This config will be useful when we want to
auto-resume from a failed training job.

    * **Type**: bool
    * **Default**: False
    * **Required**: False


**resume_ignore_no_checkpoint**

Experiment manager errors out if ``resume_if_exists`` is ``True`` and no checkpoint could be found. This
behaviour can be disabled, in which case exp_manager will print a message and
continue without restoring, by setting ``resume_ignore_no_checkpoint`` to ``True``.

    * **Type**: bool
    * **Default**: False
    * **Required**: False

**checkpoint_callback_params.save_top_k**

How many checkpoints to keep around. Example: If set to 1, only 1 checkpoint at any given time would be
kept around. The framework would automatically keep deleting checkpoints.

    * **Type**: int
    * **Required**: True

**checkpoint_callback_params.every_n_train_steps**

How often we want to checkpoint.

    * **Type**: int
    * **Required**: True

**log_parameter_norm**

Set this to log parameter norm across model parallel ranks.

    * **Type**: bool
    * **Default**: False
    * **Required**: False

**log_gradient_norm**

Set this to log gradient norm across model parallel ranks.

    * **Type**: bool
    * **Default**: False
    * **Required**: False

**enable_recovery_time_instrumentation**

Set this if you don’t want to default to not printing the detailing timing for recovery.

    * **Type**: bool
    * **Default**: False
    * **Required**: False

**save_xser**

Set this to save with torch xla serialization to reduce time saving, it’s recommended to enable ``xser``
for significantly faster save/load. Note that if the checkpoint is saved with ``xser``, it can only be
loaded with ``xser``, vice versa.

    * **Type**: bool
    * **Default**: False
    * **Required**: False

**load_xser**

Set this to load with torch xla serialization to reduce time saving, it’s recommended to enable ``xser`` for
significantly faster save/load. Note that if the checkpoint is saved with ``xser``, it can only be loaded
with ``xser``, vice versa.

    * **Type**: bool
    * **Default**: False
    * **Required**: False

**async_checkpointing**

Set this if you want to use async checkpointing. Under the hood the library uses the async checkpointing
feature provided by NeuronxDistributed's
`save API <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#id3>`_.

    * **Type**: bool
    * **Default**: False
    * **Required**: False

**resume_from_checkpoint**

Set this as the checkpoint file to load from. Check the SFT/DPO example config under ``conf`` on how to use it.

    * **Type**: str
    * **Default**: null
    * **Required**: False

.. _nxdt_config_distributed_strategy:

Distributed Strategy
--------------------

.. code-block:: yaml

    tensor_model_parallel_size: 8
    pipeline_model_parallel_size: 1
    virtual_pipeline_model_parallel_size: 1
    zero1: True
    sequence_parallel: True
    kv_replicator: 4

This setting allows users to configure the sharding strategy to be used for distributing the model across
workers.

**tensor_model_parallel_size**

`Tensor parallel degree <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#initialize-model-parallelism>`_
to be used for sharding models.

    * **Type**: int
    * **Required**: True

**pipeline_model_parallel_size**

`Pipeline parallel degree <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#initialize-model-parallelism>`_
to be used for sharding models.

    * **Type**: int
    * **Required**: True

**virtual_pipeline_model_parallel_size**

`Interleaved pipeline parallel degree <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#neuron-distributed-pipeline-model>`_.
Use a value of 1 if no pipeline parallelism is used.

    * **Type**: int
    * **Required**: True

**zero1**

Wraps the optimizer with zero1.

    * **Type**: bool
    * **Required**: True

**sequence_parallel**

To shard along the sequence dimension. Sequence Parallel is always used in conjuction with tensor parallel.
The sequence dimension will be sharded with the same degree as the ``tensor_model_parallel_size``.

    * **Type**: bool
    * **Required**: True

**kv_replicator**

This parameter is used together with ``qkv_linear`` parameter. It is used to configure the
`GQAQKVLinear module <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#gqa-qkv-linear-module>`_

    * **Type**: bool
    * **Required**: True

.. _nxdt_config_data:

Data
----

This is where we configure the dataset/dataloader. This config is dependent on the dataloader/dataset been
used. Users can add custom keys in this config and read inside the ``CustomDataModule`` using ``cfg.data``.
Currently the library adds support for 3 kinds of data modules: ``MegatronDataModule``, ``ModelAlignmentDataModule``
and ``HFDataModule``. To learn about the config parameters of ``MegatronDataModule`` please check the
``megatron_llama_7B_config.yaml``, for ``ModelAlignmentDataModule`` check the ``megatron_llama2_7B_SFT_config.yaml``
and for ``HFDataModule``, refer to ``hf_llama3_8B_config.yaml``.

The parameters that are common across all the configs are documented below.

.. code-block:: yaml

    micro_batch_size: 1
    global_batch_size: 1024


**micro_batch_size**

The batch is distributed across multiple data parallel ranks and within each rank, we accumulate gradients.
Micro batch size is the size that is used for each of those gradient calculation steps.

    * **Type**: int
    * **Required**: True

**global_batch_size**

This config along with micro-batchsize decides the gradient accumulation number automatically.

    * **Type**: int
    * **Required**: True

.. _nxdt_config_model:

Model
-----

This is where we can configure the model architecture. When building custom models, this config can be
used to parameterize the custom model. The below parameters are taken from an example of the Megatron
model config. Depending on the model and required parameters, this config can change.

HF Model
########

Let's start with the config for the HF model:

.. code-block:: yaml

    # model architecture
    model_config: /home/ubuntu/config.json
    encoder_seq_length: 4096
    max_position_embeddings: ${.encoder_seq_length}
    num_layers: 4
    hidden_size: 4096
    qkv_linear: False

    # Miscellaneous
    use_cpu_initialization: True

    ## Activation Checkpointing
    activations_checkpoint_granularity: selective
    activations_checkpoint_recompute: [CoreAttention]

    fusions:
        softmax: True
        flash_attention: False

    do_layer_norm_weight_decay: False

    optim:
        name: adamw_fp32OptState
        lr: 3e-4
        weight_decay: 0.01
        capturable: False
        betas:
        - 0.9
        - 0.999
        sched:
            name: LinearAnnealingWithWarmUp
            warmup_steps: 100
            max_steps: ${trainer.max_steps}

**model_config**

Points to the ``config.json`` path required by the ``transformers`` model implementation. One such example of
``config.json`` is `here <https://github.com/aws-neuron/neuronx-distributed/blob/main/examples/training/llama/tp_zero1_llama_hf_pretrain/7B_config_llama2/config.json>`__

    * **Type**: str
    * **Required**: True

**encoder_seq_length**

Setting the sequence length for the training job. This parameter is common for all models supported in the library.

    * **Type**: int
    * **Required**: True

**num_layers**

This config will override the number of layers inside the ``config.json`` in the ``model_config``. This is exposed
so that one can quickly increase/decrease the size of the model. This parameter is common for all models supported
in the library.

    * **Type**: int
    * **Required**: True

**hidden_size**

This config will override the ``hidden_size`` inside the ``config.json`` in the ``model_config``. This parameter
is common for all models supported in the library.

    * **Type**: int
    * **Required**: True

**qkv_linear**

This needs to be set if users want to use the
`GQAQKVLinear module <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#gqa-qkv-linear-module>`_

    * **Type**: bool
    * **Required**: True

**fuse_qkv**

This is set if users want to use fused q, k and v tensors in
`GQAQKVLinear module <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api-reference-guide-training.html#gqa-qkv-linear-module>`_ Using fuse_qkv can improve throughput. 
This parameter is True by default.

    * **Type**: bool
    * **Required**: False

**use_cpu_initialization**

Setting this flag to ``True`` will initialize the weights on ``CPU`` and then move to device. It is recommended to set
this flag to ``True``. This parameter is common for all models supported in the library.

    * **Type**: bool
    * **Required**: True

**activations_checkpoint_granularity**

This flag controls which module needs to be recomputed during the backward pass.

Values:

- ``selective`` - Enables selective recomputation of specified modules in `activations_checkpoint_recompute` during the backward pass.
- ``full`` - Saves activations at layer boundaries and recomputes the entire layer during the backward pass.
- ``null`` - Disables activation checkpointing.

More information on activation recompute can be found
`in this link <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/activation_memory_reduction.html#activation-recomputation>`_.
This parameter is common for all models supported in the library.

    * **Type**: str
    * **Possible Values**: ``selective``, ``full``, ``null``
    * **Required**: True

**activations_checkpoint_recompute**
This config specifies which modules to recompute when using ``selective`` activation checkpointing.
It accepts a list of module names as strings or `null`.

    * **Type**: list[str] or `null`
    * **Required**: False

**fusions.softmax**

Setting this flag to ``True`` will replace the ``torch.nn.Softmax`` with a fused custom ``Softmax`` operator. This
parameter is common for all models supported in the library.

    * **Type**: bool
    * **Required**: True

**fusions.flash_attention**

Setting this flag to ``True`` will insert the flash attention module for both forward and backward. This parameter is
common for all models supported in the library.

    * **Type**: bool
    * **Required**: True

**fusions.do_layer_norm_weight_decay**

Setting this flag to ``True`` will add layer norm weight decay. This parameter is common for all models supported in
the library.

    * **Type**: bool
    * **Required**: True

**optim**

This is where the optimizers can be set. Since the library is built using ``NeMo``, we can configure the optimizers
supported by ``NeMo``. All the optimzers can be configured according to the
`parameters specified here <https://github.com/NVIDIA/NeMo/blob/v1.14.0/nemo/core/config/optimizers.py>`__.

    * **Type**: config
    * **Possible Values**: ``adamw``, ``adamw_fp32OptState``, ``sgd``, ``adam``, ``adadelta``, ``adamax``,
    *  ``adagrad``, ``rmsprop``, ``rprop``, ``novograd``, ``adafactor``
    * **Required**: True

**optim.sched**

This is where the LR schedulers can be set. Since the library is built using ``NeMo``, we can configure the schedulers
supported by ``NeMo``. All the schedulers can be configured according to the
`parameters specified here <https://github.com/NVIDIA/NeMo/blob/v1.14.0/nemo/core/config/schedulers.py>`__.

    * **Type**: config
    * **Possible Values**: ``LinearAnnealingWithWarmUp``, ``CosineAnnealing``, ``WarmupPolicy``,
    *  ``WarmupHoldPolicy``, ``SquareAnnealing``, ``NoamAnnealing``, ``WarmupAnnealing``,
    *   ``StepLR``, ``rprop``, ``ExponentialLR``
    * **Required**: True

Megatron Model
##############

The library enables a
`megatron transformer <https://github.com/NVIDIA/NeMo/blob/v1.14.0/nemo/collections/nlp/models/language_modeling/megatron/gpt_model.py>`_
model which can be configured from the yaml file. The different available parameters are documented below after
the following reference example.

.. code-block:: yaml

    # model architecture
    encoder_seq_length: 4096
    max_position_embeddings: ${.encoder_seq_length}
    num_layers: 32
    hidden_size: 4096
    ffn_hidden_size: 11008
    num_attention_heads: 32
    num_kv_heads: 32
    init_method_std: 0.021
    hidden_dropout: 0
    attention_dropout: 0
    ffn_dropout: 0
    apply_query_key_layer_scaling: True
    normalization: 'rmsnorm'
    layernorm_epsilon: 1e-5
    do_layer_norm_weight_decay: False # True means weight decay on all params
    make_vocab_size_divisible_by: 8 # Pad the vocab size to be divisible by this value for computation efficiency.
    persist_layer_norm: True # Use of persistent fused layer norm kernel.
    share_embeddings_and_output_weights: False # Untie embedding and output layer weights.
    position_embedding_type: 'rope' # Position embedding type. Options ['learned_absolute', 'rope]
    rotary_percentage: 1 # If using position_embedding_type=rope, then the per head dim is multiplied by this.
    activation: 'swiglu' # ['swiglu', 'gelu']
    has_bias: False
    # Miscellaneous
    use_cpu_initialization: True

    ## Activation Checkpointing
    activations_checkpoint_granularity: selective # 'selective' or 'full'

    fusions:
        softmax: True
        flash_attention: False # Use NKI flash attention

    optim:
        name: adamw
        lr: 3e-4
        weight_decay: 0.1
        capturable: True
        betas:
        - 0.9
        - 0.95
        sched:
        name: CosineAnnealing
        warmup_steps: 2000
        constant_steps: 0
        min_lr: 3.0e-5

.. note::

    For common config, please refer to the ``HF Model`` section above.

**ffn_hidden_size**

Transformer FFN hidden size.

    * **Type**: int
    * **Required**: True

**num_attention_heads**

Number of ``Q`` attention heads.

    * **Type**: int
    * **Required**: True

**num_kv_heads**

Number of ``KV`` heads. This is where we can configure ``Q`` and ``KV`` differently to create ``GQA`` modules.

    * **Type**: int
    * **Required**: True

**init_method_std**

Standard deviation to use when we init layers of the transformer model.

    * **Type**: float
    * **Required**: True

**hidden_dropout**

Dropout probability for hidden state transformer.

    * **Type**: float
    * **Required**: True

**attention_dropout**

Dropout probability in the attention layer.

    * **Type**: float
    * **Required**: True

**ffn_dropout**

Dropout probability in the feed-forward layer.

    * **Type**: float
    * **Required**: True

**apply_query_key_layer_scaling**

Scale ``Q * K^T`` by ``(1 / layer-number)``.

    * **Type**: bool
    * **Required**: True

**normalization**

Normalization layer to use.

    * **Type**: str
    * **Possible Values**: ``rmsnorm``, ``layernorm``
    * **Required**: True

**layernorm_epsilon**

Epsilon value for layernorm.

    * **Type**: float
    * **Required**: True

**share_embeddings_and_output_weights**

Setting this parameter to ``True`` will tie the ``vocab embedding`` weight with the final ``MLP`` weight.

    * **Type**: bool
    * **Required**: True

**make_vocab_size_divisible_by**

So lets say your vocab size is ``31999`` and you set this value to 4, the framework would pad the vocab-size such that
it becomes divisible by ``4``. In this case the close divisible value is ``32K``.

    * **Type**: int
    * **Required**: True

**position_embedding_type**

Type of position embedding to be used.

    * **Type**: str
    * **Possible Values**: ``learned_absolute``, ``rope``
    * **Required**: True

**rotary_percentage**

If using ``position_embedding_type=rope``, then the per head dim is multiplied by this factor.

    * **Type**: float
    * **Required**: True

**activation**

Users can specify the activation function to be used in the model.

    * **Type**: str
    * **Possible Values**: ``swiglu``, ``gelu``
    * **Required**: True

**has_bias**

Setting this parameter to ``True`` will add bias to each of the linear layers in the model.

    * **Type**: bool
    * **Required**: True


.. _nxdt_config_overview_precision_config:

Precision
---------

This config can help to decide the dtype of the model/optimizer.

.. code-block:: yaml

    precision:
        type: 'mixed_precision' # ['bf16SR', 'fp32', 'autocast', 'mixed_precision', 'mixed_precisionSR', 'manual']
        # Set the following only if precision type is manual, otherwise they will be automatically set.
        master_weights: False
        fp32_grad_acc: False
        xla_use_bf16: '0'
        xla_downcast_bf16: '0'
        neuron_rt_stochastic_rounding_en: '0'

.. note::

    Only if the precision type is ``manual``, ``master_weights`` , ``fp32_grad_acc``, ``xla_use_bf16``, ``xla_downcast_bf16``,
    ``neuron_rt_stochastic_rounding_en`` will be picked up from the config. These parameters are for more finer control of
    precision. It is recommended to use ``mixed_precision`` config for better accuracy.

**mixed_precision**

This config will use the ``zero1`` optimizer and will keep master weights in ``fp32``. It will also perform grad
accumulation and ``grad cc`` in ``fp32``. It will also set the ``xla_downcast_bf16``. It will disable stocastic
rounding.

**mixed_precisionSR**

This is a superset config of ``mixed_precision``, only difference been the stochastic rounding. In this case, we set the
stochastic rounding.


**bf16SR**

This config will perform all operations in ``bf16``. It will rely on the stochastic rounding feature to gain accuracy.


**autocast**

This config will follow the exact same precision strategy followed by ``torch.autocast``.

.. note::
    Autocast is not supported in this release.

**manual**

To gain control of the different precision nobs, one can set the precision type to ``manual`` and control parameters
like - ``master_weights`` , ``fp32_grad_acc``, ``xla_use_bf16``, ``xla_downcast_bf16`` and
``neuron_rt_stochastic_rounding_en``.


Model Alignment Specific
------------------------

You can configure a finetuning (SFT) or model alignment (DPO) through the YAML file.

.. code-block:: yaml

    data:
        train_dir: /example_datasets/llama3_8b/training.jsonl
        val_dir: null
        dev_choose_samples: 2250
        seq_length: 4096

        alignment_strategy:
            # DPO specific config
            dpo:
                kl_beta: 0.01
                loss_type: sigmoid
                max_dpo_prompt_length: 2048
                precompute_ref_log_probs: True
                truncation_mode: keep_start

            # Alternatively, can also use SFT specific config
            sft:
                packing: True

    model:
        weight_init_only: True

**data**
    **train_dir**

    SFT/DPO training data - jsonl or arrow file

    As for SFT/DPO we use HF style ModelAlignment dataloader, we also use HF style data file paths

        * **Type**: str
        * **Required**: True

    **val_dir**

    SFT/DPO validation data - jsonl or arrow file

    As for SFT/DPO we use HF style ModelAlignment dataloader, we also use HF style data file paths

        * **Type**: str
        * **Required**: False

    **dev_choose_samples**

    If set, will use that many number of records from the
    head of the dataset instead of using all. Set to null to use full dataset

        * **Type**: integer
        * **Default**: null
        * **Required**: False

    **seq_length**

    Set sequence length for the training job.
    For DPO, it is total sequence length of prompt and (chosen/rejected) response concatenated together.

        * **Type**: integer
        * **Required**: True

    **alignment_strategy**

    Set only when using finetuning specific algorithms (SFT, DPO, etc) and related hyperparameters
    DPO-specific parameters.

        **dpo**
            **kl_beta**

            KL-divergence beta to control divergence of policy model from reference model

                * **Type**: float
                * **Default**: 0.01
                * **Required**: True

            **loss_type**

            Currently support sigmoid version of optimized DPO loss

                * **Type**: str
                * **Default**: ``sigmoid``
                * **Required**: True

            **max_dpo_prompt_length**

            Set maximum length of prompt in the concatenated prompt and (chosen/rejected) response input

                * **Type**: integer
                * **Required**: True

            **precompute_ref_log_probs**

            To enable precomputation of reference model log probabilities using pre-fit hook,
            False is not supported currently

                * **Type**: bool
                * **Required**: True

            **truncation_mode**

            To define how to truncate if size (prompt+response) exceeds seq_length
            options: ["keep_start", "keep_end"]

                * **Type**: str
                * **Default**: ``keep_start```
                * **Required**: True

    SFT-specific parameters.

        **sft**
            **packing**

            Appends multiple records in a single record until seq length
            supported by model, if false uses pad tokens to reach seq length.
            Setting it to True increases throughput but might impact accuracy.

                * **Type**: bool
                * **Default**: False
                * **Required**: False