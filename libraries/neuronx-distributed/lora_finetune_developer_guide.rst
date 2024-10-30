
.. _lora_finetune_developer_guide:

Developer guide for LoRA finetuning
===================================

This document will introduce how to enable model finetuning with LoRA.

For a complete api guide, refer to :ref:`API <api_guide>`.

Enable LoRA finetuning:
'''''''''''''''''''''''

We first set up LoRA-related configurations:

.. code:: ipython3

    lora_config = nxd.modules.lora.LoraConfig(
        enable_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        lora_verbose=True,
        target_modules=["q_proj", "v_proj", "k_proj"],
        save_lora_base=False,
        merge_lora=False,
    )


The default target modules for different model architectures can be found in `model.py <https://github.com/aws-neuron/neuronx-distributed/blob/main/src/neuronx_distributed/modules/lora/model.py>`_.


We then initialize NxD model with LoRA enabled:

.. code:: ipython3

    nxd_config = nxd.neuronx_distributed_config(
        ...
        lora_config=lora_config,
    )
    model = nxd.initialize_parallel_model(nxd_config, ...)


Save LoRA checkpoint
''''''''''''''''''''

Users can save the LoRA adapter with

.. code:: ipython3

    nxd.save_checkpoint(
        checkpoint_dir_str=checkpoint_dir, # checkpoint path
        tag=tag,     # sub-directory under checkpoint path
        model=model
    )


Because ``save_lora_base=False`` and ``merge_lora=False``, only the LoRA adapter is saved under ``checkpoint_dir/tag/``.
We can also set ``merge_lora=True`` to save the merged model, i.e., merging LoRA adapter into the base model.


Load LoRA checkpoint:
''''''''''''''''''''''

A sample usage:

.. code:: ipython3

    lora_config = LoraConfig(
        enable_lora=True,
        load_lora_from_ckpt=True,
        lora_save_dir=checkpoint_dir,  # checkpoint path
        lora_load_tag=tag,  # sub-directory under checkpoint path
    )
    nxd_config = nxd.neuronx_distributed_config(
        ...
        lora_config=lora_config,
    )
    model = nxd.initialize_parallel_model(nxd_config, ...)
   
   
The NxD model with be initialized with LoRA enabled and LoRA weights loaded. LoRA-related configurations are the same as the LoRA adapter checkpoint.