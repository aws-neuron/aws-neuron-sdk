.. _llama3_8b_tp_ptl_lora_finetune_tutorial:

Fine-tuning Llama3 8B with tensor parallelism and LoRA using Neuron PyTorch-Lightning
=====================================================================================

This tutorial shows how to fine-tune a Llama3-8B model with tensor-parallelism and LoRA adaptors. The tutorial uses the :ref:`PyTorch-lightning trainer <ptl_developer_guide>` for setting up the finetuning loop.


Setting up the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

For this experiment, we will use one trn1.32xlarge compute instance in AWS EC2.
To set up the packages in the compute instance, see
:ref:`Install PyTorch Neuron on Trn1 <setup-torch-neuronx>`.
Install the ``neuronx-distributed`` package inside the virtual environment using the following command:

.. code-block:: ipython3
   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

Next, download the scripts for fine-tuning with LoRA

1. Create a directory to hold the experiments.

.. code-block:: ipython3

   mkdir -p ~/examples/tp_llama3_8b_lora_finetune
   cd ~/examples/tp_llama3_8b_lora_finetune


2. Download training scripts for the experiments.


We download training scripts for Llama modules, data modules, the config file of Llama3-8B, and the LoRA fine-tuning script from NxD.
We also download the requirements files for package dependencies and scripts to convert Llama checkpoint to NxD checkpoint.

.. code-block:: ipython3

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/data_module.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/module_llama.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/tp_llama_hf_finetune_ptl.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/tp_zero1_llama_hf_pretrain/8B_config_llama3/config.json
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lr.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/modeling_llama_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/requirements.txt
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/requirements_ptl.txt
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/training_utils.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/convert_checkpoints.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/test/integration/modules/lora/test_llama_lora_finetune.sh
   wget https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/convert_llama_weights_to_hf.py


3. Install the additional requirements and give the right permissions to the shell script.

.. code-block:: ipython3

   python3 -m pip install -r requirements.txt
   python3 -m pip install -r requirements_ptl.txt  # Currently we're supporting Lightning version 2.4.0
   chmod +x test_llama_lora_finetune.sh
   # prepare the dataset
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab');" 


Prepare the checkpoint and dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


1. Download the Llama3-8B checkpoint

Use of this model is governed by the Meta license. In order to download the model weights and tokenizer follow the instructions in meta-llama/Meta-Llama-3-8B .

Once granted access, you can download the model. For the purposes of this tutorial we assume you have saved the Llama-3-8B model in a directory called ``models/Llama-3-8B``

2. Convert the llama checkpoint to NxD checkpoint

Use `convert_llama_weights_to_hf.py <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py>`_ to convert Llama checkpoint to HuggingFace checkpoint. 
This script will shard Llama3-8B into multiple partitions.
In order to save it as one partition, we need to set flags ``max_shard_size="64GB"`` and ``safe_serialization=False`` in ``model.save_pretrained()``.

.. code-block:: ipython3

   pip install blobfile tiktoken
   cd ~/examples/tp_llama3_8b_lora_finetune
   python convert_llama_weights_to_hf.py --input_dir models/Llama-3-8B/ --model_size 8B --llama_version 3 --output_dir models/Llama-3-8B-hf


When the HuggingFace checkpoint is ready, we can convert it to NxD checkpoint with

.. code-block:: ipython3

   cd ~/examples/tp_llama3_8b_lora_finetune
   python3 convert_checkpoints.py --tp_size 32 --qkv_linear 1 --kv_size_multiplier 4 --convert_from_full_state --config config.json --input_dir models/Llama-3-8B-hf/pytorch_model.bin --output_dir models/llama3_8b_tp32/pretrained_weight/


We then set up `PRETRAINED_PATH="models/llama3_8b_tp32"` in `tp_llama3_8b_lora_finetune_ptl.sh`.


3. Set up HuggingFace Token for Llama3 Tokenizer

We need to set up ``HF_TOKEN`` in ``test_llama_lora_finetune.sh`` to configure your Huggingface Token for Llama3-8B Tokenizer.

Refer to `Huggingface Access Tokens <https://huggingface.co/docs/hub/en/security-tokens>`_ to create your Huggingface access tokens.


1. Set the dataset for the fine-tuning job. 

In this example, we will use `Dolly <https://huggingface.co/datasets/databricks/databricks-dolly-15k>`_, which is an open source dataset
of instruction-following records on categories outlined in the `InstructGPT paper <https://arxiv.org/pdf/2203.02155>`_, including brainstorming, classification,
closed QA, generation, information extraction, open QA, and summarization.

{
  "instruction": "Alice's parents have three daughters: Amy, Jessy, and what's the name of the third daughter?",

  "context": "",

  "response": "The name of the third daughter is Alice"
}

Configure the following flags in ``test_llama_lora_finetune.sh`` to set up the dataset:

.. code-block:: ipython3

   --data_dir "databricks/databricks-dolly-15k" \
   --task "open_qa" \


Running fine-tuning
^^^^^^^^^^^^^^^^^^^

1. Enable LoRA for fine-tuning 

In ``test_llama_lora_finetune.sh``, we also need to enable LoRA by adding the below argument

.. code-block:: ipython3

   --enable_lora \


The default configuration for LoRA adapters in ``test_llama_lora_finetune.py`` is

.. code-block:: ipython3

   target_modules = ["q_proj", "v_proj", "k_proj"] if flags.qkv_linear == 0 else ["qkv_proj"]      
   lora_config = LoraConfig(
      enable_lora=flags.enable_lora,
      lora_rank=16,
      lora_alpha=32,
      lora_dropout=0.05,
      bias="none",
      lora_verbose=True,
      target_modules=target_modules,
   )


1. LoRA checkpoint

There are three checkpoint saving modes for LoRA fine-tuning and we can set different modes with LoRA flags ``save_lora_base`` and ``merge_lora``

* ``save_lora_base=False, merge_lora=False`` Save the LoRA adapter only.
* ``save_lora_base=True, merge_lora=False``  Save both the base model and the LoRA adapter seperately.
* ``save_lora_base=True, merge_lora=True``   Merge the LoRA adapter into the base model and then save the base model.


Other than the adapter, LoRA also needs to save the LoRA configuration file for adapter loading. 
The configuration can be saved into the same checkpoint with the adapter, or saved as a seperately json file.
An example of configurations for LoRA saving is

.. code-block:: ipython3

   lora_config = LoraConfig(
      ...
      save_lora_base=False,   # save the LoRA adapter only
      merge_lora=False,       # do not merge LoRA adapter into the base model
      save_lora_config_adapter=True,  # save LoRA checkpoint and configuration file in the same checkpoint
   )


After adding these flags, users can save LoRA model with 

.. code-block:: ipython3

   import neuronx_distributed as nxd
   nxd.save_checkpoint(
      checkpoint_dir_str="lora_checkpoint", 
      tag="lora", 
      model=model
   )


The output checkpoints of LoRA Adapter will be saved under folder ``lora_checkpoint/lora/``. 

.. note::
   If LoRA configuration file is saved separately, it should be placed as ``lora_adapter/adapter_config.json``.


3. Run the fine-tune script

.. code-block:: ipython3

   ./test_llama_lora_finetune.sh
