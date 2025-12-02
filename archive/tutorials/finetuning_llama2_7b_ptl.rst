.. _llama2_7b_tp_zero1_ptl_finetune_tutorial:

.. meta::
   :noindex:
   :nofollow:
   :description: This tutorial for the AWS Neuron SDK is currently archived and not maintained. It is provided for reference only.

Fine-tuning Llama2 7B with tensor parallelism and ZeRO-1 optimizer using Neuron PyTorch-Lightning 
=========================================================================================

This tutorial shows how to fine-tune Llama2 7B with tensor parallelism and ZeRO-1 using Neuron PyTorch-Lightning APIs. For pre-training information and additional context, see the :ref:`Llama2 7B Tutorial <llama2_7b_tp_zero1_ptl_tutorial>`
and :ref:`Neuron PT-Lightning Developer Guide <ptl_developer_guide>`. 


Setting up the environment
^^^^^^^^^^^^^^^^^^^^^^^

For this experiment, we will use AWS ParallelCluster with at least four trn1.32xlarge compute nodes.
To set up a cluster and prepare it for use, see `Train your model on ParallelCluster <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/devflows/training/parallelcluster/parallelcluster-training.html>`__.
To set up the packages on the head node of the cluster, see
:ref:`Install PyTorch Neuron on Trn1 <setup-torch-neuronx>`.

Install the ``neuronx-distributed`` package inside the virtual environment using the following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

Next, download the scripts for fine-tuning.


1. Create a directory to hold the experiments.

.. code:: ipython3

   mkdir -p ~/examples/tp_zero1_llama2_7b_hf_finetune_ptl
   cd ~/examples/tp_zero1_llama2_7b_hf_finetune_ptl

2. Download training scripts for the experiments.

.. code:: ipython3

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/data_module.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/module_llama.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/tp_zero1_llama2_7b_hf_finetune_ptl.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/tp_zero1_llama2_7b_hf_finetune_ptl.sh
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/finetune_config/config.json
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lr.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/modeling_llama_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/requirements.txt
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/requirements_ptl.txt
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/training_utils.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/convert_checkpoints.py

3. Install the additional requirements and give the right permissions to the shell script.

.. code:: ipython3

   python3 -m pip install -r requirements.txt
   python3 -m pip install -r requirements_ptl.txt  # Currently we're supporting Lightning version 2.4.0
   python3 -m pip install optimum-neuron==0.0.18 nltk  # Additional dependencies for evaluation
   python3 -m pip install --no-warn-conflicts transformers==4.32.1   # Ping transformers version 4.32.1
   chmod +x tp_zero1_llama2_7b_hf_finetune_ptl.sh

Download the Llama2-7B pre-trained checkpoint from HuggingFace.


1. Create a Python script ``get_model.py`` with the following lines: 

.. code:: ipython3

   import torch
   from transformers.models.llama.modeling_llama import LlamaForCausalLM
   model = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf")
   torch.save(model.state_dict(), "llama-7b-hf-pretrained.pt")

2. Run the download script and conversion script to pull and convert the checkpoint, note that conversion scripts requires high memory so need to login to a compute node to do so:

.. code:: ipython3

   ssh compute1-dy-training-0-1
   source ~/aws_neuron_venv_pytorch/bin/activate
   cd ~/examples/tp_zero1_llama2_7b_hf_finetune_ptl
   python3 get_model.py
   python3 convert_checkpoints.py --tp_size 8 --convert_from_full_model --config config.json --input_dir llama-7b-hf-pretrained.pt --output_dir llama7B-pretrained/pretrained_weight

3. (Optional) If you are loading checkpoint from different directory, set the checkpoint path by adding the following flag to ``tp_zero1_llama2_7b_hf_finetune_ptl.sh``:

   * ``--pretrained_ckpt``.

   This provides direction to the pre-trained checkpoint to be loaded.

Then, set the dataset for the fine-tuning job. In this example, we will use Dolly, which is an open source dataset
of instruction-following records on categories outlined in the InstructGPT paper, including brainstorming, classification,
closed QA, generation, information extraction, open QA, and summarization.

{
  "instruction": "Alice's parents have three daughters: Amy, Jessy, and what's the name of the third daughter?",
  
  "context": "",
  
  "response": "The name of the third daughter is Alice"
}

Configure the following flags in ``tp_zero1_llama2_7b_hf_finetune_ptl.sh``:

.. code:: ipython3

   --data_dir "databricks/databricks-dolly-15k" \
   --task "open_qa"

At this point, you are all set to start fine-tuning.

Running fine-tuning
^^^^^^^^^^^^^^^^

By this step, the cluster is all set up for running experiments. 
Before running training, first pre-compile the graphs using the :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`.
Run the command below:

.. code:: ipython3

   sbatch --exclusive \
   --nodes 1 \
   --wrap="srun neuron_parallel_compile bash $(pwd)/tp_zero1_llama2_7b_hf_finetune_ptl.sh"

This script uses a tensor-parallel size of 8.
This automatically sets the zero-1 sharding degree to 4 (32 workers / tensor_parallel_size). 

`Note`: You can use any number of nodes in this case by adjusting the number of nodes in the above 
Slurm command accordingly. Also, the number of nodes used in the parallel_compile command should be same as the number used in the actual 
training run. This is because, as the number of nodes change, the data-parallel degree changes too. This  
results in more workers participating in operations like `gradient all-reduce`, which results in new graphs getting 
created. 

After the graphs are compiled, you can run training and observe how the loss goes down.
Before the actual fine-tune started, we need  to prepare the dataset

.. code:: ipython3

   python3 -c "import nltk; nltk.download('punkt')" 

To run the training, run the above command without ``neuron_parallel_compile``:

.. code:: ipython3

   sbatch --exclusive \
   --nodes 1 \
   --wrap="srun bash $(pwd)/tp_zero1_llama2_7b_hf_finetune_ptl.sh"

At the end of fine-tuning, run evaluation once with a test data split by generating sentences and calculating ROUGE scores.
The final evaluation results and ROUGE score are then printed in your terminal.


Checkpointing
^^^^^^^^^^^^^^

To enable checkpoint saving, add the following flags to ``tp_zero1_llama2_7b_hf_finetune_ptl.sh``:

* ``--save_checkpoint`` Enables checkpoint saving.
* ``--checkpoint_freq`` Number of steps to save a checkpoint.
* ``--checkpoint_dir`` Direction to save the checkpoint.
* ``--num_kept_checkpoint`` Number of checkpoints to save. Older checkpoint are deleted manually. Set to -1 to keep all saved checkpoints.
* ``--save_load_xser`` Loads with torch_xla serialization to reduce time saving. We recommend enabling xser for significantly faster save and load times. Note that if the checkpoint is saved with xser, it can only be loaded with xser, and vice versa. 

To enable checkpoint loading, add the following flags to ``tp_zero1_llama2_7b_hf_finetune_ptl.sh``:

* ``--resume_ckpt`` Resumes the checkpoint process.
* ``--load_step`` The step to retrieve the checkpoint from.
* ``--checkpoint_dir`` Direction to load the checkpoint from.
* ``--save_load_xser`` Loads with torch_xla serialization to reduce time saving. We recommend enabling xser for significantly faster save and load times. Note that if the checkpoint is saved with xser, it can only be loaded with xser, and vice versa. 
