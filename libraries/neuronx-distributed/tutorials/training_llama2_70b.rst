.. _llama2_70b_tp_pp_tutorial:

Training Llama-2-70B with Tensor Parallelism and Pipeline Parallelism (``neuronx-distributed`` )
=========================================================================================

In this section, we showcase to pretrain a Llama2 70B model by using the tensor parallel, pipeline parallel, sequence parallel, activation
checkpoint as well as constant mask optimization in the ``neuronx-distributed`` package.

Setting up environment:
                       
For this experiment, we will use a ParallelCluster with at least 32 trn1-32xl compute nodes.
`Train your model on ParallelCluster <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/parallelcluster/parallelcluster-training.html>`__
introduces how to setup and use a ParallelCluster.

We also need to install the ``neuronx-distributed`` package using the following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

Let’s download the scripts for pretraining:

.. code:: ipython3

   mkdir -p ~/examples/tp_pp_llama2_70b_hf_pretrain
   cd ~/examples/tp_pp_llama2_70b_hf_pretrain
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/tp_pp_llama2_70b_hf_pretrain/activation_checkpoint.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/tp_pp_llama2_70b_hf_pretrain/config.json
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/tp_pp_llama2_70b_hf_pretrain/lr.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/tp_pp_llama2_70b_hf_pretrain/run_llama_70b_tp_pp.sh
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/tp_pp_llama2_70b_hf_pretrain/run_llama_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/tp_pp_llama2_70b_hf_pretrain/training_utils.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/adamw_fp32_optim_params.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/convert_checkpoints.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/get_dataset.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/modeling_llama2_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/llama2/requirements.txt
   python3 -m pip install -r requirements.txt
   chmod +x run_llama_70b_tp_pp.sh

To tokenize the data, we must request the tokenizer from hugging face and meta by following the instructions at the following link: `HuggingFace Llama 2 7B Model <https://huggingface.co/meta-llama/Llama-2-7b>`__ . 

Use of the Llama 2 model is governed by the Meta license. In order to download the model weights and tokenizer, please visit the above website and accept their License before requesting access. After access has been granted, you may use the download scripts provided by Meta to download the model weights and tokenizer to your cluster.

Once you have downloaded the tokenizer and model weights, you can copy the ``tokenizer.model`` to the ``~/examples/tp_pp_llama2_70b_hf_pretrain`` directory.

Next let’s download and pre-process the dataset:

.. code:: ipython3

   cd ~/examples/tp_pp_llama2_70b_hf_pretrain
   python3 get_dataset.py

In case you see an error of the following form when downloading data: ``huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ubuntu/examples/tp_pp_llama2_70b_hf_pretrain'. Use `repo_type` argument if needed.`` This could be because of a stale cache. Try deleting the cache using: 

.. code:: ipython3

   sudo rm -rf /home/ubuntu/.cache/


At this point, you are all set to start training.


Running training

We first pre-compile the graphs using the ``neuron_parallel_compile``. Let’s run the command below:

.. code:: ipython3

   sbatch --exclusive \
   --nodes 32 \
   --wrap="srun neuron_parallel_compile bash $(pwd)/run_llama_70b_tp_pp.sh"

This script uses a tensor-parallel size of 8, pipeline-parallel size of 8
To run the training, we just use the above command but without ``neuron_parallel_compile``.

.. code:: ipython3

   sbatch --exclusive \
   --nodes 32 \
   --wrap="srun bash $(pwd)/run_llama_70b_tp_pp.sh"


Sequence Parallel

Please refer to :ref:`GPT-NeoX 6.9B tutorial<gpt_neox_tp_zero1_tutorial>` on how to enable sequence parallel.

On top of it, we further coalesced parallel matrix multiply to improve throughput:

* We coalesced ``query``, ``key`` and ``value`` into one matrix multiply
* We coalesced ``gate_proj`` and ``up_proj`` into one matrix multiply

Please check ``modeling_llama2_nxd.py`` and ``tp_dp_gpt_neox_20b_hf_pretrain.py`` for details.


Gradient checkpointing:
Check :ref:`PP Developer Guide<pp_developer_guide>` for more detail


Save/Load Checkpoint (refer to :ref:`API GUIDE<api_guide>` for more context about checkpoint APIs):

To enable checkpoint saving, add the following flags to ``run_llama_70b_tp_pp.sh``:

* ``--checkpoint_freq`` Number of steps to save a checkpoint, set to -1 to disable saving checkpoint, should set as -1 when pre-compling graph
* ``--checkpoint_dir`` Direction to save the checkpoint 
* ``--num_kept_checkpoint`` Number of checkpoints to save, older checkpoint will be deleted manually, set to -1 to keep all saved checkpoints.
* ``--save_load_xser`` Save with torch xla serialization to reduce time saving, it's recommended to enable xser for significantly faster save/load 

To enable checkpoint loading, add the following flags to ``run_llama_70b_tp_pp.sh``:

* ``--loading_step`` Step to retrieve checkpoint from, set to -1 to disable checkpoint loading
* ``--checkpoint_dir`` Direction to load the checkpoint from
* ``--save_load_xser`` load with torch xla serialization to reduce time saving, it's recommended to enable xser for significantly faster save/load. Note that if the chekpoint is saved with xser, it can only be loaded with xser, vice versa. 


Load pretrained model:

We also provide option to load from pretrained HF model. Before loading, convert the full model to sharded model with ``convert_checkpoints.py``:

.. code:: ipython3

   python3 convert_checkpoints.py --tp_size <tp_size> --pp_size <pp_size> --n_layers <number_of_layers>  --input_dir  <path_to_full_model> --output_dir <sharded_model_path> --convert_from_full_model 

And add ``--pretrained_weight_dir <sharded_model_path>`` flag to ``run_llama_70b_tp_pp.sh``


Convert sharded model to full model with ``convert_checkpoints.py``:

.. code:: ipython3

   python3 convert_checkpoints.py --tp_size <tp_size> --pp_size <pp_size> --n_layers <number_of_layers>  --input_dir  <sharded_model_dir> --output_dir <full_model_dir> --convert_to_full_model 