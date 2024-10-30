.. _llama2_tp_pp_ptl_tutorial:

Training Llama-2-7B/13B/70B using Tensor Parallelism and Pipeline Parallelism with Neuron PyTorch-Lightning 
=========================================================================================

In this section, we showcase to pretrain a Llama2 7B/13B/70B with Tensor Parallelism and Pipeline Parallel using Neuron PyTorch-Lightning APIs, please refer to :ref:`Llama2 7B Tutorial <llama2_7b_tp_zero1_tutorial>`, :ref:`Llama2 13B/70B Tutorial <llama2_tp_pp_tutorial>`
and :ref:`Neuron PT-Lightning Developer Guide <_ptl_developer_guide>` for more context.


Setting up environment:
^^^^^^^^^^^^^^^^^^^^^^^
                       
For this experiment, we will use AWS ParallelCluster with at least four trn1.32xlarge compute nodes(at least 32 nodes are needed for 13B/70B model size).
`Train your model on ParallelCluster <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/parallelcluster/parallelcluster-training.html>`__
introduces how to setup and use a ParallelCluster.
To setup the packages on the headnode of the ParallelCluster, follow the instructions mentioned here:
:ref:`Install PyTorch Neuron on Trn1 <setup-torch-neuronx>`.

We also need to install the ``neuronx-distributed`` package inside the virtual env using the following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

Let’s download the scripts for pretraining:


1. Creating a directory to hold our experiments

.. code:: ipython3

   mkdir -p ~/examples/llama2_lightning
   cd ~/examples/llama2_lightning

2. Downloading training scripts for our experiments

.. code:: ipython3

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/data_module.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/module_llama.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/run_llama_nxd_ptl.py

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/get_dataset.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lr.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/modeling_llama_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/requirements.txt
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/requirements_ptl.txt
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/training_utils.py

If you want to pre-train Llama 7B, you would need to run the following steps -

.. code:: ipython3

    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/run_llama_7b_tp_ptl.sh
    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/tp_zero1_llama_hf_pretrain/7B_config_llama2/config.json
    chmod +x run_llama_7b_tp_ptl.sh

If you want to pre-train Llama 13B, you would need to run the following steps -

.. code:: ipython3

    mkdir -p ~/examples/llama2_lightning/13B_config
    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/run_llama_13b_tp_pp_ptl.sh
    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/tp_pp_llama_hf_pretrain/13B_config_llama2/config.json -P 13B_config/
    chmod +x run_llama_13b_tp_pp_ptl.sh

If you want to pre-train Llama 70B, you would need to run the following steps -

.. code:: ipython3

    mkdir -p ~/examples/llama2_lightning/70B_config
    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/lightning/run_llama_70b_tp_pp_ptl.sh
    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/main/examples/training/llama/tp_pp_llama_hf_pretrain/70B_config_llama2/config.json -P 70B_config/
    chmod +x run_llama_70b_tp_pp_ptl.sh

3. Installing the additional requirements and giving the right permissions to our shell script

.. code:: ipython3

   python3 -m pip install -r requirements.txt
   python3 -m pip install -r requirements_ptl.txt  # Currently we're supporting Lightning version 2.1.0


Next, we tokenize our dataset. 
``Note``: To tokenize the data, we must request the tokenizer from `HuggingFace` and `Meta` by following 
the instructions at the following link: `HuggingFace Llama 2 7B Model <https://huggingface.co/meta-llama/Llama-2-7b>`__ .
Use of the Llama 2 model is governed by the Meta license. In order to download the model weights and tokenizer, please 
visit the above website and accept their License before requesting access. After access has been granted, 
you may use the download scripts provided by Meta to download the model weights and tokenizer to your cluster.

Once you have downloaded the tokenizer and model weights, you can copy the ``tokenizer.model`` to the ``~/examples/llama2_lightning`` directory.

Next let’s download and pre-process the dataset:

.. code:: ipython3

   cd ~/examples/llama2_lightning
   python3 get_dataset.py --llama-version 2  # currently we only support Llama-2 models

``Note``: In case you see an error of the following form when downloading data: ``huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ubuntu/examples/llama2_lightning'. Use `repo_type` argument if needed.`` 
This could be because of a stale cache. Try deleting the cache using: 

.. code:: ipython3

   sudo rm -rf /home/ubuntu/.cache/


At this point, you are all set to start training.

Training Llama2-7B with Tensor Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By this step, the ParallelCluster is all setup for running experiments. 
Before we run training, we first pre-compile the graphs using the :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`.
Let’s run the command below:

.. code:: ipython3

   sbatch --exclusive \
   --nodes 4 \
   --cpus-per-task 128 \
   --wrap="srun neuron_parallel_compile bash $(pwd)/run_llama_7b_tp_ptl.sh"

This script uses a tensor-parallel size of 8.
This will automatically set the zero-1 sharding degree to 16 (4 * 32 workers / tensor_parallel_size). 

``Note``: You can use any number of nodes in this case, would just need to adjust the number of nodes in the above 
slurm command accordingly. Also, the number of nodes used in parallel_compile command should be same as the actual 
training run. This is because, as the number of nodes change, the data-parallel degree would change too. This would 
result in more workers participating in operations like `gradient all-reduce` which would result in new graphs getting 
created. 

Once the graphs are compiled we can now run training and observe our loss goes down.
To run the training, we just run the above command but without ``neuron_parallel_compile``.

.. code:: ipython3

   sbatch --exclusive \
   --nodes 4 \
   --cpus-per-task 128 \
   --wrap="srun bash $(pwd)/run_llama_7b_tp_ptl.sh"

Training Llama2-13B/70B with Tensor Parallelism and Pipeline Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we use ``Llama70B`` as an example. To run 13B, simply change the script from ``run_llama_70b_tp_pp.sh`` to ``run_llama_13B_tp_pp.sh``
Before we run training, we first pre-compile the graphs using the :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`.
Let’s run the command below:

Pre-compiling

.. code:: ipython3

   sbatch --exclusive \
   --nodes 32 \
   --cpus-per-task 128 \
   --wrap="srun neuron_parallel_compile bash $(pwd)/run_llama_70b_tp_pp_ptl.sh"

This script uses a tensor-parallel size of 8, pipeline-parallel size of 8
To run the training, we just use the above command but without ``neuron_parallel_compile``.

.. code:: ipython3

   sbatch --exclusive \
   --nodes 32 \
   --cpus-per-task 128 \
   --wrap="srun bash $(pwd)/run_llama_70b_tp_pp_ptl.sh"


Checkpointing:
^^^^^^^^^^^^^^

To enable checkpoint saving, add following flags to ``run_llama_7b_tp_ptl.sh``/ ``run_llama_13b_tp_pp.sh`` /  ``run_llama_70B_tp_pp.sh``:
* ``--save_checkpoint`` Add this flag to enable checkpoint saving
* ``--checkpoint_freq`` Number of steps to save a checkpoint
* ``--checkpoint_dir`` Direction to save the checkpoint 
* ``--num_kept_checkpoint`` Number of checkpoints to save, older checkpoint will be deleted manually, set to -1 to keep all saved checkpoints
* ``--save_load_xser`` load with torch xla serialization to reduce time saving, it's recommended to enable xser for significantly faster save/load. Note that if the chekpoint is saved with xser, it can only be loaded with xser, vice versa. 

To enable checkpoint loading, add following flags to ``run_llama_7b_tp_ptl.sh``/ ``run_llama_13b_tp_pp.sh`` /  ``run_llama_70B_tp_pp.sh``:
* ``--resume_ckpt`` 
* ``--load_step`` Step to retrieve checkpoint from
* ``--checkpoint_dir`` Direction to load the checkpoint from
* ``--save_load_xser`` load with torch xla serialization to reduce time saving, it's recommended to enable xser for significantly faster save/load. Note that if the chekpoint is saved with xser, it can only be loaded with xser, vice versa. 
