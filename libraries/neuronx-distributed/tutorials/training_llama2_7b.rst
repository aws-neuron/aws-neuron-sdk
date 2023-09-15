.. _llama2_7b_tp_zero1_tutorial:

Training Llama2 7B with Tensor Parallelism and ZeRO-1 Optimizer (``neuronx-distributed`` )
=========================================================================================

In this section, we showcase to pretrain a Llama2 7B model by using the sequence parallel, selective
checkpoint as well as constant mask optimization in the ``neuronx-distributed`` package.

Setting up environment:
                       
For this experiment, we will use a ParallelCluster with at least four trn1-32xl compute nodes.
`Train your model on ParallelCluster <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/parallelcluster/parallelcluster-training.html>`__
introduces how to setup and use a ParallelCluster.
We need first to create and activate a python virtual env on the head node of the ParallelCluster.
Next follow the instructions mentioned here:
:ref:`Install PyTorch Neuron on Trn1 <setup-torch-neuronx>` to install neuron python packages.

We also need to install the ``neuronx-distributed`` package using the following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

Let’s download the scripts for pretraining:

.. code:: ipython3

   mkdir -p ~/examples/tp_zero1_llama2_7b_hf_pretrain
   cd ~/examples/tp_zero1_llama2_7b_hf_pretrain
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_zero1_llama2_7b_hf_pretrain/tp_zero1_llama2_7b_hf_pretrain.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_zero1_llama2_7b_hf_pretrain/tp_zero1_llama2_7b_hf_pretrain.sh
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_zero1_llama2_7b_hf_pretrain/modeling_llama2_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_zero1_llama2_7b_hf_pretrain/adamw_fp32_optim_params.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_zero1_llama2_7b_hf_pretrain/get_dataset.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_zero1_llama2_7b_hf_pretrain/requirements.txt
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_zero1_llama2_7b_hf_pretrain/config.json
   python3 -m pip install -r requirements.txt
   chmod +x tp_zero1_llama2_7b_hf_pretrain.sh

To tokenize the data, we must request the tokenizer from hugging face and meta by following the instructions at the following link: `HuggingFace Llama 2 7B Model <https://huggingface.co/meta-llama/Llama-2-7b>`__ .

Use of the Llama 2 model is governed by the Meta license. In order to download the model weights and tokenizer, please visit the above website and accept their License before requesting access. After access has been granted, you may use the download scripts provided by Meta to download the model weights and tokenizer to your cluster.

Once you have downloaded the tokenizer and model weights, you can copy the ``tokenizer.model`` to the ``~/examples/tp_zero1_llama2_7b_hf_pretrain`` directory.

Next let’s download and pre-process the dataset:

.. code:: ipython3

   cd ~/examples/tp_zero1_llama2_7b_hf_pretrain
   python3 get_dataset.py

In case you see an error of the following form when downloading data: ``huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ubuntu/examples/tp_zero1_llama2_7b_hf_pretrain'. Use `repo_type` argument if needed.`` This could be because of a stale cache. Try deleting the cache using: 

.. code:: ipython3

   sudo rm -rf /home/ubuntu/.cache/


At this point, you are all set to start training.

Running training

We first pre-compile the graphs using the ``neuron_parallel_compile``. Let’s run the command below:

.. code:: ipython3

   sbatch --exclusive \
   --nodes 4 \
   --wrap="srun neuron_parallel_compile bash $(pwd)/tp_zero1_llama2_7b_hf_pretrain.sh"

This script uses a tensor-parallel size of 8.
This will automatically set the zero-1 sharding degree to 16 (4 * 32 workers / tensor_parallel_size).
Once the graphs are compiled we can now run training and observe our loss goes down.
To run the training, we just the above command but without ``neuron_parallel_compile``.

.. code:: ipython3

   sbatch --exclusive \
   --nodes 4 \
   --wrap="srun bash $(pwd)/tp_zero1_llama2_7b_hf_pretrain.sh"


Sequence Parallel

Please refer to :ref:`GPT-NeoX 6.9B tutorial<gpt_neox_tp_zero1_tutorial>` on how to enable sequence parallel.

On top of it, we further coalesced parallel matrix multiply to improve throughput:

* We coalesced ``query``, ``key`` and ``value`` into one matrix multiply
* We coalesced ``gate_proj`` and ``up_proj`` into one matrix multiply

Please check ``modeling_llama2_nxd.py`` and ``tp_dp_gpt_neox_20b_hf_pretrain.py`` for details.


Selective Activation Checkpoint

Instead of checkpointing and recomputing full transformer layers, we checkpoint and recompute only parts of each transformer
layer that take up a considerable amount of memory but are not computationally expensive to recompute, or selective activation
recomputation:

* Rewrite the attention layer into ``core_attn`` function: it takes ``query``, ``key`` and ``value`` as inputs and performs attention.
* We checkpoint ``core_attn`` with ``torch.utils.checkpoint.checkpoint``.


Constant Attention Mask

In decoder transformer, we use casual attention masks to predict next token based on previous tokens. To enable it:

* We use a constant triangular matrix as the casual masks
* We detect constants in compiler with constant folding and save computation.
