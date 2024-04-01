.. _llama2_tp_pp_tutorial:

Training Llama-2-13B/70B with Tensor Parallelism and Pipeline Parallelism (``neuronx-distributed`` )
=========================================================================================

In this section, we showcase to pretrain a Llama2 13B and 70B model by using the tensor parallel, pipeline parallel, sequence parallel, activation
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

   mkdir -p ~/examples/tp_pp_llama2_hf_pretrain
   cd ~/examples/tp_pp_llama2_hf_pretrain
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_pp_llama2_hf_pretrain/activation_checkpoint.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_pp_llama2_hf_pretrain/logger.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/lr.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_pp_llama2_hf_pretrain/run_llama_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/training_utils.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/convert_checkpoints.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/get_dataset.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/modeling_llama_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/requirements.txt

If you want to pre-train Llama 70B, you would need to run the following steps -

.. code:: ipython3

    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_pp_llama2_hf_pretrain/run_llama_70b_tp_pp.sh
    chmod +x run_llama_70b_tp_pp.sh
    mkdir 70B_config && cd 70B_config
    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_pp_llama2_hf_pretrain/70B_config/config.json
    cd .. && cp 70B_config/config.json .

If you want to pre-train Llama 13B, you would need to run the following steps -


.. code:: ipython3

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_pp_llama2_hf_pretrain/run_llama_13B_tp_pp.sh
   chmod +x run_llama_13B_tp_pp.sh
   mkdir 13B_config && cd 13B_config
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama2/tp_pp_llama2_hf_pretrain/13B_config/config.json
   cd .. && cp 13B_config/config.json .

The below tutorial uses ``Llama70B`` as an example. To run 13B, simply change the script from ``run_llama_70b_tp_pp.sh`` to ``run_llama_13B_tp_pp.sh``.

First, let's get all the needed dependencies

.. code:: ipython3

    python3 -m pip install -r requirements.txt

To tokenize the data, we must request the tokenizer from hugging face and meta by following the instructions at the following link: `HuggingFace Llama 2 7B Model <https://huggingface.co/meta-llama/Llama-2-7b>`__ . 

Use of the Llama 2 model is governed by the Meta license. In order to download the model weights and tokenizer, please visit the above website and accept their License before requesting access. After access has been granted, you may use the download scripts provided by Meta to download the model weights and tokenizer to your cluster.

Once you have downloaded the tokenizer and model weights, you can copy the ``tokenizer.model`` to the ``~/examples/tp_pp_llama2_hf_pretrain`` directory.

Next let’s download and pre-process the dataset:

.. code:: ipython3

   cd ~/examples/tp_pp_llama2_hf_pretrain
   python3 get_dataset.py

In case you see an error of the following form when downloading data: ``huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ubuntu/examples/tp_pp_llama2_hf_pretrain'. Use `repo_type` argument if needed.`` This could be because of a stale cache. Try deleting the cache using: 

.. code:: ipython3

   sudo rm -rf /home/ubuntu/.cache/

In case you see an error of the following form when downloading data: ```NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.``` Try upgrading pip:

.. code:: ipython3

   pip install -U datasets


At this point, you are all set to start training.


Running training

We first pre-compile the graphs using the ``neuron_parallel_compile``. Let’s run the command below:

.. code:: ipython3

   sbatch --exclusive \
   --nodes 32 \
   --cpus-per-task 128 \
   --wrap="srun neuron_parallel_compile bash $(pwd)/run_llama_70b_tp_pp.sh"

This script uses a tensor-parallel size of 8, pipeline-parallel size of 8
To run the training, we just use the above command but without ``neuron_parallel_compile``.

.. code:: ipython3

   sbatch --exclusive \
   --nodes 32 \
   --cpus-per-task 128 \
   --wrap="srun bash $(pwd)/run_llama_70b_tp_pp.sh"


To achieve better performance, the script applies few techniques:

`Sequence Parallelism and Selective Activation Checkpointing`

As explained in the :ref:`Activation Memory Recomputation Doc <activation_memory_reduction>`, both `Sequence Parallelism` 
and `Selective activation checkpointing` can help with activation memory reduction thereby allowing us to fit bigger 
models with less number of devices. 
Please refer to :ref:`Activation Memory Reduction Developer Guide <activation_memory_reduction_developer_guide>` on how to 
enable sequence parallel and selective activation checkpointing. 


`GQAQKVColumnParallelLinear Layer`:

In LLama 70B GQA module, the K and V attention heads are `8` whereas Q has `64` attentions heads. Since the number of 
attention heads should be divisible by tensor_parallel_degree, we would end up using a tp_degree of 8. Hence to fit 
a 70B model, we would have to use a higher pipeline-parallel degree. Using higher pipeline-parallel degree works well 
when the global batch size is very high, however, as the data-parallel degree increases at higher cluster size, the 
batch size per node decreases. This would result in higher `pipeline bubble <https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/>`__ 
thereby reducing performance. To mitigate this issue, one can use the :ref:`GQAQKVColumnParallelLinear <parameters-11>` layer with the
`kv_size_multiplier` set to 4. This would repeat the KV heads and make them 32. This would allow doing tensor-parallelism 
using tp_degree of 32. This reduces the activation memory per device and thereby eventually allows using a pipeline 
parallel degree of 4. This can be enabled by passing the argument:

.. code:: ipython3

   torchrun $DISTRIBUTED_ARGS run_llama_nxd.py \
   ... \
   --qkv_linear 1 \
   --kv_replicator 4 \
   --tb_dir $tb_dir |& tee $LOG_PATH/log

The above changes are already included in the `run_llama_70b_tp_pp.sh`. For Llama13B model we only do 8-way tensor parallelism so
we do not need this change.



`Save/Load Checkpoint` (refer to :ref:`API GUIDE<api_guide>` for more context about checkpoint APIs):

To enable checkpoint saving, add the following flags to ``run_llama_70b_tp_pp.sh``:

* ``--checkpoint_freq`` Number of steps to save a checkpoint, set to -1 to disable saving checkpoint, should set as -1 when pre-compling graph
* ``--checkpoint_dir`` Direction to save the checkpoint
* ``--num_kept_checkpoint`` Number of checkpoints to save, older checkpoint will be deleted manually, set to -1 to keep all saved checkpoints.
* ``--save_load_xser`` Save with torch xla serialization to reduce time saving, it's recommended to enable xser for significantly faster save/load 
* ``--async_checkpoint_saving`` Whether to use asynchronous checkpoint saving to reduce saving time.

To enable checkpoint loading, add the following flags to ``run_llama_70b_tp_pp.sh``:

* ``--loading_step`` Step to retrieve checkpoint from, set to -1 to disable checkpoint loading. Set to ``latest_if_exists`` to load the latest checkpoint under ``checkpoint_dir``.
* ``--checkpoint_dir`` Direction to load the checkpoint from
* ``--save_load_xser`` load with torch xla serialization to reduce time saving, it's recommended to enable xser for significantly faster save/load. Note that if the chekpoint is saved with xser, it can only be loaded with xser, vice versa. 

Load pretrained model:

We also provide option to load from pretrained HF model. Before loading, convert the full model to sharded model with ``convert_checkpoints.py``:

.. code:: ipython3

   python3 convert_checkpoints.py --tp_size <tp_size> --pp_size <pp_size> --n_layers <number_of_layers>  --input_dir  <path_to_full_model> --output_dir <sharded_model_path> --convert_from_full_model 

And add ``--pretrained_weight_dir <sharded_model_path>`` flag to ``run_llama_70b_tp_pp.sh``


Convert sharded model to full model with ``convert_checkpoints.py``:

.. code:: ipython3

   python3 convert_checkpoints.py --tp_size <tp_size> --pp_size <pp_size> --n_layers <number_of_layers>  --input_dir  <sharded_model_dir> --output_dir <full_model_dir> --convert_to_full_model --kv_size_multiplier <kv_size_multiplier> --config config.json
