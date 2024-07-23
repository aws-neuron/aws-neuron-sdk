.. _llama2_tp_pp_tutorial:
.. _llama3_tp_pp_tutorial:

Training Llama-3.1-70B, Llama-3-70B or Llama-2-13B/70B with Tensor Parallelism and Pipeline Parallelism (``neuronx-distributed`` )
================================================================================================================

In this section, we showcase to pretrain Llama 3.1, Llama3 70B and Llama2 13B/70B model by using the tensor parallel, pipeline parallel, sequence parallel, activation
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

   mkdir -p ~/examples/tp_pp_llama_hf_pretrain
   cd ~/examples/tp_pp_llama_hf_pretrain
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/activation_checkpoint.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/logger.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/lr.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/run_llama_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/training_utils.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/convert_checkpoints.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/get_dataset.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/modeling_llama_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/requirements.txt



If you want to pre-train Llama3.1 70B, you would need to run the following steps -

.. code:: ipython3

    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/run_llama3_70B_tp_pp.sh
    chmod +x run_llama3_70B_tp_pp.sh
    mkdir 70B_config_llama3 && cd 70B_config_llama3
    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/70B_config_llama3.1/config.json
    cd ..

If you want to pre-train Llama3 70B, you would need to run the following steps -

.. code:: ipython3

    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/run_llama3_70B_tp_pp.sh
    chmod +x run_llama3_70B_tp_pp.sh
    mkdir 70B_config_llama3 && cd 70B_config_llama3
    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/70B_config_llama3/config.json
    cd ..


For llama2 13B, you would need to run the following steps -

.. code:: ipython3

    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/run_llama2_13B_tp_pp.sh
    chmod +x run_llama2_13B_tp_pp.sh
    mkdir 13B_config_llama2 && cd 13B_config_llama2
    wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/13B_config_llama2/config.json
    cd .. 


If you want to pre-train Llama2 70B, you would need to run the following steps -


.. code:: ipython3

   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/run_llama2_70B_tp_pp.sh
   chmod +x run_llama2_70B_tp_pp.sh
   mkdir 70B_config_llama2 && cd 70B_config_llama2
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/llama/tp_pp_llama_hf_pretrain/70B_config_llama2/config.json
   cd ..



The below tutorial uses ``Llama3.1 70B`` as an example. To run Llama2 70B or 13B, simply change the script from ``run_llama3_70B_tp_pp.sh`` to ``run_llama2_70B_tp_pp.sh`` or ``run_llama2_13B_tp_pp.sh``.

First, let's get all the needed dependencies

.. code:: ipython3

    python3 -m pip install -r requirements.txt
    

To tokenize the data, we must request the tokenizer from hugging face and meta by following the instructions at the following link: `HuggingFace Llama 3 8B Model <https://huggingface.co/meta-llama/Meta-Llama-3-8B>`__ . 

Use of the Llama models is governed by the Meta license. In order to download the model weights and tokenizer, please visit the above website and accept their License before requesting access. After access has been granted, you may use the following python3 script along with your own hugging face token to download and save required tokenizer.

Run the following from ``~/examples/tp_pp_llama_hf_pretrain`` directory:

.. code:: ipython3

   from transformers import AutoTokenizer

   tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', token='your_own_hugging_face_token')  
   # For llama2 uncomment line below
   # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token='your_own_hugging_face_token') 

   tokenizer.save_pretrained(".")

For Llama3.1/Llama3, make sure your ``~/examples/tp_pp_llama2_hf_pretrain`` directory has the following files:

.. code:: ipython3

   './tokenizer_config.json', './special_tokens_map.json', './tokenizer.json'


For Llama2, you can just copy the ``tokenizer.model`` to the ``~/examples/tp_pp_llama2_hf_pretrain`` directory.


Next let’s download and pre-process the dataset:

.. code:: ipython3

   cd ~/examples/tp_pp_llama_hf_pretrain
   python3 get_dataset.py --llama-version 3  # change the version number to 2 for Llama-2 models

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
   --wrap="srun neuron_parallel_compile bash $(pwd)/run_llama3_70B_tp_pp.sh"

This script uses a tensor-parallel size of 8, pipeline-parallel size of 8
To run the training, we just use the above command but without ``neuron_parallel_compile``.

.. code:: ipython3

   sbatch --exclusive \
   --nodes 32 \
   --cpus-per-task 128 \
   --wrap="srun bash $(pwd)/run_llama3_70B_tp_pp.sh"


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


`Flash Attention:`

We're introducing flash attention function for better performance/memory efficiency. Currently it's enabled by default, to disable it
set ``--use_flash_attention 0`


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

   python3 convert_checkpoints.py --tp_size <tp_size> --pp_size <pp_size> --n_layers <number_of_layers>  --input_dir  <sharded_model_dir> --output_dir <full_model_dir> --convert_to_full_model --kv_size_multiplier <kv_size_multiplier> --config config.json --qkv_linear True --load_xser True
