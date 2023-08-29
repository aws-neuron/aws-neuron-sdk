.. _gpt_neox_tp_zero1_tutorial:

Training GPT-NeoX 6.9B with Tensor Parallelism and ZeRO-1 Optimizer (``neuronx-distributed`` )
=========================================================================================

In this section, we showcase to pretrain a GPT-NeoX 6.9B model by using tenser parallelism
and zero-1 optimzer in the ``neuronx-distributed`` package.

Setting up environment:
                       

For this experiment, we will use a ParallelCluster with at least four trn1-32xl compute nodes.
`Train your model on ParallelCluster <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/parallelcluster/parallelcluster-training.html>`__
introduces how to setup and use a ParallelCluster.
We need first to create and activiate a python virtual env on the head node of the ParallelCluster.
Next follow the instructions mentioned here:
`Install PyTorch Neuron on
Trn1 <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/setup/pytorch-install.html#pytorch-neuronx-install>`__
to install neuron python packages.

We also need to install the ``neuronx-distributed`` package using the following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

Let’s download the scripts for pretraining.

.. code:: ipython3

   mkdir -p ~/examples/tp_dp_gpt_neox_hf_pretrain
   cd ~/examples/tp_dp_gpt_neox_hf_pretrain
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_6.9b_hf_pretrain/tp_dp_gpt_neox_6.9b_hf_pretrain.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_6.9b_hf_pretrain/tp_dp_gpt_neox_6.9b_hf_pretrain.sh
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/common/adamw_fp32_optim_params.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/common/get_dataset.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/common/requirements.txt
   python3 -m pip install -r requirements.txt

Next let’s download and pre-process the dataset:

.. code:: ipython3

   cd ~/examples/tp_dp_gpt_neox_hf_pretrain
   python3 get_dataset.py

At this point, you are all set to start training

Running training
                

We first pre-compile the graphs using the ``neuron_parallel_compile``.
Suppose the cluster quene name is ``compute1-dy-training-0`` and we are using node 1-4,
let’s run the command below:

.. code:: ipython3

   sbatch --exclusive \
   --nodelist=compute1-dy-training-0-[1-4] \
   --wrap="srun neuron_parallel_compile bash $(pwd)/tp_dp_gpt_neox_6.9b_hf_pretrain.sh"

This script uses a tensor-parallel size of 8.
This will automatically set the zero-1 sharding degree to 16 (4 * 32 workers / tensor_parallel_size).
Once the graphs are compiled we can now run training and observe our loss goes down.
To run the training, we just the above command but without ``neuron_parallel_compile``.

.. code:: ipython3

   sbatch --exclusive \
   --nodelist=compute1-dy-training-0-[1-4] \
   --wrap="srun bash $(pwd)/tp_dp_gpt_neox_6.9b_hf_pretrain.sh"

ZeRO-1 Optimizer
                

The training script uses ZeRO-1 optimizer, where the optimizer states are partitioned across
the ranks so that each rank updates only its partition.
Below shows the code snippet of using ZeRO-1 optimizer in training script:

.. code:: ipython3

   from neuronx_distributed.optimizer import NeuronZero1Optimizer

   optimizer = NeuronZero1Optimizer(
        optimizer_grouped_parameters,
        AdamW_FP32OptimParams,
        lr=flags.lr,
        pin_layout=False,
        sharding_groups=parallel_state.get_data_parallel_group(as_list=True),
        grad_norm_groups=parallel_state.get_tensor_model_parallel_group(as_list=True),
    )
