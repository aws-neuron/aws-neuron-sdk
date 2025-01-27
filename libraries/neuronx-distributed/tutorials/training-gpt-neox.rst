.. _gpt_neox_tp_zero1_tutorial:

Training GPT-NeoX 6.9B with Tensor Parallelism and ZeRO-1 Optimizer
=========================================================================================

In this section, we showcase to pretrain a GPT-NeoX 6.9B model by using tensor parallelism
and zero-1 optimizer in the ``neuronx-distributed`` package. Please refer to the `Neuron Samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_6.9b_hf_pretrain>`__ to view the files in this tutorial.

**Setting up environment:**
                       

For this experiment, we will use a ParallelCluster with at least four trn1-32xl compute nodes.
`Train your model on ParallelCluster <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/parallelcluster/parallelcluster-training.html>`__
introduces how to setup and use a ParallelCluster.
We need first to create and activate a python virtual env on the head node of the ParallelCluster.
Next follow the instructions mentioned here:
:ref:`Install PyTorch Neuron on Trn1 <setup-torch-neuronx>` to install neuron python packages.

We also need to install and clone the ``neuronx-distributed`` package using the following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --index-url https://pip.repos.neuron.amazonaws.com
   git clone git@github.com:aws-neuron/neuronx-distributed.git

Let’s download the scripts for pretraining.

.. literalinclude:: nxd-source-code/gpt_neox_tp_zero1/gpt_neox_6_9b.sh
   :language: shell
   :lines: 4-10

Next let’s download and pre-process the dataset:

.. literalinclude:: nxd-source-code/gpt_neox_tp_zero1/gpt_neox_6_9b.sh
   :language: shell
   :lines: 12

At this point, you are all set to start training.

**Running training**
                

We first pre-compile the graphs using the ``neuron_parallel_compile``.
Let’s run the command below:

.. literalinclude:: nxd-source-code/gpt_neox_tp_zero1/gpt_neox_6_9b.sh
   :language: shell
   :lines: 16-18

This script uses a tensor-parallel size of 8.
This will automatically set the zero-1 sharding degree to 16 (4 * 32 workers / tensor_parallel_size).
Once the graphs are compiled we can now run training and observe our loss goes down.
To run the training, we just the above command but without ``neuron_parallel_compile``.

.. literalinclude:: nxd-source-code/gpt_neox_tp_zero1/gpt_neox_6_9b.sh
   :language: shell
   :lines: 20-22

**ZeRO-1 Optimizer**
                

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
