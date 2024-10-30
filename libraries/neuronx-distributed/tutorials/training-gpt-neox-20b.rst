.. _gpt_neox_20b_tp_zero1_tutorial:

Training GPT-NeoX 20B with Tensor Parallelism and ZeRO-1 Optimizer 
=========================================================================================

In this section, we showcase to pretrain a GPT-NeoX 20B model by using the sequence parallel optimization
of tensor parallelism in the ``neuronx-distributed`` package. Please refer to the `Neuron Samples repository <https://github.com/aws-neuron/aws-neuron-samples/tree/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain>`__ to view the files in this tutorial.

This GPT-NeoX 20B tutorial differs from the :ref:`GPT-NeoX 6.9B tutorial<gpt_neox_tp_zero1_tutorial>` in the following ways:

* sequence parallel optimization has been applied
* parallel cross entropy has been applied
* the model size has been increased from 6.9B to 20B
* the TP degree has been increased from 8 to 32

Setting up environment is same as the :ref:`GPT-NeoX 6.9B tutorial<gpt_neox_tp_zero1_tutorial>`.

**Let’s download the scripts for pretraining:**

.. code:: ipython3

   mkdir -p ~/examples/tp_dp_gpt_neox_hf_pretrain
   cd ~/examples/tp_dp_gpt_neox_hf_pretrain
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain.sh
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/modeling_gpt_neox_nxd.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/utils.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/tp_dp_gpt_neox_hf_pretrain/common/adamw_fp32_optim_params.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/tp_dp_gpt_neox_hf_pretrain/common/get_dataset.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/tp_dp_gpt_neox_hf_pretrain/common/requirements.txt
   python3 -m pip install -r requirements.txt

Next let’s download and pre-process the dataset:

.. code:: ipython3

   cd ~/examples/tp_dp_gpt_neox_hf_pretrain
   python3 get_dataset.py

At this point, you are all set to start training.

**Running training**

We first pre-compile the graphs using the ``neuron_parallel_compile``.
Suppose the cluster queue name is ``compute1-dy-training-0`` and we are using node 1-4,
let’s run the command below:

.. code:: ipython3

   sbatch --exclusive \
   --nodelist=compute1-dy-training-0-[1-4] \
   --cpus-per-task 128 \
   --wrap="srun neuron_parallel_compile bash $(pwd)/tp_dp_gpt_neox_20b_hf_pretrain.sh"

This script uses a tensor-parallel size of 32.
This will automatically set the zero-1 sharding degree to 4 (4 * 32 workers / tensor_parallel_size).
Once the graphs are compiled we can now run training and observe our loss goes down.
To run the training, we just the above command but without ``neuron_parallel_compile``.

.. code:: ipython3

   sbatch --exclusive \
   --nodelist=compute1-dy-training-0-[1-4] \
   --cpus-per-task 128 \
   --wrap="srun bash $(pwd)/tp_dp_gpt_neox_20b_hf_pretrain.sh"


**Sequence Parallel**

We made the following model level modifications to enable sequence parallel:

* turn on ``sequence_parallel_enabled`` of ``ColumnParallelLinear`` and ``RowParallelLinear``
  in ``GPTNeoXAttention`` and ``GPTNeoXMLP``;
* replace torch ``LayerNorm`` in ``GPTNeoXLayer`` and ``GPTNeoXModel`` with neuronx-distributed  ``LayerNorm``
  with ``sequence_parallel_enabled``
  turned on;
* dimension transposition of intermediate states in the forward function of ``GPTNeoXAttention``.
* dimension transposition and collective communication of intermediate states in the forward function of ``GPTNeoXModel``.

In the training training script level, we enable:

* all-reduce sequence parallel gradients at the gradient accumulation boundary.

Please check `modeling_gpt_neox_nxd.py <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/modeling_gpt_neox_nxd.py>`__ and `tp_dp_gpt_neox_20b_hf_pretrain.py <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain.py>`__ for details.


**Parallel Cross Entropy**

To enable parallel cross entropy, we made the following model level modeifincations:

* replace the ``CrossEntropyLoss`` with neuronx-distributed ``parallel_cross_entropy`` in the forward
  function of ``GPTNeoXForCausalLM``.
* use ``ColumnParallelLinear`` for the ``embed_out`` layer in ``GPTNeoXForCausalLM``.

Please check ``modeling_gpt_neox_nxd.py`` for details.
