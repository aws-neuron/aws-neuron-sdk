.. _tp_training_tutorial:

Training with Tensor Parallelism 
===========================================================

Keeping the above changes made in :ref:`Developer guide <tp_developer_guide>`, let’s now run an end-to-end training
with tensor-parallelism. This section is adopted from `BERT pretraining
tutorial <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/bert.html#hf-bert-pretraining-tutorial>`__
which used data-parallel training to scale the throughput. In this
section we modify that tutorial to showcase the use of
tensor-parallelism which should enable us to scale the size of the
model.

Setting up environment:
                       
For this experiment, we will use a trn1-32xl machine with the storage
set to 512GB at least.
Follow the instructions mentioned here: 
:ref:`Install PyTorch Neuron on Trn1 <setup-torch-neuronx>`. 
It is recommended to work out of python virtual env so as to avoid package installation issues.

We also have to install the ``neuronx-distributed`` package using the
following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

Make sure the transformers version is set to ``4.26.0`` (Note: If you have transformers-neuronx in your environment, you need to uninstall it to avoid a conflict with the transformers version.)

Let’s download the scripts and datasets for pretraining.

.. code:: ipython3

   mkdir -p ~/examples/tp_dp_bert_hf_pretrain
   cd ~/examples/tp_dp_bert_hf_pretrain
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/tp_dp_bert_hf_pretrain/tp_dp_bert_large_hf_pretrain_hdf5.py
   wget https://raw.githubusercontent.com/aws-neuron/neuronx-distributed/master/examples/training/tp_dp_bert_hf_pretrain/requirements.txt
   python3 -m pip install -r requirements.txt

Next let’s download the tokenizer and the sharded datasets:

.. code:: ipython3

   mkdir -p ~/examples_datasets/
   pushd ~/examples_datasets/
   aws s3 cp s3://neuron-s3/training_datasets/bert_pretrain_wikicorpus_tokenized_hdf5/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar .  --no-sign-request
   tar -xf bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar
   rm bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar
   popd

At this point, you are all set to start training

Running training
                

We first pre-compile the graphs using the ``neuron_parallel_compile``.
This process is similar to one discussed in the `BERT pretraining
tutorial <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/bert.html#hf-bert-pretraining-tutorial>`__
. Let’s run the command below:

.. code:: ipython3

   cd ~/examples/tp_dp_bert_hf_pretrain
   export XLA_DOWNCAST_BF16=1
   neuron_parallel_compile torchrun --nproc_per_node=32 \
   tp_dp_bert_large_hf_pretrain_hdf5.py \
   --tensor_parallel_size 8 \
   --steps_this_run 10 \
   --batch_size 64 \
   --grad_accum_usteps 64 |& tee compile_log.txt

This script uses a tensor-parallel size of 8. This will automatically
set the data-parallel degree to 4 (32 workers / tensor_parallel_size).
Once the graphs are compiled we can now run training and observe our
loss go down. To run the training, we just the above command but without
``neuron_parallel_compile``.

.. code:: ipython3

   XLA_DOWNCAST_BF16=1 torchrun --nproc_per_node=32 \
   tp_dp_bert_large_hf_pretrain_hdf5.py \
   --tensor_parallel_size 8 \
   --steps_this_run 10 \
   --batch_size 64 \
   --grad_accum_usteps 64 |& tee training_log.txt

You would notice that the throughput is lower when you run the
``dp_bert_large_hf_pretrain_hdf5.py``. This is expected as the number of
data-parallel workers have gone down (from 32 to 4). However, if you
open ``neuron-top`` in another terminal, you should see the memory
utilization per core for this script is lower than the
``dp_bert_large_hf_pretrain_hdf5.py``. Since the memory requirement has
gone down, you can scale the size of model either by increasing the
number of layers/attention heads/hidden sizes.

The loss curve should match to the loss curve we would get from the
data_parallel counterpart.

Known Issues:
~~~~~~~~~~~~~

1. Currently the checkpoints dumped during training are sharded and
   users would have to write a script to combine the checkpoints
   themselves. This should be fixed in the future release
