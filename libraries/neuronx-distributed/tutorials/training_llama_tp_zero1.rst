.. _llama2_7b_tp_zero1_tutorial:

Training Llama3.1-8B, Llama3-8B and Llama2-7B with Tensor Parallelism and ZeRO-1 Optimizer
========================================================================================================

In this section, we showcase how to pre-train Llama3.1-8B, Llama3 8B and Llama2 7B model on four Trn1.32xlarge instances 
using the Neuron Distributed library. We will use AWS ParallelCluster to orchestrate the training jobs. 
To train the LLama model in this example, we will apply the following optimizations using the 
Neuron Distributed library:

1. :ref:`Tensor Parallelism <tensor_parallelism_overview>`
2. :ref:`Sequence Parallel <activation_memory_reduction>`
3. :ref:`Selective checkpointing <activation_memory_reduction>`
4. :ref:`ZeRO-1 <zero1-gpt2-pretraining-tutorial>`


Setting up environment:
^^^^^^^^^^^^^^^^^^^^^^^
                       
For this experiment, we will use AWS ParallelCluster with at least four Trn1.32xlarge compute nodes.
`Train your model on ParallelCluster <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/devflows/training/parallelcluster/parallelcluster-training.html>`__
introduces how to setup and use a ParallelCluster.
To setup the packages on the headnode of the ParallelCluster, follow the instructions mentioned here:
:ref:`Install PyTorch Neuron on Trn1 <setup-torch-neuronx>`.

We also need to install and clone the ``neuronx-distributed`` package inside the virtual env using the following commands:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com
   git clone git@github.com:aws-neuron/neuronx-distributed.git

Let’s download the scripts for pretraining:


1. Navigate to a directory to hold our experiments

.. literalinclude:: nxd-source-code/llama_tp_zero1/llama_tp_zero1_setup.sh
   :language: shell
   :lines: 4

2. Link the training scripts for our experiments

.. literalinclude:: nxd-source-code/llama_tp_zero1/llama_tp_zero1_setup.sh
   :language: shell
   :lines: 5-8

If you want to pre-train Llama3.1 8B, you would need to run the following steps -

.. literalinclude:: nxd-source-code/llama_tp_zero1/llama_31_8b.sh
   :language: shell
   :lines: 5-7

If you want to pre-train Llama3 8B, you would need to run the following steps -

.. literalinclude:: nxd-source-code/llama_tp_zero1/llama_3_8b.sh
   :language: shell
   :lines: 5-6


If you want to pre-train Llama2 7B, run the following steps -

.. literalinclude:: nxd-source-code/llama_tp_zero1/llama_2_7b.sh
   :language: shell
   :lines: 5-6

3. Installing the additional requirements

.. literalinclude:: nxd-source-code/llama_tp_zero1/llama_tp_zero1_setup.sh
   :language: shell
   :lines: 10


To tokenize the data, we must request the tokenizer from hugging face and meta by following the instructions at the following link: `HuggingFace Llama 3 8B Model <https://huggingface.co/meta-llama/Meta-Llama-3-8B>`__ . 

Use of the Llama models is governed by the Meta license. In order to download the model weights and tokenizer, please visit the above website and accept their License before requesting access. After access has been granted, you may use the following python3 script along with your own hugging face token to download and save the tokenizer.

Run the following from ``~/examples/tp_zero1_llama_hf_pretrain`` directory:

.. code:: ipython3

   from huggingface_hub import login
   from transformers import AutoTokenizer

   login(token='your_own_hugging_face_token')

   tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')  
   # For llama2 uncomment line below
   # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf') 

   tokenizer.save_pretrained(".")

For Llama3.1/Llama3, make sure your ``~/examples/tp_zero1_llama_hf_pretrain`` directory has the following files:

.. code:: ipython3

   './tokenizer_config.json', './special_tokens_map.json', './tokenizer.json'


For Llama2, you just copy the ``tokenizer.model`` to the ``~/examples/tp_zero1_llama_hf_pretrain`` directory.
Next let’s download and pre-process the dataset:

.. literalinclude:: nxd-source-code/llama_tp_zero1/llama_3_8b.sh
   :language: shell
   :lines: 11

`Note:` In case you see an error of the following form when downloading data: ``huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/ubuntu/examples/tp_zero1_llama_hf_pretrain'. Use `repo_type` argument if needed.`` 
This could be because of a stale cache. Try deleting the cache using: 

.. literalinclude:: nxd-source-code/llama_tp_zero1/llama_3_8b.sh
   :language: shell
   :lines: 8


At this point, you are all set to start training. The below tutorial uses ``Llama3 8B`` as an example. To run Llama2 7B, simply change the script from ``tp_zero1_llama3_8B_hf_pretrain.sh`` to ``tp_zero1_llama2_7B_hf_pretrain.sh``

Running training
^^^^^^^^^^^^^^^^

By this step, the ParallelCluster is all setup for running experiments. 
Before we run training, we first pre-compile the graphs using the :ref:`neuron_parallel_compile <pytorch-neuronx-parallel-compile-cli>`.
Let’s run the command below:

.. literalinclude:: nxd-source-code/llama_tp_zero1/llama_3_8b.sh
   :language: shell
   :lines: 15-18

This script uses a tensor-parallel size of 8.
This will automatically set the zero-1 sharding degree to 16 (4 * 32 workers / tensor_parallel_size). 

`Note`: You can use any number of nodes in this case, would just need to adjust the number of nodes in the above 
slurm command accordingly. Also, the number of nodes used in parallel_compile command should be same as the actual 
training run. This is because, as the number of nodes change, the data-parallel degree would change too. This would 
result in more workers participating in operations like `gradient all-reduce` which would result in new graphs getting 
created. 

Once the graphs are compiled we can now run training and observe our loss goes down.
To run the training, we just run the above command but without ``neuron_parallel_compile``.

.. literalinclude:: nxd-source-code/llama_tp_zero1/llama_3_8b.sh
   :language: shell
   :lines: 20-23


Performance:
^^^^^^^^^^^^

To achieve better performance, the script applies few techniques:

`Sequence Parallelism and Selective Activation Checkpointing`

As explained in the :ref:`Activation Memory Recomputation Doc <activation_memory_reduction>`, both `Sequence Parallelism` 
and `Selective activation checkpointing` can help with activation memory reduction thereby allowing us to fit bigger 
models with less number of devices. 
Please refer to :ref:`Activation Memory Reduction Developer Guide <activation_memory_reduction_developer_guide>` on how to 
enable sequence parallel and selective activation checkpointing.

`Coalescing Q, K, V layers:`

We coalesced parallel matrix multiply to improve throughput:

* We coalesced ``query``, ``key`` and ``value`` into one matrix multiply
* We coalesced ``gate_proj`` and ``up_proj`` into one matrix multiply

Please check ``modeling_llama_nxd.py`` and ``tp_dp_gpt_neox_20b_hf_pretrain.py`` for details.
`Note:` Because we coalesced the layers above, the `pretrained checkpoint provided here <https://huggingface.co/meta-llama/Llama-2-7b>`__ 
cannot be loaded out of the box for fine-tuning, and would require preprocessing. The Q,K,V layers 
and the gate_proj and up_proj layers need to be coalesced in the checkpoint before loading.

`Logging:`

Currently for better performance we log loss values every 10 steps. Logging frequently will result in frequent 
syncs between device and CPU which are expensive. Hence, it is recommended to do less frequent logging if possible.


`Flash Attention:`

We're introducing flash attention function for better performance/memory efficiency. Currently it's enabled by default, to disable it
set ``--use_flash_attention 0`

Checkpointing:
^^^^^^^^^^^^^^

Currently by default, the checkpoint is saved at the end of training. You can modify that behaviour by saving 
the checkpoint after every `N steps` inside the training loop:

.. code:: ipython3

   from neuronx_distributed.parallel_layers import checkpointing
   if global_step % every_n_steps_checkpoint == 0:
      state_dict = {
         "model": model.state_dict(),
         "global_step": global_step,
         "epoch": epoch,
         "scheduler": scheduler.state_dict()
      }
      checkpointing.save(state_dict, flags.output_dir)
      optimizer.save_sharded_state_dict(flags.output_dir)

Here we have to save the model state_dict using the `checkpointing.save` API and the optimizer state_dict using 
the `optimizer.save_sharded_state_dict`. This is because, currently, `checkpointing.save` API only saves on 
data-parallel rank 0, while in case of Zero1 Optimizer, the optimizer states are distributed across all data-parallel 
ranks. Hence, we use Zero1 Optimizer's save API to save the optimizer states.

`Time to save a checkpoint:`

Checkpoint save time can vary depending on what location the checkpoint is saved. If the checkpoint is saved in 
the `home` directory, the checkpointing time can be higher. The same time can be reduce by 4x if the checkpoint 
is dumped to FSX file system. 

By default, `checkpoint.save` API allows one tensor-parallel rank at a time to save the checkpoint. This is done 
in order to avoid HOST OOM. When all tensor-parallel ranks try to save at the same time, they would end up copying 
weights to CPU at the same time. This can result in HOST OOM. `Note:` Since, we use `XLA_DOWNCAST_BF16` flag for 
BF16 training, even though the weights on device are on bf16, the weights on CPU are copied in FP32 format. In case, 
you want to avoid this typecasting from BF16 to FP32 when copying weights from device to CPU for checkpoint saving, 
you can pass `down_cast_bf16=True` to the checkpointing.save API as follows:

.. code:: ipython3

   from neuronx_distributed.parallel_layers import checkpointing
   if global_step % every_n_steps_checkpoint == 0:
      state_dict = {
         "model": model.state_dict(),
         "global_step": global_step,
         "epoch": epoch,
         "scheduler": scheduler.state_dict()
      }
      checkpointing.save(state_dict, flags.output_dir, down_cast_bf16=True)

This should not only reduce the HOST memory pressure when saving weights, but at the same time reduce model checkpointing 
time by half. `Note:` We are saving checkpoint in sharded format, wherein each tensor-parallel rank is 
saving one shard. To deploy these pretrained models, one would have to combine these shards by loading them and 
concatenating the tensor-parallel layers together. (We are working on a checkpoint conversion script that 
combines the shards into a single checkpoint)

In addition to the above method, if we want to speed up checkpoint saving for the model further, we can do so by:

.. code:: ipython3

   from neuronx_distributed.parallel_layers import checkpointing
   if global_step % every_n_steps_checkpoint == 0:
      state_dict = {
         "model": model.state_dict(),
         "global_step": global_step,
         "epoch": epoch,
         "scheduler": scheduler.state_dict()
      }
      checkpointing.save(state_dict, flags.output_dir, down_cast_bf16=True, save_xser=True)

The `save_xser` uses torch-xla's `xser.save <https://pytorch.org/xla/release/2.1/index.html#saving-and-loading-xla-tensors>`__ 
to save the tensors serially. This API will copy one tensor at a time to the disk. This will allow all the ranks to 
save the checkpoint at the same time. This speeds up checkpoint saving especially for large models as all ranks 
are saving at the same time. Moreover, the risk of HOST OOM is completely eliminated because only one tensor is copied 
to CPU at a time. 

`Note:` If we use `save_xser` to save the checkpoint, we would have to pass `load_xser` to the 
`checkpoint.load` API. 
Also, if you use `save_xser`, the checkpoint folder would contain a `.pt` file for each tensor instead of a 
single `.pt` for the entire state_dict. To read this checkpoint in your checkpoint conversion script, you would 
have to use `xser.load <https://pytorch.org/xla/release/2.1/index.html#saving-and-loading-xla-tensors>`__ API 
instead of `torch.load` to load the checkpoint. The `xser.load` should load the serialized checkpoint and return 
the full state_dict.

Finally, to speed up optimizer saving time, you can increase the number of workers saving at the same time. 
This can be done as follows:

.. code:: ipython3

   if global_step % every_n_steps_checkpoint == 0:
      ...
      optimizer.save_sharded_state_dict(flags.output_dir, num_workers_per_step=32)

By default, `num_workers_per_step` is set to 8.

