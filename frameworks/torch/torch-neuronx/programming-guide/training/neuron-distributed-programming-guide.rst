.. _neuronx-distributed:

NeuronxDistributed
==================

.. contents:: Table of Contents
   :local:
   :depth: 2

Neuronx Distributed is a package for supporting different distributed
training/inference mechanism for Neuron devices. It would provide xla
friendly implementations of some of the more popular distributed
training/inference techniques. As the size of the model scales, fitting
these models on a single device becomes impossible and hence we have to
make use of model sharding techniques to partition the model across
multiple devices. As part of this library, we enable support for Tensor
Parallel sharding technique.

Tensor Parallelism
~~~~~~~~~~~~~~~~~~

Tensor Parallelism is a technique in which a tensor is split into N
chunks along a particular dimension such that each device only holds 1/N
chunk of the tensor. Computation is performed using this partial chunk
so as to get partial output. These partial outputs are collected from
all devices ensuring the correctness of the computation is maintained.

Taking a general matrix multiplication as an example, let’s say we have
C = AB. We can split B along the column dimension into [B0 B1 B2 … Bn]
and each device holds a column. We then multiply A with each column in B
on each device, we will get [AB0 AB1 AB2 … ABn]. At this moment, each
device still holds partial results, e.g. device rank 0 holds AB0. To
make sure the result is correct, we need to all-gather the partial
result and concatenate the tensor along the column dimension. In this
way, we are able to distribute the tensor over devices while making sure
the computation flow remains correct.

.. image:: /libraries/neuronx-distributed/images/tp.png

Fig and TP explanation is borrowed from [1]

Similarly we can perform the partition along the row dimensions and
create a RowParallel Linear layer. In RowParallelLinear layer, we
partition the weight matrix along the row dimension. Let’s say we have C
= AB. We can split B along the row dimension into [B0 B1 B2 … Bn] and
each device holds a row. We then multiply each column of A on each
device, we will get [A0B0 A1B1 A2B2 … AnBn]. At this moment, each device
still holds partial results, e.g. device rank 0 holds A0B0. To make sure
the result is correct, we need to all-reduce sum the partial result from
all devices to produce the final output.

Using this principle of sharded linear layers, we can construct MLPs of
arbitrary depth until the need to operate on the whole output tensor, in
which case we would have to construct the output but gathering it from
all devices.

.. image:: /libraries/neuronx-distributed/images/mlp.png

Here is an illustration from the Megatron-LM paper In the above case, as
you can see two linear layers are implemented using Column Parallel and
Row Parallel linear layers, wherein the ColumnParallel Linear shards
along the columns and then it is followed by RowParallel Linear layer
which takes in parallel inputs (sharded outputs from
ColumnParallelLinear). Consider the example shown in the above diagram,
Z = (X\ *A)*\ B. In this case we split the first matrix multiplication
over column dimension such that each device after first matrix
multiplication holds partial result of Y0=XA0,Y1=XA1 and so on. For the
second matrix multiplication, we partition the weight matrix over row
dimension and since the inputs are already columns sharded and we can
multiply them to produce partial outputs. These outputs finally requires
an all-reduce sum, since we want to sum up the single column*row result.

Tensor Parallelism for Transformers: A transformer block

.. image:: /libraries/neuronx-distributed/images/self-attention.png

Fig: Taken from Megatron-LM paper As seen from the figure above, a
simple self attention block has the QKV linear layer followed by MLP.
Using the same Column and Row Parallel linear layers, we can partition
the self-attention block across devices thereby reducing the memory
footprint on each device, since each device now only holds partial
parameters. This weight distribution strategy allows us to scale large
model training across devices.

API Definitions:
~~~~~~~~~~~~~~~~

To support tensor-parallelism on Neuron, we adopted the Apex Library
built for CUDA devices. We modified the implementations to work with
XLA. Here are the tensor-parallel APIs that can be used to enable tensor
parallelism:

Parallel Model State:
^^^^^^^^^^^^^^^^^^^^^

Initialize Model Parallelism:
'''''''''''''''''''''''''''''

::

   def neuronx_distributed.parallel_state.initialize_model_parallel(
           tensor_model_parallel_size=1)

This module would initialize the distributed model training and allows
users to set the number of tensor_parallel world size.

Parameters:
           

``tensor_model_parallel_size`` : This should set the number of tensor
parallel workers. Note the default value is set to 1

Other helper APIs:
''''''''''''''''''

-  ``neuronx_distributed.parallel_state.get_data_parallel_size()`` :
   Returns the data parallel world size depending on the number of
   global workers and tensor parallel workers.
-  ``neuronx_distributed.parallel_state.get_tensor_model_parallel_size()``
   : Returns the tensor parallel world size.
-  ``neuronx_distributed.parallel_state.get_tensor_model_parallel_rank()``
   : Returns the rank of the worker within the tensor parallel group
-  ``neuronx_distributed.parallel_state.get_data_parallel_rank()`` :
   Returns the rank of the worker in the data parallel group.
-  ``neuronx_distributed.parallel_state.get_data_parallel_group(as_list=False)``
   : Returns the data parallel group after taking into account the
   tensor parallel size and the global world size. as_list argument when
   set to True, would return the group as a List[List] otherwise it
   would return a torch.distributed.group.
-  ``neuronx_distributed.parallel_state.get_tensor_model_parallel_group(as_list=False)``
   : Returns the tensor parallel group after taking into account the
   tensor parallel size and the global world size. as_list argument when
   set to True, would return the group as a List[List] otherwise it
   would return a torch.distributed.group.
- ``move_model_to_device(model, device)``: This api moves the model to device by 
  preserving tensor parallel attributes.

Parallel Layers:
^^^^^^^^^^^^^^^^

Majority of parameters within the transformer based model reside in the
Embedding and Linear layers. Hence, to reduce the number of parameters
on a single device because of these layers, we provided sharded
Embedding and Linear layers.

Parallel Embedding:
'''''''''''''''''''

::

   class neuronx_distributed.parallel_layers.ParallelEmbedding(
       num_embeddings, embedding_dim, init_method=init.normal_,
       dtype=torch.float32, device=None)

This module is intended to replace torch.nn.Embedding . In cases where
the vocab size is too large, we can shard the Embedding table across
workers. Note: The embedding table would be sharded across all the
tensor-parallel workers.

.. _parameters-1:

Parameters:
           

-  ``num_embeddings (int)`` : size of the dictionary of embeddings
-  ``embedding_dim (int)`` : the size of each embedding vector
-  ``init_method: (torch.nn.init)`` : Initialization function for the
   embedding weights.
-  ``dtype: (dtype)`` : Datatype for the weights
-  ``device: (torch.device)`` : Device to initialize the weights on. By
   default, the weights would be initialized on CPU

ColumnParallel Linear Layer:
''''''''''''''''''''''''''''

::

   class neuronx_distributed.parallel_layers.ColumnParallelLinear(
       input_size, output_size, bias=True, gather_output=True,
       dtype=torch.float32, device=None)

This module would perform a Column wise partition of the weight matrix.
Linear layer is defined as ``Y = XA + b`` , here A is parallelized along
second dimension as ``A = [A_1, A_2 .... A_p]`` . ``Note``: This layer
is designed to operate on 3-dimensional inputs.

.. _parameters-2:

Parameters:
           

-  ``input_size: (int)`` : First dimension of the weight matrix
-  ``output_size: (int)`` : Second dimension of the weight matrix
-  ``bias: (bool)``: If set to True, bias would be added
-  ``gather_output: (bool)`` : If true, call all-gather on output and
   make Y available to all Neuron devices, otherwise, every Neuron
   device will have its output which is Y_i = XA_i
-  ``dtype: (dtype)`` : Datatype for the weights
-  ``device: (torch.device)`` : Device to initialize the weights on. By
   default, the weights would be initialized on CPU

RowParallel Linear Layer:
'''''''''''''''''''''''''

::

   class neuronx_distributed.parallel_layers.RowParallelLinear(
       input_size, output_size, bias=True, input_is_parallel=False,
       dtype=torch.float32, device=False
   )

The linear layer is defined as ``Y = XA + b``. A is parallelized along
its first dimension and X along its second. ``Note``: This layer is
designed to operate on 3-dimensional inputs.

.. _parameters-3:

Parameters:
           

-  ``input_size: (int)`` : First dimension of the weight matrix
-  ``output_size: (int)`` : Second dimension of the weight matrix
-  ``bias: (bool)`` : If set to True, bias would be added
-  ``input_is_parallel: (bool)`` : If true, we assume that the input is
   already split across the Neuron devices and we do not split again.
   This is useful when we have a ColumnParallel Layer just before the
   Row Parallel layer
-  ``dtype: (dtype)`` : Datatype for the weights
-  ``device: (torch.device)`` : Device to initialize the weights on. By
   default, the weights would be initialized on CPU

Checkpointing:
^^^^^^^^^^^^^^

These are set of APIs for saving and loading the checkpoint. These APIs
take care of saving and loading the shard depending the tensor parallel
rank of the worker.

Save Checkpoint:
''''''''''''''''

::

   def neuronx_distributed.parallel_layers.save(state_dict, save_dir)

This API will save the model from each tensor-parallel rank in the
save_dir . Only workers with data parallel rank equal to 0 would be
saving the checkpoints. Each tensor parallel rank would be creating a
``tp_rank_i`` folder inside ``save_dir`` and each ones saves its shard
in the ``tp_rank_i`` folder.

.. _parameters-4:

Parameters:
           

-  ``state_dict: (dict)`` : Model state dict. Its the same dict that you
   would save using torch.save
-  ``save_dir: (str)`` : Model save directory.

Load Checkpoint
'''''''''''''''

::

   def neuronx_distributed.parallel_layers.load(
       load_dir, model=None, model_key='model', sharded=True)

This API will automatically load checkpoint depending on the tensor
parallel rank. For large models, one should pass the model object to the
load API to load the weights directly into the model. This could avoid
host OOM, as the load API would load the checkpoints for one tensor
parallel rank at a time.

.. _parameters-5:

Parameters:
           

-  ``load_dir: (str)`` : Directory where the checkpoint is saved.
-  ``model``: (torch.nn.Module): Model object
-  ``model_key: (str)`` :The model key used when saving the model in the
   state_dict.
-  ``sharded: (bool)`` : If the checkpoint is not sharded, pass False.
   This is useful (especially during inference) when the model is
   trained using a different strategy and you end up saving a single
   unsharded checkpoint. You can then load this unsharded checkpoint
   onto the sharded model. When this attribute is set to ``False`` , it
   is necessary to pass the model object. Note: The keys in the
   state-dict should have the same name as in the model object, else it
   would raise an error.

Gradient Clipping:
''''''''''''''''''

With tensor parallelism, we need to handle the gradient clipping as we
have to accumulate the total norm from all the tensor parallel ranks.
This should be handled by the following API

::

   def neuronx_distributed.parallel_layers.clip_grad_norm(
       parameters, max_norm, norm_type=2)

.. _parameters-6:

Parameters:
           

-  ``parameters (Iterable[Tensor] or Tensor)`` : an iterable of Tensors
   or a single Tensor that will have gradients normalized
-  ``max_norm (float or int)`` :max norm of the gradients
-  ``norm_type (float or int)`` : type of the used p-norm. Can be ‘inf’
   for infinity norm.

Model Trace:
^^^^^^^^^^^^

We can use the tensor parallel layers to perform large model inference
too. For performing inference, we can re-use the Parallel model built
above for training and then use the trace APIs provided by the
neuronx_distributed package to trace it for inference. One can use the
following set of APIs for running distributed inference:

::

   def neuronx_distributed.trace.parallel_model_trace(func, inputs, tp_degree=1)

This API would launch tensor parallel workers, where each worker would
trace its own model. These traced models would be wrapped with a single
TensorParallelModel module which can then be used like any other traced
model.

.. _parameters-7:

Parameters:
           

-  ``func : (Function)``: This is a function that returns a ``Model``
   object. The ``parallel_model_trace`` API would call this function
   inside each worker and run trace against them. Note: This differs
   from the ``torch_neuronx.trace`` where the ``torch_neuronx.trace``
   requires a model object to be passed.
-  ``inputs: (torch tensors)`` : The inputs that needs to be passed to
   the model.
-  ``tp_degree: (int)`` : How many devices to be used when performing
   tensor parallel sharding

Trace Model Save/Load:
^^^^^^^^^^^^^^^^^^^^^^

Save:
'''''

::

   def neuronx_distributed.trace.parallel_model_save(model, save_dir)

This API should save the traced model in save_dir . Each shard would be
saved in its respective directory inside the save_dir. Parameters:

-  ``model: (TensorParallelModel)`` : Traced model produced using the
   parallel_model_trace api.
-  ``save_dir: (str)`` : The directory where the model would be saved

Load:
'''''

::

   def neuronx_distributed.trace.parallel_model_load(load_dir)

This API will load the sharded traced model into ``TensorParallelModel``
for inference.

.. _parameters-8:

Parameters:
'''''''''''

-  ``load_dir: (str)`` : Directory which contains the traced model.

Developer guide
~~~~~~~~~~~~~~~

Training
^^^^^^^^

For training models with tensor-parallelism, one would have to make few
changes to their model/training script. Below we walk through the
different changes one would have to make to shard the models across
devices.

Creating DataLoader:
''''''''''''''''''''

When we shard the model across devices using tensor parallelism, all the
tensor parallel workers are operating on the same batch of data. Hence,
to ensure that each tensor parallel worker is getting the same data, we
make use of ``DistributedSampler`` as shown in the snippet below

.. code:: ipython3

   def create_pretraining_dataset(
       input_file, max_pred_length, mini_batch_size, worker_init
   ):
       train_data = pretraining_dataset(
           input_file=input_file, max_pred_length=max_pred_length
       )
       # To distribute the data across different workers in the world, 
       # we use the DistributedSampler. The num_replicas should be equal
       # to the data_parallel_world_size. Note: data_parallel_rank=0 can have
       # multiple tensor parallel ranks and each of these should get the same 
       # data. 
       train_sampler = DistributedSampler(
           train_data,
           num_replicas=parallel_state.get_data_parallel_world_size(),
           rank=parallel_state.get_data_parallel_rank(),
       )
       train_dataloader = DataLoader(
           train_data,
           sampler=train_sampler,
           batch_size=mini_batch_size,
           num_workers=0,
           worker_init_fn=worker_init,
           drop_last=True,
           pin_memory=True,
       )
       return train_dataloader

Creating Model:
'''''''''''''''

One can create models by replacing the large linear layers with
``ColumnParallel`` and ``RowParallel`` Linear layers. In case of
transformers, we have a good structure where the Attention block usually
have linear projections for QKV and this is followed by a fully
connected layer. Let’s take a look at the example for the BERT model. We
make the attention module of BERT model to use tensor parallel layers,
thereby adding the ability to shard the model across devices.

.. code:: ipython3

   class ParallelSelfAttention(transformers.models.bert.modeling_bert.BertSelfAttention):
       def __init__(self, config, position_embedding_type=None):
           super().__init__(config, position_embedding_type)

           self.query = ColumnParallelLinear(config.hidden_size,
                                             self.all_head_size,
                                             gather_output=False)
           self.key = ColumnParallelLinear(config.hidden_size,
                                           self.all_head_size,
                                           gather_output=False)
           self.value = ColumnParallelLinear(config.hidden_size,
                                             self.all_head_size,
                                             gather_output=False)
           # Since we shard the number of attention heads across tensor parallel
           # ranks, each rank would have a subset of heads, hence, we update
           # the num_attention_heads here.
           tp_size = parallel_state.get_tensor_parallel_size()
           self.num_attention_heads = self.num_attention_heads // tp_size
           self.all_head_size = self.all_head_size // tp_size

As seen we just had to swap out the linear layers with ColumnParallel
Linear layers and the rest of the forward method of the attention layer
can work as is. Note: In the above ColumnParallelLinear layer we are not
gathering output from each rank, in other words, each ranks is working
on its own shard. We can make gather_output=True and that would gather
output and you would get a full dim output. However, gathering output
from all ranks would introduce an all-gather operation which can be
expensive depending on the size of the tensor. In the case of attention
module, we know that the SelfAttention block is followed by MLP block.
Hence, we replace the linear layer there with a RowParallelLinear as
shown below:

.. code:: ipython3

   class ParallelSelfOutput(transformers.models.bert.modeling_bert.BertSelfOutput):
       def __init__(self, config):
           super().__init__(config)
           self.dense = RowParallelLinear(config.hidden_size,
                                          config.hidden_size,
                                          input_is_parallel=True)

As seen we just had to replace the dense layer here, and pass the
``input_is_parallel`` argument. This way, the ``RowParallelLinear``
should operator on partitions and get a collective result.

Making just the above two changes can help you partition good chunk of
your model across multiple workers, thereby allowing models of larger
size to be trained on a single instance. Note: Majority of the
parameters of a transformer model are in these linear layers and hence
partitioning these layers can help you scale.

Final Training script:
''''''''''''''''''''''

Once the dataloader and model changes are done, we are ready to build
the training script. Good news, you can use the same training loop as
before for data-parallel training, and would need just the minor tweaks
to get it all started.

.. code:: ipython3

   from neuronx_distributed.parallel_layers import parallel_state, clip_grad_norm

   neuronx_distributed.parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
   dataloader = create_pretraining_dataset(
    input_file, max_pred_length, mini_batch_size, worker_init)

   model = YourNewlyBuiltParallelModel(config)
   # We have to move the model to device using this API, because when
   # we move model to device using .to(device), the model parameter's
   # attributes aren't preserved. This causes some of the tensor parallel
   # attributes to be lost. Hence, this API takes care of preserving the
   # tensor parallel attributes.
   parallel_layers.move_model_to_device(model, device)

   for inputs, labels in dataloader:
       output = model(*inputs)
       loss = loss_fn(output, labels)
       loss.backward()
       # Here we use clip_grad_norm from neuronx_distributed as that 
       # can handle tensor parallel ranks
       clip_grad_norm(model.parameters(), max_norm)
       # For the optimzer step, we have to pass the data_parallel group
       xm.optimizer_step(
           optimzer, 
           groups=parallel_state.get_data_parallel_group(as_list=True)
       )
       optimizer.zero_grad()
       scheduler.step()

Few things to take note of in the above code snippet: 1. We are
initializing the model parallel with tensor parallel size of 2. This
will shard the model across 2 devices. 2. We use the
``move_model_to_device`` API to move model to device. This is equivalent
to doing ``model.to(device)``. We need to explicity call this API since
some of the tensor-parallel attributes do not get copied over when we
move the model to device using ``model.to(device)``. 3. We are calling
the ``clip_grad_norm`` from ``parallel_layers``. This clip_grad_norm
should take care of accumulating the max_norm from the tensor_parallel
ranks and producing the correct output. 4. We pass the
``data_parallel_group`` to the ``optimizer_step``. If we don’t pass the
group, default would be all the workers in the world.

Saving Model:
'''''''''''''

Once training is done, we want to save the model. This can be done
easily by calling the save api from
``neuronx_distributed.parallel_layers`` . Here is an example:

.. code:: ipython3

   neuronx_distributed.parallel_layers.save({
               'epoch': epoch,
               'model': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'loss': loss,
               ...
               }, PATH)

Note the ``model`` key used here, we need to provide the same key during
model load.

Training Tutorial:
^^^^^^^^^^^^^^^^^^

Keeping the above changes in mind, let’s now run an end-to-end trainging
with tensor-parallelism. This section is adopted from `BERT pretraining
tutorial <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/bert.html#hf-bert-pretraining-tutorial>`__
which used data-parallel training to scale the throughput. In this
section we modify that tutorial to showcase the use of
tensor-parallelism which should enable us to scale the size of the
model.

Setting up environment:
                       

For this experiment, we will use a trn1-32xl machine with the storage
set to 512GB atleast. Next follow the instructions mentioned here:
`Install PyTorch Neuron on
Trn1 <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/setup/pytorch-install.html#pytorch-neuronx-install>`__
to create a pytorch environment. It is recommended to work out of python
virtual env so as to avoid package installation issues.

We also have to install the ``neuronx-distributed`` package using the
following command:

.. code:: ipython3

   python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com

Make sure the transformers version is set to ``4.26.0``

Let’s download the scripts and datasets for pretraining.

.. code:: ipython3

   mkdir -p ~/examples/tp_dp_bert_hf_pretrain
   cd ~/examples/tp_dp_bert_hf_pretrain
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_dp_bert_hf_pretrain/tp_dp_bert_large_hf_pretrain_hdf5.py
   wget https://raw.githubusercontent.com/aws-neuron/aws-neuron-samples/master/torch-neuronx/training/tp_dp_bert_hf_pretrain/requirements.txt
   python3 -m pip install -r requirements.txt

Next let’s download the tokenizer and the sharded datasets:

.. code:: ipython3

   mkdir -p ~/examples_datasets/
   pushd ~/examples_datasets/
   aws s3 cp s3://neuron-s3/training_datasets/bert_pretrain_wikicorpus_tokenized_hdf5/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar .  --no-sign-request
   tar -xf bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar
   rm bert_pretrain_wikicorpus_tokenized_hdf5_seqlen128.tar
   aws s3 cp s3://neuron-s3/training_datasets/bert_pretrain_wikicorpus_tokenized_hdf5/bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512.tar .  --no-sign-request
   tar -xf bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512.tar
   rm bert_pretrain_wikicorpus_tokenized_hdf5_seqlen512.tar
   popd

At this point, you are all set to start training

Running training
                

We first pre-compile the graphs using the ``neuron_parallel_compile``.
This process is similar to one discussed in the `BERT pretraining
tutorial <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/bert.html#hf-bert-pretraining-tutorial>`__
. Let’s run the command below:

.. code:: ipython3

   cd ~/examples/tp_dp_bert_hf_pretrain
   neuron_parallel_compile XLA_DOWNCAST_BF16=1 torchrun --nproc_per_node=32 \
   tp_dp_bert_large_hf_pretrain_hdf5.py \
   --tensor_parallel_degree 2 \
   --steps_this_run 10 \
   --batch_size 16 \
   --grad_accum_usteps 32 |& tee compile_log.txt

This script uses a tensor-parallel size of 2. This will automatically
set the data-parallel degree to 16 (32 workers / tensor_parallel_size).
Once the graphs are compiled we can now run training and observe our
loss go down. To run the training, we just the above command but without
``neuron_parallel_compile``.

.. code:: ipython3

   XLA_DOWNCAST_BF16=1 torchrun --nproc_per_node=32 \
   tp_dp_bert_large_hf_pretrain_hdf5.py \
   --tensor_parallel_degree 2 \
   --steps_this_run 10 \
   --batch_size 16 \
   --grad_accum_usteps 32 |& tee compile_log.txt

You would notice that the throughput is lower when you run the
``dp_bert_large_hf_pretrain_hdf5.py``. This is expected as the number of
data-parallel workers have gone down (from 32 to 16). However, if you
open ``neuron-top`` in another terminal, you should see the memory
utilization per core for this script is lower than the
``dp_bert_large_hf_pretrain_hdf5.py``. Since the memory requirement has
gone down, you can scale the size of model either by increasing the
number of layers/attention heads/hidden sizes.

The loss curve should match to the loss curve we would get from the
data_parallel counterpart.

Inference
~~~~~~~~~

For running model inference, we would need to trace the distributed
model. Before we run the inference, let’s get a checkpoint that we can
use. Let’s run the below block of code:

.. code:: ipython3

    import torch_neuronx
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
    torch.save({"model":model.state_dict()}, "bert/bert.pt")

If you already have a checkpoint from the above training or by running
training from another source, feel free to skip the above step.

Once we have the checkpoint we are ready to trace the model and run
inference against it. Let’s look at the example below:

.. code:: ipython3

    import os
    import neuronx_distributed
    from neuronx_distributed.parallel_layers import layers, parallel_state
    import torch
    import torch_neuronx
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput
    
    
    def encode(tokenizer, *inputs, max_length=128, batch_size=1):
        tokens = tokenizer.encode_plus(
            *inputs,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return (
            torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
            torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
            torch.repeat_interleave(tokens['token_type_ids'], batch_size, 0),
        )
    
    
    # Create the tokenizer and model
    name = "bert-base-cased-finetuned-mrpc"
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    
    # Set up some example inputs
    sequence_0 = "The company HuggingFace is based in New York City"
    sequence_1 = "Apples are especially bad for your health"
    sequence_2 = "HuggingFace's headquarters are situated in Manhattan"
    
    paraphrase = encode(tokenizer, sequence_1, sequence_2)
    not_paraphrase = encode(tokenizer, sequence_1, sequence_1)
    
    def get_model():
        model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
        # Here we build a model with tensor-parallel layers.
        # Note: If you already have a Model class that does this, we can use that directly
        and load the checkpoint in it.
        class ParallelSelfAttention(BertSelfAttention):
            def __init__(self, config, position_embedding_type=None):
                super().__init__(config, position_embedding_type)
                self.query = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
                self.key = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
                self.value = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
                self.num_attention_heads = self.num_attention_heads // parallel_state.get_tensor_model_parallel_size()
                self.all_head_size = self.all_head_size // parallel_state.get_tensor_model_parallel_size()
    
        class ParallelSelfOutput(BertSelfOutput):
            def __init__(self, config):
                super().__init__(config)
                self.dense = layers.RowParallelLinear(config.hidden_size,
                                           config.hidden_size,
                                           input_is_parallel=True)
    
        for layer in model.bert.encoder.layer:
            layer.attention.self = ParallelSelfAttention(model.config)
            layer.attention.output = ParallelSelfOutput(model.config)
        
        # Here we created a checkpoint as mentioned above. We pass sharded=False, since the checkpoint
        # we obtained is unsharded. In case you are using the checkpoint from the tensor-parallel training,
        # you can set the sharded=True, as that checkpoint will contain shards from each tp rank.
        neuronx_distributed.parallel_layers.load("bert/bert.pt", model, sharded=False)
        
        return model
    
    # Note how we are passing a function that returns a model object, which needs to be traced.
    # This is mainly done, since the model initialization needs to happen within the processes
    # that get launched internally within the parallel_model_trace.
    model = neuronx_distributed.trace.parallel_model_trace(get_model, paraphrase, tp_degree=2)
    
    # Once traced, we now save the trace model for future inference. This API takes care
    # of saving the checkpoint from each tensor parallel worker
    neuronx_distributed.trace.parallel_model_save(model, "tp_models")
    
    # We now load the saved model and will run inference against it
    model = neuronx_distributed.trace.parallel_model_load("tp_models")
    
    print(model(*paraphase))

Known Issues:
~~~~~~~~~~~~~

1. Currently the checkpoints dumped during training are sharded and
   users would have to write a script to combine the checkpoints
   themselves. This should be fixed in the future release

