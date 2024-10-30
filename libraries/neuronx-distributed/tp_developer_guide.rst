.. _tp_developer_guide:

Developer guide for Tensor Parallelism 
=================================================================

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