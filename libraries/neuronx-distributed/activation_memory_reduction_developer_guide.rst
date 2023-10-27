.. _activation_memory_reduction_developer_guide:

Developer guide for Activation Memory reduction (``neuronx-distributed`` )
=================================================================

Sequence Parallelism
^^^^^^^^^^^^^^^^^^^^

To combine sequence parallelism with tensor-parallelism, one needs to follow the steps below:

Model changes for Tensor-Parallel block:
'''''''''''''''''''''''''''''''''''''''

For tensor-parallelism, we replace the linear layers with ColumnParallel and RowParallel Linear 
layers as mentioned `here <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/tp_developer_guide.html#creating-model>`__.
To enable sequence-parallel, we need to pass the `sequence_parallel_enabled` for the ColumnParallel and RowParallel linear layers.
Setting this argument to `true`, the ColumnParallel and RowParallel Linear layers will introduce the `all-gather` and `reduce-scatter` 
operations for gathering and distributing the activations along the sequence dimension.

.. code:: ipython3
   
   from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention

   class class GPTNeoXAttentionNxD(GPTNeoXAttention):
       def __init__(self, config):
           super().__init__(config)
           ....
           self.query_key_value = ColumnParallelLinear(
                                    config.hidden_size,
                                    3 * config.hidden_size,
                                    stride=3,
                                    gather_output=False,
                                    init_method=init_method,
                                    sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                                )
           self.dense = RowParallelLinear(
                            config.hidden_size,
                            config.hidden_size,
                            input_is_parallel=True,
                            init_method=init_method,
                            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                        )
           ....

Model changes for Non-Tensor-Parallel block:
''''''''''''''''''''''''''''''''''''''''''''

In a transformer module, the non-tensor parallel block contains mainly the Layer-Norm modules. Since we partition 
the computation along the sequence dimension for the layer-norm, we 
need to sum up the gradients along the sequence dimension for the Layer-norm. To help us do that, 
we use the Layer-norm provided from `neuronx-distributed.parallel_layers.layer_norm`. The Layer-norm in 
neuronx-distributed should uses the same forward and backward as `torch.nn.LayerNorm`, however, it just marks
the weights as sequence-parallel weights. This tagging allows us to look for weights with sequence-parallel 
tagging and reduce those gradients along the tensor-parallel degree. Hence we need to add the following two changes:


.. code:: ipython3

   from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
   from neuronx_distributed.parallel_layers import layer_norm

   class GPTNeoXLayerNxD(GPTNeoXLayer):
       def __init__(self, config):
           super().__init__(config)
           ...
           self.input_layernorm = layer_norm.LayerNorm(
                                    config.hidden_size,
                                    eps=config.layer_norm_eps,
                                    sequence_parallel_enabled=config.sequence_parallel_enabled
                                  )
           self.post_attention_layernorm = layer_norm.LayerNorm(
                                                config.hidden_size,
                                                eps=config.layer_norm_eps,
                                                sequence_parallel_enabled=config.sequence_parallel_enabled
                                            )

Once we replace the layernorm with neuronx-distributed's layernorm, it will `mark the weights <https://github.com/aws-neuron/neuronx-distributed/blob/main/src/neuronx_distributed/parallel_layers/layer_norm.py#L32>`__ 
as sequence-parallel weights. Note: If your model is using RMSNorm or any other layer that parallelizes in the sequence-dimension,
you can mark the weights as sequence-parallel weights by using the following code:

.. code:: ipython3

    setattr(param, "sequence_parallel_enabled", sequence_parallel_enabled)

Once marked, we then use this attribute when we compute gradients for layer-norm. We need to add the following code before our optimizer.step:

.. code:: ipython3

    def allreduce_sequence_parallel_gradients(optimizer):
        """ All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
            Modified from megatron-lm:
            https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
        """
        from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
        grads = []
        for param_group in optimizer.__getstate__()['param_groups']:
            for group, params in param_group.items():
                if group == 'params':
                    for p in params:
                        if isinstance(p, torch.Tensor) and p.grad is not None:
                            sequence_parallel_param = getattr(p, 'sequence_parallel_enabled', False)
                            if sequence_parallel_param:
                                grads.append(p.grad.data)
        for grad in grads:
            reduce_from_tensor_model_parallel_region(grad)

As seen in the above code, we reduce the gradients from all tensor parallel devices. This is because the compute is divided along the 
sequence dimension across all the devices participating in the tensor parallel group. For reference implementation, check 
the `GPTNeoX-20B modeling code <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain.py#L273C1-L289C55>`__ .

Transposing the activations:
''''''''''''''''''''''''''''

Sequence-parallelism implementation requires the sequence dimension to be the 0th dimension whereas the tensor-parallel region 
requires the sequence dimension to be the first dimension. All our model implementation keeps the sequence dimension 
as 1st dimension and batch dimension as 0th dimnesion. Hence, to accomodate sequence parallelism, we need to insert a few 
transpose operations at the following places:

1. Before we start looping through all the layers, we need to transpose the sequence and batch dimension. We 
also need to partiton the inputs along the sequence dimensions such that each tp-rank gets a part. This can be done as:

.. code:: ipython3

    form neuronx_distributed.parallel_layers.mappings import scatter_to_sequence_parallel_region
    # NxD code change: sequence parallel uses seq_len as the 0-th dim
    if self.config.sequence_parallel_enabled:
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states = scatter_to_sequence_parallel_region(hidden_states)

2. Since the attention block requires the sequence dimension to be 1st dimension, we transpose the output of QKV projection and then 
 transpose it back before the final MLP of the attention block. 

.. code:: ipython3

    # Within the attention module
    qkv = self.query_key_value(hidden_states)

    if config.sequence_parallel_enabled:
        qkv = qkv.transpose(0,1)
    ...

    attn_output = attn_output.transpose(0,1)
    attn_output = self.dense(attn_output)


3. Finally before returning the final output, we need to put all the partial activations along the sequence dimension 
back together. This can be done as follows:

.. code:: ipython3

    form neuronx_distributed.parallel_layers.mappings import gather_from_sequence_parallel_region
    if self.config.sequence_parallel_enabled:
        hidden_states = gather_from_sequence_parallel_region(hidden_states, to_model_parallel=False)
        hidden_states = hidden_states.transpose(0, 1).contiguous()

    return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

These are the only major changes required to add sequence-parallelism on top of tensor-parallelism. Note: Sequence-parallelism 
uses the same tensor-parallel group. 
For reference implementation, follow `GPTNeoX-20B model script <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/tp_dp_gpt_neox_hf_pretrain/tp_dp_gpt_neox_20b_hf_pretrain/modeling_gpt_neox_nxd.py>`__.

Activation Recomputation
^^^^^^^^^^^^^^^^^^^^^^^^

As seen in the `App notes on Activation Memory Recomputation` we can reduce the activation memory by recomputing few operations from 
the forward pass during the backward run. To replay some of the compute, we can use the 
`torch.utils.checkpoint.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`__. To use this API, we need 
to put the compute, we want to replay, inside a function which can be passed to the `checkpoint` API. This API takes care 
of maintaining the RNG seed, not saving the activations and also inserting the forward recompute during the gradient computation.

To enable selective activation checkpointing for the attention block, we can simply pass the attention block to the checkpoint 
api as follows:

.. code:: ipython3

    if config.selective_activation_checkpointing_is_enabled:
        attn_output = torch.utils.checkpoint.checkpoint(self._attn, query, key, value, attention_mask, head_mask)
    else:
        attn_output = self._attn(query, key, value, attention_mask, head_mask)

Note: To use torch.utils.checkpoint, it is mandatory to use `-O1 <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html?highlight=--O1#cmdoption-neuronx-cc-arg-0>`__ 
compiler flag. If this is not enabled, the Neuron compiler would eliminate the duplicate recompute as an 
optimization and hence you would not see any memory gains.