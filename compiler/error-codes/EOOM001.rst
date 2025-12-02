.. _error-code-eoom001:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EOOM001.

NCC_EOOM001
===========

**Error message**: The combined memory needed for the model tensors exceeds the high-bandwidth memory limit.

The memory usage consists of:

- I/O tensors: Input and output activation tensors
- Internal allocations: Scratchpad memory for intermediate computations
- SBUF spills: Data that cannot fit in on-chip SBUF memory and must spill to HBM

There are several ways to potentially fix this issue.

1. Simply reduce the batch/tensor size if possible
2. Utilize pipeline/tensor parallelism via neuronx-distributed

Short snippet of tensor parallelism:

.. code-block:: python

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

For more information: 

- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/activation_memory_reduction.html
- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/app_notes/nxd-training-pp-appnote.html
