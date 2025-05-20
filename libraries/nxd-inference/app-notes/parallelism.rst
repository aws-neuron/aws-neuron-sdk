.. _nxdi-parallelism-user-guide:

Parallelism Techniques for LLM Inference
========================================


.. contents:: Table of contents
   :local:
   :depth: 2

Overview
--------



Large language models (LLMs) have grown exponentially in size in the past few years, requiring
increasing accelerator memory to run the model. In order to effectively generate predictions from an LLM, it
is often necessary to use one or more **parallelism techniques** to shard operations across multiple available accelerators.
**Model parallelism**, such as tensor and sequence parallelism described in this document, can reduce memory requirements per NeuronCore 
by sharding the model across multiple cores. **Data parallelism**, on the other hand, enables
higher throughput by sharding input data.

Tensor Parallelism
---------------

Tensor parallelism is a technique in which a tensor is split into a number of chunks along the intermediate
dimension, resulting in sharding not only model parameters but also intermediate activations.
Tensor parallelism has relatively high communication volume and presents a synchronization point in forward pass,
making it costly to scale beyond 1 node. When tensors are sharded across multiple EC2 instances, the collective communication
at these synchronization points must occur through network interfaces like Elastic Fabric Adapter (EFA) instead of
the faster chip-to-chip NeuronLink connections.

A basic transformer MLP block contains a single matrix multiplication (matmul) called the up-projection, 
which increases the dimensionality from the hidden_size to the intermediate_size, and a single output matmul called the down projection, 
which reduces the dimensionality back to the hidden_size, with a non-linear activation function in-between. 
In order to avoid running collective operations (synchronization point) after each matrix multiply, we
defer collective to run after 2nd linear layer. To ensure correctness of non-linear activation
function computation (``f(x+y) != f(x) + f(y)`` for non-linear ``f`` like silu in SwiGLU), we split first linear layer
along columns (ColumnParallel) and second linear layer along rows (RowParallel), then run an AllReduce collective
operation at the end.

Modern transformer architectures use SwiGLU activation function, where the MLP block has 3 matrices, first
up and gate projection and later a down projection. We can view up and gate projection as the same
(referred to as first matrix multiply or first linear layer) in this context because they have the same
sharding approach. In this case up and gate projection is column parallel, while down projection is row parallel.

In attention, we similarly split Q, K and V projections in column parallel fashion and use row parallel for
final output (O) projection, then run AllReduce with input tensor size equal to
``batch_size x sequence_length x hidden_size x per_element_bytes`` bytes. Here,``per_element_bytes`` depends on the
numerical format of your tensors. When using BF16, for example, it would be ``2``. 
AllReduce input tensor size is the same for both MLP and attention blocks, resulting in two AllReduce operations
with with the same input size and output size as per AllReduce algorithm per transformer layer.

.. figure:: /images/sharding/tensor_parallel.png
   :alt: Image: sharding_tensor_parallel.png

   Image visualizing transformer layer like llama3 with SwiGLU activation layer in MLP.

How to Use Tensor Parallelism with NxD Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tensor parallelism can be enabled by setting the ``tp_degree`` parameter in NeuronConfig. See
:ref:`nxdi-feature-guide-tensor-parallelism` for more detail.

Code example, defining NeuronConfig:
::
    neuron_config = NeuronConfig(tp_degree=32)

See :ref:`tensor_parallelism_overview` for a detailed reference of the concepts underlying tensor parallelism.

Sequence Parallelism
---------------

One drawback of tensor parallelism is that it replicates attention/MLP layer norm and dropout operations across all NeuronCores.
These operations are less compute intensive compared to linear layers, but still requires
significant memory. These computations are independent along the sequence dimension, allowing us to shard
across the sequence dimension. This requires AllGather in the transition from a sequence to a tensor parallel 
region and ReduceScatter in the transition from tensor to sequence parallel region during inference.
Sequence parallelism is especially useful for longer sequences and usually used in conjunction with tensor parallelism.


.. figure:: /images/sharding/sequence_parallel.png
   :alt: Image: sharding_sequence_parallel.png

   Image visualizing how sequence and tensor parallelism intertwine in transformer layer like Llama 3.

How to Use Sequence Parallelism with NxD Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sequence parallelism can be enabled by setting the ``sequence_parallel_enabled`` parameter in NeuronConfig. See 
:ref:`nxdi-feature-guide-sequence-parallelism` for more detail.

Code example, defining NeuronConfig:
::
    neuron_config = NeuronConfig(sequence_parallel_enabled=True)

Flash Decoding
--------------

Flash decoding enables inference on long sequences by partitioning the KV cache. The technique is useful for 
long sequences and used in decoding phase. It is motivated by the fact that assuming KV caching, K and V memory
footprint scales with sequence length, while Q has sequence length equal to 1 during decoding.

Flash decoding shards K and V, and at the start uses AllGather to gather all Q heads in each
KV partition. Each KV partition computes partial softmax (also called log-sum-exp) which uses AllGather
to compute log-sum-exp scaling factor and correction denominator after “local” attention computation
(multiply Q and K, then apply the mask). Lastly, the algorithm performs ReduceScatter on attention results at the end.

How to Use Flash Decoding with NxD Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Flash decoding can be enabled by setting the ``flash_decoding_enabled`` parameter in NeuronConfig.
The technique is only supported with GQA (grouped query attention).

Code example, defining NeuronConfig:
::
    neuron_config = NeuronConfig(flash_decoding_enabled=True)


Data Parallelism
--------------

Data parallelism will replicate the model (same architecture, weights and hyperparameters) but will shard input data.
By distributing the data across NeuronCores or even different instances, data parallelism reduces
the total execution time of large batch size inputs using parallelization across sharded inputs instead of
sequential execution. Compared to batch parallel where KV cache is sharded, each data parallel replica has
its own individual cache for separate sequences.

Data parallelism works as standalone technique or can be used in conjunction with other model sharding techniques such as tensor parallelism. 
For example, Trn2 instances has 64 NeuronCores available when using default Logical NeuronCore configuration (LNC) set to 2, so you can use a
tensor parallel degree of 16 and a data parallel degree of 4, resulting in four copies of the model, each with disjunct data partitioning and
with each model sharded across 16 logical NeuronCores. Model replicas can run on the same instance or separate instances.
Data parallelism doesn't introduce any additional collective operations during inference.

.. figure:: /images/sharding/data_parallel.png
   :alt: Image: sharding_data_parallel.png

   Image visualizing how data parallelism shards inputs.

How to Use Data Parallelism with NxD Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`nxdi-trn2-llama3.3-70b-dp-tutorial` for detailed guidance on how to use vLLM to apply data parallelism along with tensor
parallelism to increase model inference throughput in NxDI. 