.. _context_parallelism_overview:

Context Parallelism Overview 
===============================

Context parallelism (CP) is a technique used in deep learning model training to train large context models.
CP parallelizes the processing of neural network activations across multiple devices by partitioning the input 
tensors along the sequence dimension. CP reduces the memory footprint and computational cost of processing long sequences.
Unlike Sequence Parallelism (SP) that partitions the activations of specific layers, CP divides the activations of all layers.

The implementation of Context Parallelism in NxD leverages `Ring Attention <https://arxiv.org/abs/2310.01889>`_. Ring Attention
enables efficient communication between devices by organizing them in a ring topology, allowing tokens to attend to each other 
across devices without needing full attention computation on each device. This reduces memory overhead while extending the 
feasible context length beyond traditional transformer models.

For more details, refer to Context Parallelism in Megatron <https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html>_

.. image:: /libraries/neuronx-distributed/images/cp.png
   :alt: Image: image.png

Fig: Context Parallelism in NxD (Figure adapted from `Megatron 
CP <https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html>`_).
In NxD's TP implementation, we make use of All-Gather (AG), Reduce-Scatter (RS) collectives. Further
CP is applied to all layers including LayerNorm (LN), Linear (LIN) and Fully-Connected (FC) layers.
The figure shows a transformer layer running with TP2 and CP2. Assuming sequence length is 8K, each device processes 4K tokens. 
Device0 and Device2 form a CP group and exchange KV with each other; similarly, Device1 and Device3 form a CP group and exchange KV with each other. 
The collective communication to exchange KV is handled by NxD using approaches described in the 
`Ring Attention <https://arxiv.org/abs/2310.01889>`_ paper.
   

