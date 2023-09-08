.. _nemo-megatron-index:

AWS Neuron Reference for NeMo Megatron
======================================

AWS Neuron Reference for NeMo Megatron is a library that includes modified versions of the open-source packages `NeMo <https://github.com/NVIDIA/NeMo>`_ and `Apex <https://github.com/NVIDIA/apex>`_ that have been adapted for use with AWS Neuron and AWS EC2 Trn1 instances.
The library supports Tensor Parallel, Pipeline parallel and Data Parallel configurations for distributed training of large language models like GPT-3 175B. The APIs have been optimized for XLA based computation and high performance communication over Trainium instances.
The library uses various techniques to improve memory utilization such as sequence parallelism which reduces activation memory footprint, selective or full activation checkpointing which allows larger model configurations to fit. SPMD optimizations are also used whenever possible to reduce the number of graphs obtained.



.. dropdown::  Setup  (``neuronx-nemo-megatron``)
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    The library can be installed from `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_


.. dropdown::  Tutorials  (``neuronx-nemo-megatron``)
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in
   
    * `Launch a GPT-3 pretraining job using neuronx-nemo-megatron <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-gpt-job.md>`_
    * `Launch a Llama 2 pretraining job using neuronx-nemo-megatron <https://github.com/aws-neuron/aws-neuron-parallelcluster-samples/blob/master/examples/jobs/neuronx-nemo-megatron-llamav2-job.md>`_