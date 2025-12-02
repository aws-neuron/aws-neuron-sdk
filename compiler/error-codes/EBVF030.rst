.. _error-code-ebvf030:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EBVF030.

NCC_EBVF030
===========

**Error message**: The number of instructions generated exceeds the limit.

Consider applying model parallelism as partitioning the model will help break large computational graphs into smaller subgraphs.

For more information: 

- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/neuronx-distributed/api_guide.html#api-guide
- https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/app_notes/nxd-training-pp-appnote.html
