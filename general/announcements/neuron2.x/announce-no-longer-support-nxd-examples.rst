.. post:: May 15, 2025
    :language: en
    :tags: announce-eol-nxd-examples

.. _announce-eol-nxd-examples:

Announcing migration of NxD Core inference examples from NxD Core repository to NxD Inference repository starting this release
--------------------------------------------------------------------------------------------------------------------------------

Starting with :ref:`Neuron Release 2.23 <neuron-2.23.0-whatsnew>`, the following models and modules in NxD Core inference examples are now only available through NxD Inference package:

- Llama
- Mixtral
- DBRX

=========================================================================================================================
I currently utilize one of the mentioned inference samples from the NxD Core repository in my model code. What do I do?
=========================================================================================================================

For customers who want to deploy models out of the box, please use the NxD Inference model hub, which is the recommended option. With NxD Inference, you can import and use these models and modules in your applications. 
Customers will need to update their applications to use examples under the NxD Inference repository: https://github.com/aws-neuron/neuronx-distributed-inference.
Any models compiled with inference code from the NxD Core repository will need to be re-compiled. Please refer to the :ref:`nxd-examples-migration-guide` for guidance and see :ref:`nxdi-overview` for more information.

========================================================
I would like to continue using NxD Core. What do I do?
=======================================================

For customers who want to continue using NxD Core without NxD Inference, please refer to the Llama3.2 1B sample as a reference implementation: https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/llama
