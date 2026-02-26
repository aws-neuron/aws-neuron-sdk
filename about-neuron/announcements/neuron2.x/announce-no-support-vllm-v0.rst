.. post:: February 26, 2026
    :language: en
    :tags: announce-no-support-vllm

.. _announce-no-support-vllm-v0:

Neuron no longer supports vLLM V0 starting with Neuron 2.28
------------------------------------------------------------

Starting with Neuron 2.28 release, vLLM V0 will no longer be supported. This includes the vLLM V0 Neuron forks in the AWS Neuron `upstreaming-to-vllm GitHub repo <https://github.com/aws-neuron/upstreaming-to-vllm>`__ and vLLM V0-based Neuron Inference Deep Learning Containers.

Customers are recommended to use vLLM V1-based inference containers as documented in the :doc:`vLLM V1 user guide </libraries/nxd-inference/developer_guides/vllm-user-guide-v1>`. Additionally, Neuron will be updating existing vLLM-based tutorials to use vLLM V1 in the coming release.

See :ref:`vLLM on Neuron <nxdi-vllm-user-guide-v1>` for more information on vLLM V1 support.
