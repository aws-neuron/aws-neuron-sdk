.. post:: December 16, 2025
    :language: en
    :tags: announcement-end-of-support-vllm-v0

.. _announcement-end-of-support-vllm-v0:

Announcing End of Support for vLLM V0 starting with Neuron 2.28
----------------------------------------------------------------

Neuron Release 2.27 will be the last release to support vLLM V0. In Neuron 2.27 release, vLLM V1 support is introduced for Neuron using the ``vllm-neuron`` plugin. Review the sources in the `Neuron vLLM GitHub Repository <https://github.com/vllm-project/vllm-neuron>`__.

Starting with the Neuron 2.28 release, vLLM V0 will not be supported. Support will be dropped for vLLM V0 Neuron forks of the `upstreaming-to-vllm <https://github.com/aws-neuron/upstreaming-to-vllm/>`__ Neuron GitHub repo, along with vLLM V0-based Neuron Inference Deep Learning Containers.

Customers should migrate to vLLM V1 using the :doc:`vLLM V1 user guide </libraries/nxd-inference/developer_guides/vllm-user-guide-v1>`. Customers are recommended to start using vLLM V1 based inference containers that are released with Neuron v2.27.0. We plan to update the existing vLLM-based tutorials to use vLLM V1 in the coming release.

See :doc:`vLLM on Neuron </libraries/nxd-inference/vllm/index>` for more information on vLLM V1.
