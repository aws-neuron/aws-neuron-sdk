.. post:: October 16, 2026
    :language: en
    :tags: announce-eos-nxdi-nxd-core-inference

.. _announce-eos-nxdi-nxd-core-inference:

Announcing end of support for the NxD Inference library and NxD Core Inference APIs on Trn2 and Trn3 starting in Neuron 2.34
----------------------------------------------------------------------------------------------------------------------------

As announced in :ref:`Neuron 2.30 <announce-maintenance-nxdi-nxd-core-inference>` and reaffirmed in :ref:`Neuron 2.31 <announce-nxdi-maintenance-support-2-31>`, the NxD Inference library and NxD Core Inference APIs have been in maintenance mode, receiving critical bug fixes and security patches only.

Starting with Neuron Release 2.34, Neuron will end support for the NxD Inference library and the NxD Core Inference APIs on Trn2 and Trn3 instances. No further releases, bug fixes, or security patches will be provided after this release to support Trn2 and Trn3 instances.

Current users on Trn2 and Trn3 should migrate to the `vLLM Neuron plugin <https://github.com/vllm-project/vllm-neuron>`_, which does not depend on the NxD Inference libraries and is the recommended inference solution for text generation and vision understanding LLMs on Trn2 and Trn3. For migration guidance, see `Migrating from NxD Inference to vLLM Neuron <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/getting-started/migration-nxdi-to-vllm-neuron.html>`_.

On Inf2 and Trn1 instances, the NxD Inference library and NxD Core Inference APIs remain in maintenance mode at Neuron SDK 2.31, as described in :ref:`Neuron 2.31 <announce-nxdi-maintenance-support-2-31>`. These instances continue to receive critical bug fixes and security patches only.

NxD Core model build and trace APIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The NxD Core model build and trace APIs (for example ``neuronx_distributed.trace.parallel_model_trace`` and the :ref:`ModelBuilder / ModelBuilderV2 <nxd-core-model-builder-v2>` APIs) are superseded by native PyTorch support on Trainium through :ref:`TorchNeuron <native-pytorch-trainium>`, which provides ``torch.compile``-based compilation for Neuron.

Current users of these APIs should migrate to the native PyTorch ``torch.compile(backend="neuron")`` workflow. For details and migration guidance, see :ref:`Native PyTorch for AWS Trainium <native-pytorch-trainium>`.
