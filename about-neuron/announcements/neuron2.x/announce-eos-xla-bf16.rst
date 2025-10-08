.. post:: May 15, 2025
    :language: en
    :tags: announce-eos-xla-bf

.. _announce-eos-xla-bf:

Announcing end of support for XLA_USE_BF16 and XLA_DOWNCAST_BF16 starting next release
----------------------------------------------------------------------------------------

Starting with :ref:`Neuron Release 2.23 <neuron-2.23.0-whatsnew>`, Neuron will begin phasing out support for the ``XLA_DOWNCAST_BF16`` and ``XLA_USE_BF16`` environment variables. In this release, usage of these variables will trigger warnings. Neuron will end support in a subsequent release, aligned with the torch-xla maintenance schedule.

Customers are recommended to migrate to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to convert their model to BF16 format. For detailed migration guidance, please refer to :ref:`migration_from_xla_downcast_bf16`.
