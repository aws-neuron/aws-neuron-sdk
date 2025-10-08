.. post:: June 24, 2025
    :language: en
    :tags: announce-no-longer-support-xla-env-vars

.. _announce-eos-longer-support-xla-bf16-vars:

Announcing end of support XLA_USE_BF16 and XLA_DOWNCAST_BF16 environment variables starting next release
---------------------------------------------------------------------------------------------------------

:ref:`Neuron Release 2.24 <neuron-2-24-0-whatsnew>` will be the last release to support the following environment variables:

- XLA_USE_BF16
- XLA_DOWNCAST_BF16

**I currently utilize these environment variables in my model code. What do I do?**

Customers are recommended to migrate to automatic mixed-precision or use ``model.to(torch.bfloat16)`` command to convert their model to BF16 format. For detailed migration guidance, please refer to :ref:`migration_from_xla_downcast_bf16`.
