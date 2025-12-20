.. post:: December 19, 2025
    :language: en
    :tags: announce-nxdi-changes

.. _announce-nxdi-changes:

Announcing changes to NxDI in the upcoming releases
====================================================

As part of our transition to native PyTorch support, we are simplifying NxDI to provide a more streamlined developer experience.

**What's changing:**

In the upcoming releases, we will introduce NxDI v2 that will not use NxDI ModelBuilder APIs. Instead, it will use ``torch.compile`` for model compilation. We will also simplify the NxDI APIs for modeling to align with native PyTorch primitives.

**Timeline and migration:**

While we introduce these changes, we will maintain both NxDI v1 and NxDI v2 simultaneously to ensure a smooth migration path for our customers. We will provide detailed migration guidance, timelines, and updated documentation as we approach the transition. More information about the migration path and specific release dates will be shared in the next release (Neuron 2.28).


