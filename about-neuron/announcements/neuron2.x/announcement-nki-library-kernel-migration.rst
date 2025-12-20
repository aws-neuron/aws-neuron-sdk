.. post:: December 16, 2025
    :language: en
    :tags: announcement-nki-library-kernel-migration

.. _announcement-nki-library-kernel-migration:

Announcing NKI Library Kernel Migration to New nki.* Namespace in Neuron 2.28
------------------------------------------------------------------------------

Some NKI Library kernels currently use the legacy ``neuronxcc.nki.*`` namespace. Starting with Neuron 2.28, all NKI Library kernels will migrate to the new ``nki.*`` namespace.

The new ``nki.*`` namespace introduces changes to NKI APIs and language constructs that improve usability and performance. This transition ensures consistency across all NKI kernels and allows us to focus development efforts on a single, modern namespace.

See :doc:`the NKI Kernel Migration Guide </nki/deep-dives/nki-migration-guide>` for more information.
