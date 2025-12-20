.. post:: December 16, 2025
    :language: en
    :tags: announcement-end-of-support-neuronxcc-nki

.. _announcement-end-of-support-neuronxcc-nki:

Announcing End of Support for neuronxcc.nki Namespace Starting with Neuron 2.28
--------------------------------------------------------------------------------

Neuron 2.27 will be the last to include support for the neuronxcc.nki.* namespace. Starting with Neuron 2.28, this namespace will no longer be supported.

The new ``nki.*`` namespace introduces changes to NKI APIs and language constructs. 

Existing kernels using ``neuronxcc.nki.*`` must migrate to the new nki.* namespace. A kernel migration guide is available in the Neuron 2.27 documentation.

See :doc:`the NKI Kernel Migration Guide </nki/deep-dives/nki-migration-guide>` for more information.
