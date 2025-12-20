.. post:: December 16, 2025
    :language: en
    :tags: announcement-nki-library-namespace-changes

.. _announcement-nki-library-namespace-changes:

Announcing NKI Library Namespace Changes in Neuron 2.28
--------------------------------------------------------

NKI Library kernels are published in the `NKI Library GitHub repository <https://github.com/aws-neuron/nki-library>`__. In Neuron 2.27, these kernels are also shipped as part of neuronx-cc using the nkilib.* namespace. To avoid namespace conflicts when customers use kernels from the open source repository, the repository uses the ``nkilib_standalone.nkilib.*`` namespace.

Starting with Neuron 2.28 the open source repository namespace will change from ``nkilib_standalone.nkilib.*`` to ``nkilib.*``, providing a consistent namespace between the open source repository and the shipped version.

See :doc:`NKI Library </nki/library/index>` for more information.
