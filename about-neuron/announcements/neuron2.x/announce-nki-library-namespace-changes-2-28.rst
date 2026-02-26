.. post:: February 26, 2026
    :language: en
    :tags: announce-nki-library-changes

.. _announce-nki-library-namespace-changes-2-28:

NKI Library namespace changes starting with Neuron 2.28
--------------------------------------------------------

Starting with Neuron 2.28, the open source repository namespace has changed from ``nkilib_standalone.nkilib.*`` to ``nkilib.*``, providing a consistent namespace between the open source repository and the shipped version. If customers want to add or modify NKI Library kernels, they can build and install them to replace the default implementation without changing model imports.

See :ref:`NKI Library <nkl_home>` for more information.
