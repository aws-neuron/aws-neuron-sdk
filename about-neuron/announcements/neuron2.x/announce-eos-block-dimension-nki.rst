.. post:: June 24, 2025
    :language: en
    :tags: announce-eos-block-dimension-nki

.. _announce-eos-block-dimension-nki:

Announcing end of support for NKI block dimension starting next release
--------------------------------------------------------------------------

:ref:`Neuron release 2.24 <neuron-2-24-0-whatsnew>` will be the last release to include support for the NKI block dimension in NKI tensor creation routines. Starting with this release, using the block dimension will generate EOS warnings. In the next release (Neuron Release 2.25), these warnings will be upgraded to errors.

Customers are recommended to refer to the :ref:`nki_block_dimension_migration_guide` for detailed instructions on updating their code.
