nki.collectives
===============

.. currentmodule:: nki.collectives


The ``nki.collectives`` module provides APIs for multi-core collective communication
operations such as all-reduce and all-gather across NeuronCores.

.. _nki-collectives:

NKI Collectives
---------------

Collective operations for multi-rank communication.

.. autosummary::
   :toctree: generated
   :nosignatures:

   all_reduce
   all_gather
   reduce_scatter
   all_to_all
   all_to_all_v
   collective_permute
   collective_permute_implicit
   collective_permute_implicit_reduce
   collective_permute_implicit_current_processing_rank_id
   rank_id


Constants
--------------

.. autosummary::
   :toctree: generated
   :template: nki-custom-class-template.rst
   :nosignatures:

   ReplicaGroup
