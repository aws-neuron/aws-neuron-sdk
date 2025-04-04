nki.isa
========

.. currentmodule:: nki.isa


.. _nki-isa:

NKI ISA
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   nc_matmul
   nc_transpose
   activation
   activation_reduce
   tensor_reduce
   tensor_partition_reduce
   tensor_tensor
   tensor_tensor_scan
   scalar_tensor_tensor
   tensor_scalar
   tensor_scalar_reduce
   tensor_copy
   tensor_copy_dynamic_src
   tensor_copy_dynamic_dst
   tensor_copy_predicated
   reciprocal
   iota
   dropout
   affine_select
   memset
   bn_stats
   bn_aggr
   local_gather
   dma_copy
   max8
   nc_find_index8
   nc_match_replace8
   nc_stream_shuffle


Accumulation Command
--------------------
.. autosummary::
   :toctree: generated
   :template: nki-custom-class-attr-only-template.rst
   :nosignatures:

   reduce_cmd


Engine Types
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   tensor_engine
   vector_engine
   scalar_engine
   gpsimd_engine
   dma_engine
   unknown_engine


Target
-------------

.. autosummary::
   :toctree: generated
   :template: nki-custom-class-attr-only-template.rst
   :nosignatures:

   engine

.. autosummary::
   :toctree: generated
   :nosignatures:

   nc_version
   get_nc_version
