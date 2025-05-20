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
   range_select
   memset
   bn_stats
   bn_aggr
   local_gather
   dma_copy
   max8
   nc_find_index8
   nc_match_replace8
   nc_stream_shuffle


NKI ISA Config Enums
--------------------
.. autosummary::
   :toctree: generated
   :template: nki-custom-class-attr-only-template.rst
   :nosignatures:

   engine
   reduce_cmd
   dge_mode


Target
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   nc_version
   get_nc_version

