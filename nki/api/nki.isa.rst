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
   nc_matmul_mx
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
   tensor_scalar_cumulative
   tensor_copy
   tensor_copy_dynamic_src
   tensor_copy_dynamic_dst
   tensor_copy_predicated
   reciprocal
   quantize_mx
   iota
   dropout
   affine_select
   range_select
   select_reduce
   sequence_bounds
   memset
   bn_stats
   bn_aggr
   local_gather
   dma_copy
   dma_transpose
   dma_compute
   max8
   nc_n_gather
   nc_find_index8
   nc_match_replace8
   nc_stream_shuffle
   register_alloc
   register_load
   register_move
   register_store
   core_barrier
   sendrecv
   rng
   rand2
   rand_set_state
   rand_get_state
   set_rng_seed



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

