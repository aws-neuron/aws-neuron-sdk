.. _nki-language:

nki.language
====================

.. currentmodule:: nki.language


Memory operations
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   load
   store
   load_transpose2d
   atomic_rmw
   copy
   broadcast_to


.. _nl_creation:

Creation operations
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ndarray
   empty_like
   zeros
   zeros_like
   ones
   full
   rand
   random_seed
   shared_constant
   shared_identity_matrix


.. _nki-lang-math:

Math operations
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add
   subtract
   multiply
   divide
   power
   maximum
   minimum
   max
   min
   mean
   var
   sum
   prod
   all
   abs
   negative
   sign
   trunc
   floor
   ceil
   mod
   fmod
   exp
   log
   cos
   sin
   tan
   tanh
   arctan
   sqrt
   rsqrt
   sigmoid
   relu
   gelu
   gelu_dx
   gelu_apprx_tanh
   gelu_apprx_sigmoid
   silu
   silu_dx
   erf
   erf_dx
   softplus
   mish
   square
   softmax
   rms_norm
   dropout
   matmul
   transpose
   reciprocal


Bitwise operations
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   bitwise_and
   bitwise_or
   bitwise_xor
   invert
   left_shift
   right_shift


Logical operations
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   equal
   not_equal
   greater
   greater_equal
   less
   less_equal
   logical_and
   logical_or
   logical_xor
   logical_not


Tensor manipulation operations
-------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ds
   arange
   mgrid
   expand_dims


Indexing/Searching operations
-----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   where
   gather_flattened


Collective communication operations
------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   all_reduce
   .. all_gather
   .. reduce_scatter
   .. all_to_all


.. _nl_iterators:

Iterators
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   static_range
   affine_range
   sequential_range


Memory Hierarchy
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   par_dim
   psum
   sbuf
   hbm
   private_hbm
   shared_hbm


Others
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   program_id
   num_programs
   program_ndim
   spmd_dim
   nc
   device_print
   loop_reduce

.. _nl_datatypes:

Data Types
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   tfloat32
   .. float32
   bfloat16
   .. float16
   float8_e4m3
   float8_e5m2
   .. int32
   .. uint32
   .. int16
   .. uint16
   .. int8
   .. uint8
   .. bool_


Constants
-----------

.. autosummary::
   :toctree: generated
   :template: nki-custom-class-attr-only-template.rst
   :nosignatures:

   tile_size
   fp32
