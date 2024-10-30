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


Creation operations
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ndarray
   zeros
   zeros_like
   ones
   full
   rand
   random_seed
   shared_constant


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
   exp
   log
   cos
   sin
   tanh
   arctan
   sqrt
   rsqrt
   sigmoid
   relu
   gelu
   gelu_dx
   gelu_apprx_tanh
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



Bitwise operations
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   bitwise_and
   bitwise_or
   bitwise_xor
   invert


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

   arange
   mgrid
   expand_dims


Sorting/Searching operations
-----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   where


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
   device_print
   loop_reduce


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
   :nosignatures:

   tile_size
