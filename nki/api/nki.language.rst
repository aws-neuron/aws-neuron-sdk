.. _nki-language:

nki.language
====================

.. currentmodule:: nki.language

The ``nki.language`` module provides high-level constructs for writing NKI kernels.
It includes tensor creation, indexing, type casting, math operations, and loop constructs
that the NKI compiler translates into efficient hardware instructions.

.. _nl_creation:

Creation operations
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ndarray
   zeros
   ones
   full
   zeros_like
   empty_like
   shared_identity_matrix
   rand
   random_seed

.. _nl_tensor_ops:

Tensor operations
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   load
   load_transpose2d
   store
   copy
   matmul
   transpose

.. _nl_math:

Math operations
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   abs
   abs_max
   abs_min
   add
   arctan
   ceil
   cos

   .. divide : not supported

   exp
   floor
   log
   maximum
   minimum
   multiply
   negative
   power
   reciprocal
   rsqrt
   sign
   sin
   sqrt
   square
   subtract
   tan
   tanh
   trunc

.. _nl_activation_and_backpropagation:

Activation and Backpropagation functions
-----------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   relu
   sigmoid
   silu
   silu_dx
   gelu
   gelu_dx
   gelu_apprx_sigmoid
   gelu_apprx_sigmoid_dx
   gelu_apprx_tanh
   mish
   softplus
   softmax
   erf
   erf_dx


.. _nl_normalization_and_regularization:

Normalization and Regularization functions
------------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dropout
   rms_norm


.. _nl_reduction:

Reduction operations
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   all
   max
   mean
   min
   prod
   sum
   var

.. _nl_comparison:

Comparison operations
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   equal
   not_equal
   less
   less_equal
   greater
   greater_equal

.. _nl_logical:

Logical operations
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   logical_and
   logical_or
   logical_xor
   logical_not

.. _nl_bitwise:

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

.. _nl_tensor_manipulation_operations:

Tensor manipulation operations
-------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   broadcast_to
   ds
   expand_dims


.. _nl_indexing:

Indexing operations
------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   where
   gather_flattened

.. _nl_iterators:

Iterators
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   affine_range
   dynamic_range
   sequential_range
   static_range


.. _nl_memory_hierarchy:

Memory Hierarchy
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   psum
   sbuf
   hbm
   private_hbm
   shared_hbm
   is_psum
   is_sbuf
   is_hbm
   is_on_chip

.. _nl_others:

Others
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   device_print
   no_reorder
   program_id
   num_programs
   program_ndim

.. _nl_datatypes:

Data Types
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   bool_
   int8
   int16
   int32
   uint8
   uint16
   uint32
   float16
   float32
   bfloat16
   tfloat32
   float8_e4m3
   float8_e5m2
   float8_e4m3fn
   float8_e5m2_x4
   float8_e4m3fn_x4
   float4_e2m1fn_x4


.. _nl_constants:

Constants
----------

.. list-table::

   * - :doc:`tile_size <nki.language.tile_size>`
     - Hardware tile size constants (pmax, psum_fmax, gemm_stationary_fmax, etc.)

.. toctree::
   :hidden:

   nki.language.tile_size


.. _nl_operator_constants:

Operator Constants
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   prelu
   bypass
