.. _nki-language:

nki.language
====================

.. currentmodule:: nki.language

.. _nl_creation:

Creation operations
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ndarray
   zeros

Tensor manipulation operations
-------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ds

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
   float8_e5m2_x4
   float8_e4m3fn_x4
   float4_e2m1fn_x4


Constants
-----------

.. autosummary::
   :toctree: generated
   :template: nki-custom-class-attr-only-template.rst
   :nosignatures:

   tile_size
