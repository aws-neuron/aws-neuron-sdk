.. _nki_known_issues:

NKI Known Issues 
==========================

This document outlines some of the known issues and limitations for the NKI beta release.

Unsupported Syntax:
--------------------

#. Top-level tensors must be on HBM. The input and output tensors of the top-level NKI kernel 
   (the kernel function decorated with ``nki_jit``/``nki.baremetal`` or called by JAX ``nki_call``)
   must be located in HBM. We currently do not support using tensors stored in SBUF or PSUM 
   as the input or output of the top-level kernel. Tensors must be loaded from HBM into SBUF 
   before use, and output tensors must be stored from SBUF back into HBM. 
   See :doc:`nl.load <api/generated/nki.language.load>` and :doc:`nl.store <api/generated/nki.language.store>`.

#. Top-level input and output tensors have to be distinct. We do not support reading and writing to the same tensor.
   See corresponding :ref:`error message <nki-errors-err_read_modify_write_on_kernel_parameter>` for more info.

#. Indexing:

   * Tile on SBUF/PSUM must have at least 2 dimensions as described :ref:`here <nki-fig-pm-memory>`. If using a 1D tile on SBUF/PSUM, 
     users may get an "``Insufficient rank``" error. Workaround this by creating a 2D tile, e.g.,

     .. code-block:: python

      buf = nl.zeros((128, ), dtype=dtype, buffer=nl.sbuf)  # this won't work
      buf = nl.zeros((128, 1), dtype=dtype, buffer=nl.sbuf) # this works

   * Users must index their ``[N, 1]`` or ``[1, M]`` shaped 2D buffers with both indices, 
     do ``my_sbuf[0:N, 0]`` or ``my_sbuf[0, 0:M]`` to access them, since accessing in 1D ``my_sbuf[0:N]`` won't work.

   * Use ``nl.arange`` for indirect load/store access indexing, ``nl.mgrid`` won't work. See code examples 
     in :doc:`nl.load <api/generated/nki.language.load>` and :doc:`nl.store <api/generated/nki.language.store>`.

   * If indexing with ``[0, 0]`` gets internal errors, try using ``[0:1, 0:1]`` or ``nl.mgrid[0:1, 0:1]`` instead.

   * If indexing with ``[0:1, ...]`` gets internal errors, try using ``[0, ...]`` instead.

#. Masks conjunction: Use ``&`` to combine masks. We do not support using ``and`` for masks. 
   See examples in :ref:`NKI API Masking <nki-mask>`.

#. :doc:`nisa.bn_stats <api/generated/nki.isa.bn_stats>` does not support mask on the reduce dimension, 
   the mask sent to ``bn_stats`` could not contain any indices from the reduction dimension.

#. Partition dimension broadcasting is not supported on operator overloads (i.e, ``+``, ``-``, ``*``, ``/``),
   use ``nki.language`` APIs instead (i.e, ``nl.add``, ``nl.multiply``, ...).


Unexpected Behavior:
--------------------------

#. Simulation using :doc:`nki.simulate_kernel <api/generated/nki.simulate_kernel>`:

   *  Custom data types like ``nl.float32r``, ``nl.bfloat16``, and ``nl.float8_e4m3`` simulate 
      in ``fp32`` precision. Also, NumPy API calls outside of the NKI kernel, such as ``np.allclose`` 
      may not work with the above types.

   *  :doc:`nl.rand <api/generated/nki.language.rand>` generates the same values for subsequent calls to ``nl.rand()``.

   *  :doc:`nl.random_seed <api/generated/nki.language.random_seed>` is a no-op in simulation.

   *  :doc:`nisa.dropout <api/generated/nki.isa.dropout>` is a no-op in simulation.

   *  Masks don't work in simulation, and garbage data is generated in tensor elements that are 
      supposed to be untouched based on API masking.

#. Execution: 

   * Unwritten output tensor will have garbage data. See detail :ref:`here <nki-output-garbage-data>`.

   * :doc:`nl.invert <api/generated/nki.language.invert>` (aka ``bitwise_not``) produces incorrect result 
     with ``bool`` input type, use ``int8`` type instead.

#. Profiler:
  
   * When using ``neuron-profile`` use the flag ``--disable-dge`` to workaround a temporary issue with DMA information.
     See the :ref:`Profile using neuron-profile <nki-neuron-profile-capture-cmdline>` section 
     for more details.

#. Optimization:
  
   * Users need to declare their NKI buffers as small as possible to avoid buffer overflow errors.
     An error "``[GCA046] Some infinite-cost nodes remain``" may mean there's a
     buffer overflow, workaround this by creating smaller local buffers.

#. Compiler passes:
  
   *  NKI ISA API may not be one-to-one with generated hardware ISA instructions. The compiler 
      may aid in the support of these instruction calls by adding additional instructions.
  
   *  NKI ISA :doc:`nisa.nc_transpose <api/generated/nki.isa.nc_transpose>` API's ``engine`` 
      param may not be respected in some corner cases, such as if the transpose is merged 
      with load/store into intermediate operations during compilation.