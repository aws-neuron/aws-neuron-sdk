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

#. Partition dimension broadcasting is not supported on operator overloads (i.e, ``+``, ``-``, ``*``, ``/``, ``<<``, ``>>``, etc),
   use ``nki.language`` APIs instead (i.e, ``nl.add``, ``nl.multiply``, ...).

#. When direct allocation API is used, non-IO HBM tensors are not supported.

   * All tensors declared with ``buffer=nl.shared_hbm`` must be returned as the result of the kernel.

   * Tensors declared with ``buffer=nl.hbm`` or ``buffer=nl.private_hbm`` are not allowed.

   * An error "``[NKI005] (float32 [128, 512] %'<name of the hbm tensor>':5)0: DRAM location of kind
     Internal mapping failed. Only input/output/const DRAM location is supported!``" will be thrown when such
     tensor is encountered.

#. ``+=`` only works reliably in the special context of PSUM accumulation for matmuls within an affine loop:

   .. code-block:: python

      # condition 1: a psum buffer with zeros
      psum_buf = nl.zeros(..., buffer=nl.psum)

      # condition 2: an affine range loop
      for i in nl.affine_range(N):
         # condition 3: add matmul results from TensorEngine
         psum_buf += nl.matmul(...) # or nisa.nc_matmul

   Avoid using ``+=`` unless all three conditions above are met in your kernel code. Use ``var[...] = var + new_var``
   explicitly as a workaround.


Unexpected Behavior:
--------------------------

#. Simulation using :doc:`nki.simulate_kernel <api/generated/nki.simulate_kernel>`:

   *  Custom data types like ``nl.float32r``, ``nl.bfloat16``, ``nl.float8_e4m3``, and ``nl.float8_e5m2`` simulate
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
