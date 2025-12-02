.. _nki_rn:

Neuron Kernel Interface (NKI) release notes
==============================================
.. .. contents:: Table of Contents
..    :local:

..    :depth: 2


Neuron Kernel Interface (NKI) (Beta) [2.26]
------------------------------------------------
Date: 09/18/2025

* new ``nki.language`` APIs:

  * ``nki.language.gelu_apprx_sigmoid`` - Gaussian Error Linear Unit activation function with sigmoid approximation.
  * ``nki.language.tile_size.total_available_sbuf_size`` to get total available SBUF size

* new ``nki.isa`` APIs:

  * ``nki.isa.select_reduce`` - selectively copy elements with max reduction 
  * ``nki.isa.sequence_bounds`` - compute sequence bounds of segment IDs
  * ``nki.isa.dma_transpose`` 

    * ``axes`` param to define 4D transpose for some supported cases
    * ``dge_mode`` to specify Descriptor Generation Engine (DGE).

  * ``nl.gelu_apprx_sigmoid`` op support on ``nki.isa.activation``

* fixes / improvements:

  * ``nki.language.store`` supports PSUM buffer with extra additional copy inserted.

* docs/tutorial improvements:

  * ``nki.isa.dma_transpose`` API doc and example
  * ``nki.simulate_kernel`` example improvement
  * use ``nl.fp32.min`` in tutorial code instead of a magic number

* better error reporting:

  * indirect indexing on transpose
  * mask expressions


Neuron Kernel Interface (NKI) (Beta) [2.24]
------------------------------------------------
Date: 06/24/2025

* ``sqrt`` valid data range extended for accuracy improvement with wider numerical values support.
* ``nki.language.gather_flattened`` new API
* ``nki.isa.nc_match_replace8`` additional param ``dst_idx`` 
* improved docs/examples on ``nki.isa.nc_match_replace8``, ``nki.isa.nc_stream_shuffle`` 
* improved error messages

Neuron Kernel Interface (NKI) (Beta) [2.23]
------------------------------------------------
Date: 05/20/2025

* ``nki.isa.range_select`` (for trn2) new instruction
* ``abs``, ``power`` ops supported on to nki.isa tensor instruction
* ``abs`` op supported on ``nki.isa.activation`` instruction
* GpSIMD engine support added to ``add``, ``multiply`` in 32bit integer to nki.isa tensor operations
* ``nki.isa.tensor_copy_predicated`` support for reversing predicate. 
* ``nki.isa.tensor_copy_dynamic_src``, ``tensor_copy_dynamic_dst`` engine selection.
* ``nki.isa.dma_copy`` additional support with ``dge_mode``, ``oob_mode``, and in-place add ``rmw_op``.
* ``+=, -=, /=, *=`` operators now work consistently across loop types, PSUM, and SBUF,  
* fixed simulation for instructions: ``nki.language.rand``, ``random_seed``, ``nki.isa.dropout``
* fixed simulation masking behavior
* Added warning when the block dimension is used for SBUF and PSUM tensors, see: :ref:`NKI Block Dimension Migration Guide <nki_block_dimension_migration_guide>` 

Neuron Kernel Interface (NKI) (Beta) [2.22]
------------------------------------------------
Date: 04/03/2025

* New modules and APIs:

  * ``nki.profile``
  * ``nki.isa`` new APIs:
    
    * ``tensor_copy_dynamic_dst``
    * ``tensor_copy_predicated``
    * ``max8``, ``nc_find_index8``, ``nc_match_replace8``
    * ``nc_stream_shuffle``
  
  * ``nki.language`` new APIs: ``mod``, ``fmod``, ``reciprocal``, ``broadcast_to``, ``empty_like``

* Improvements:

  * ``nki.isa.nc_matmul`` now supports PE tiling feature 
  * ``nki.isa.activation`` updated to support reduce operation and ``reduce`` commands
  * ``nki.isa.engine`` enum
  * ``engine`` parameter added to more ``nki.isa`` APIs that support engine selection (ie, ``tensor_scalar``, ``tensor_tensor``, ``memset``)
  * Documentation for ``nki.kernels`` have been moved to the GitHub: https://aws-neuron.github.io/nki-samples. 
    The source code can be viewed at https://github.com/aws-neuron/nki-samples.
    
    * These kernels are still shipped as part of Neuron package in ``neuronxcc.nki.kernels`` module

* Documentation updates:

  * Kernels public repository https://aws-neuron.github.io/nki-samples
  * Updated :ref:`profiling guide <profile-using-nki-profile>` to use ``nki.profile`` instead of ``nki.benchmark``
  * NKI ISA Activation functions table now have :ref:`valid input data ranges<tbl-act-func>` listed
  * NKI ISA Supported Math operators now have :ref:`supported engine<tbl-aluop>` listed
  * Clarify ``+=`` syntax support/limitation

Neuron Kernel Interface (NKI) (Beta) [2.21]
------------------------------------------------
Date: 12/16/2024

* New modules and APIs:

  * ``nki.compiler`` module with Allocation Control and Kernel decorators,
    see guide for more info.
  * ``nki.isa``: new APIs (``activation_reduce``, ``tensor_partition_reduce``,
    ``scalar_tensor_tensor``, ``tensor_scalar_reduce``, ``tensor_copy``, 
    ``tensor_copy_dynamic_src``, ``dma_copy``), new activation functions(``identity``, 
    ``silu``, ``silu_dx``), and target query APIs (``nc_version``, ``get_nc_version``).
  * ``nki.language``: new APIs (``shared_identity_matrix``, ``tan``,
    ``silu``, ``silu_dx``, ``left_shift``, ``right_shift``, ``ds``, ``spmd_dim``, ``nc``).
  * New ``datatype <nl_datatypes>``: ``float8_e5m2``
  * New ``kernels`` (``allocated_fused_self_attn_for_SD_small_head_size``,
    ``allocated_fused_rms_norm_qkv``) added, kernels moved to public repository.


* Improvements:

  * Semantic analysis checks for nki.isa APIs to validate supported ops, dtypes, and tile shapes.
  * Standardized naming conventions with keyword arguments for common optional parameters.
  * Transition from function calls to kernel decorators (``jit``, 
    ``benchmark``, ``baremetal``, ``simulate_kernel``).

* Documentation updates:

  * New :doc:`Direct Allocation Developer Guide </nki/how-to-guides/nki_direct_allocation_guide>`
  * Tutorial for :doc:`SPMD usage with multiple Neuron Cores on Trn2 </nki/tutorials/spmd_multiple_nc_tensor_addition>`

Neuron Kernel Interface (NKI) (Beta)
------------------------------------------------
Date: 12/03/2024

* NKI support for Trainium2, including full integration with Neuron Compiler.
  Users can directly shard NKI kernels across multiple Neuron Cores from an SPMD launch grid.
  See :doc:`tutorial </nki/tutorials/spmd_multiple_nc_tensor_addition>` for more info.
  See :doc:`Trainium2 Architecture Guide </nki/about/trainium2_arch>` for an initial version of the architecture specification
  (more details to come in future releases).
* New calling convention in NKI kernels, where kernel output tensors are explicitly returned from the kernel instead
  of pass-by-reference. See any :doc:`NKI tutorial </nki/tutorials/index>` for code examples.

Neuron Kernel Interface (NKI) (Beta) [2.20]
-------------------------------------------
Date: 09/16/2024

* This release includes the beta launch of the Neuron Kernel Interface (NKI) (Beta).
  NKI is a programming interface enabling developers to build optimized compute kernels
  on top of Trainium and Inferentia. NKI empowers developers to enhance deep learning models
  with new capabilities, performance optimizations, and scientific innovation.
  It natively integrates with PyTorch and JAX, providing a Python-based programming environment
  with Triton-like syntax and tile-level semantics offering a familiar programming experience
  for developers. Additionally, to enable bare-metal access precisely programming the instructions
  used by the chip, this release includes a set of NKI APIs (``nki.isa``) that directly emit
  Neuron Instruction Set Architecture (ISA) instructions in NKI kernels.

