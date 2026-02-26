.. meta::
    :description: Release notes for the Neuron Kernel Interface (NKI) component across all Neuron SDK versions
    :keywords: NKI, Neuron Kernel Interface, release notes, nki.language, nki.isa, kernels
    :date-modified: 02/26/2026

.. _nki_rn:

Release Notes for Neuron Component: Neuron Kernel Interface (NKI)
==================================================================

The release notes for the Neuron Kernel Interface (NKI) component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _nki-2-28-0-rn:   

Neuron Kernel Interface (NKI) (Beta 2 - 0.2.0) [2.28] (Neuron 2.28.0 Release)
-----------------------------------------------------------------------------

Date of Release: 02/26/2026

New Features
~~~~~~~~~~~~

* LNC (Large Neuron Core) multi-core support:

  * **Shared buffers and canonical outputs**: The compiler now tracks
    :doc:`shared_hbm </nki/api/generated/nki.language.shared_hbm>` tensors declared in kernels
    and canonicalizes LNC kernel outputs into a consistent form. This is foundational
    infrastructure for multi-core kernel compilation.
    See :doc:`LNC Overview </nki/get-started/about/lnc>`.

  * **Private HBM tensors**: Users can declare tensors private to a single NeuronCore using the
    :doc:`private_hbm </nki/api/generated/nki.language.private_hbm>` memory type, distinct from
    regular and shared HBM.

  * **Intra-LNC collectives**: New ISA instruction types for multi-core collective operations
    such as cross-core reductions and broadcasts. See full API listing under
    :doc:`nki.collectives </nki/api/nki.collectives>` below.

* New ``nki.isa`` APIs:

  * :doc:`nki.isa.nonzero_with_count </nki/api/generated/nki.isa.nonzero_with_count>` — returns nonzero element indices and their count, useful for sparse computation and dynamic masking
  * ``nki.isa.exponential`` — computes element-wise exponential on tensors. See :doc:`nki.isa.activation </nki/api/generated/nki.isa.activation>`.

* New :doc:`nki.collectives </nki/api/nki.collectives>` module, enabling collective communication across multiple NeuronCores directly from NKI kernels:

  * :doc:`nki.collectives.all_reduce </nki/api/generated/nki.collectives.all_reduce>`
  * :doc:`nki.collectives.all_gather </nki/api/generated/nki.collectives.all_gather>`
  * :doc:`nki.collectives.reduce_scatter </nki/api/generated/nki.collectives.reduce_scatter>`
  * :doc:`nki.collectives.all_to_all </nki/api/generated/nki.collectives.all_to_all>`
  * :doc:`nki.collectives.collective_permute </nki/api/generated/nki.collectives.collective_permute>`
  * :doc:`nki.collectives.collective_permute_implicit </nki/api/generated/nki.collectives.collective_permute_implicit>`
  * :doc:`nki.collectives.collective_permute_implicit_reduce </nki/api/generated/nki.collectives.collective_permute_implicit_reduce>`
  * :doc:`nki.collectives.rank_id </nki/api/generated/nki.collectives.rank_id>`

* New ``dtypes``:

  * :doc:`nki.language.float8_e4m3fn </nki/api/generated/nki.language.float8_e4m3fn>` — for FP8 inference and training workloads

* New NKI language features:

  * ``no_reorder`` blocks — use ``with no_reorder(): ...`` to prevent the compiler from reordering instructions within a block, for kernels where instruction ordering affects correctness
  * ``__call__`` special method support — callable objects (classes with ``__call__``) can now be used as functions within NKI kernels
  * ``tensor.view`` method — tensors now support ``.view()`` for reshaping
  * :doc:`Shared constants </nki/api/generated/nki.language.shared_constant>` can now be passed to kernels as string arguments, not just tensor objects

Improvements
~~~~~~~~~~~~

* Updated ``nki.isa`` APIs:

  * :doc:`nki.isa.dma_transpose </nki/api/generated/nki.isa.dma_transpose>` now supports indirect addressing
  * :doc:`nki.isa.dma_copy </nki/api/generated/nki.isa.dma_copy>` now supports ``unique_indices`` parameter
  * :doc:`nki.isa.register_alloc </nki/api/generated/nki.isa.register_alloc>` now accepts an optional tensor argument to pre-fill the allocated register with initial values

* Compiler output improvements:

  * The compiler no longer truncates diagnostic output; users now receive the full set of warnings and errors

Breaking Changes
~~~~~~~~~~~~~~~~

* :doc:`nki.isa.nc_matmul </nki/api/generated/nki.isa.nc_matmul>` parameter ``psumAccumulateFlag`` has been removed. This parameter had no effect on compilation or execution. Simply remove it from your kernel code.

* :doc:`nki.isa.nc_matmul </nki/api/generated/nki.isa.nc_matmul>` parameter ``is_moving_zero`` has been renamed to ``is_moving_onezero`` to match hardware semantics, consistent with the companion ``is_stationary_onezero`` parameter. Kernels that passed ``is_moving_zero`` by name should update to ``is_moving_onezero``.

* ``nki.tensor`` has moved to ``nki.meta.tensor``. Users should update their imports accordingly.

.. note::

   The previously announced removal of the ``neuronxcc.nki.*`` namespace has 
   been postponed from Neuron 2.28 to Neuron 2.29. Both the ``neuronxcc.nki.*`` 
   and ``nki.*`` namespaces continue to be supported in this release. We 
   encourage customers to migrate to the ``nki.*`` namespace using the 
   :doc:`NKI Migration Guide </nki/deep-dives/nki-migration-guide>`.

Bug Fixes
~~~~~~~~~

* Fixed incorrect default value for ``on_false_value`` in ``nki.isa.range_select``. The default
  was ``0.0`` instead of negative infinity (``-inf``). This caused ``range_select`` to write zeros
  for out-of-range elements instead of the expected negative-infinity sentinel, which could produce
  incorrect results in downstream reductions (e.g., max-pooling or top-k).
  See :doc:`nki.isa.range_select </nki/api/generated/nki.isa.range_select>`.

* Fixed default value parsing for keyword-only arguments in NKI kernels. When a Python function
  used keyword-only arguments with default values (arguments after ``*`` in the signature), the
  NKI compiler did not associate the defaults with their corresponding parameter names.
  This caused keyword-only arguments to appear as required even when they had defaults, leading to
  "missing argument" errors during kernel compilation.

* Fixed wrong default for ``reduce_cmd`` in :doc:`nki.isa.activation </nki/api/generated/nki.isa.activation>`.
  The default was incorrectly set to ``ZeroAccumulate`` instead of ``Idle``, causing the accumulator
  to be zeroed before every activation call even when no reduction was requested.

* Fixed missing ALU operators (``rsqrt``, ``abs``, ``power``) in
  :doc:`nki.isa.tensor_scalar </nki/api/generated/nki.isa.tensor_scalar>` and
  :doc:`nki.isa.tensor_tensor </nki/api/generated/nki.isa.tensor_tensor>`. Passing these operators
  previously raised an "unsupported operator" error.
  See :doc:`NKI Language Guide </nki/get-started/nki-language-guide>`.

* Fixed ``float8_e4m3fn`` to ``float8_e4m3`` conversion for kernel inputs and outputs. When a
  tensor with dtype ``float8_e4m3fn`` was passed to the compiler, the automatic conversion to
  ``float8_e4m3`` could fail with a size-check error. The conversion now validates sizes
  correctly before casting.
  See :doc:`nki.language.float8_e4m3 </nki/api/generated/nki.language.float8_e4m3>`.

* Fixed dynamic for loop incorrectly incrementing the loop induction variable. In loops with a
  runtime-determined trip count (``sequential_range`` with non-constant bounds), the compiler
  generated incorrect increment code, causing the loop counter to never advance and the loop to
  run indefinitely or produce incorrect iteration values.
  See :doc:`nki.language.sequential_range </nki/api/generated/nki.language.sequential_range>`.

* Fixed reshape of ``shared_hbm`` and ``private_hbm`` tensors failing partition size check.
  Reshape only recognized plain ``hbm`` memory as exempt from partition-dimension size validation.
  Tensors allocated in ``shared_hbm`` or ``private_hbm`` (used for cross-kernel and
  kernel-private storage) incorrectly triggered a "partition size mismatch" error when reshaped.
  See :doc:`nki.language.shared_hbm </nki/api/generated/nki.language.shared_hbm>` and
  :doc:`nki.language.private_hbm </nki/api/generated/nki.language.private_hbm>`.

* Fixed bias shape checking in :doc:`nki.isa.activation </nki/api/generated/nki.isa.activation>`.
  The ``bias`` parameter was not validated for shape correctness. A bias tensor with a free
  dimension other than 1 (e.g., shape ``(128, 64)`` instead of ``(128, 1)``) was accepted
  without validation, which could produce incorrect results. The compiler now raises an error if the bias
  free dimension is not 1.

* Fixed incorrect line numbers in stack traces and error reporting. An off-by-one error in the
  line offset calculation caused all reported line numbers to be shifted by one.
  Additionally, error location was sometimes lost when errors propagated across file boundaries.

* Fixed invalid keyword arguments being silently ignored instead of raising an error. When calling
  an NKI API with a misspelled or unsupported keyword argument, the argument was ignored
  without warning.
  The compiler now validates all keyword argument names against the function signature and raises
  an ``unexpected keyword argument`` error for unrecognized names.

* Fixed ``nki.jit`` in auto-detection mode returning an uncalled kernel object instead of
  executing the kernel. When ``nki.jit`` was used without specifying a framework mode (e.g.,
  ``@nki.jit`` with no ``mode`` argument), the auto-detection path constructed the appropriate
  framework-specific kernel object but returned it without calling it. The user received a kernel
  object instead of the computed result, requiring an extra manual invocation.
  See :doc:`nki.jit </nki/api/generated/nki.jit>`.

* Fixed stale kernel object state between trace invocations. When tracing the same kernel
  multiple times (e.g., with different input shapes), compiler state was not fully reset
  between invocations, causing name collisions and incorrect results.
  The trace state is now fully reset before each invocation.

* Improved 'removed during code migration' error messages with clear descriptions of
  unimplemented features. APIs not available in this release (``nki.baremetal``, ``nki.benchmark``,
  ``nki.profile``, ``nki.simulate_kernel``) previously raised a generic
  ``NotImplementedError("removed during code migration")`` message. Each now raises a specific
  message naming the unsupported API. Additionally, calling an ``nki.jit`` kernel with no
  arguments now raises a clear error instead.
  See :doc:`NKI Migration Guide </nki/deep-dives/nki-migration-guide>`.

* Fixed nested ``nki_jit`` decorators not being allowed. The NKI compiler only recognized
  ``@nki.jit``-decorated functions when they were plain function objects. Nested decorators
  (e.g., ``@my_wrapper @nki.jit``) wrapped the function in a non-function object, causing the
  compiler to skip it. The compiler now correctly unwraps decorator chains to find the underlying
  kernel function. See :doc:`nki.jit </nki/api/generated/nki.jit>`.

Known Issues
~~~~~~~~~~~~

* ``nki.isa.range_select``: The ``on_false_value`` and ``reduce_cmd`` parameters are incorrectly 
ignored by the NKI compiler. The ``on_false_value`` is always set to ``(-3.4028235e+38)`` 
and ``reduce_cmd`` is always set to ``reduce_cmd.reset_reduce``, regardless of the values passed in.

.. _nki-2-27-0-rn:

Neuron Kernel Interface (NKI) (Beta 2 - 0.1.0) [2.27] (Neuron 2.27.0 Release)
-----------------------------------------------------------------------------

Date: 12/25/2025

Improvements
~~~~~~~~~~~~~~~

* new ``nki.language`` APIs:

  * ``nki.language.device_print``

* new ``nki.isa`` APIs:

  * ``nki.isa.dma_compute``
  * ``nki.isa.nki.isa.quantize_mx``
  * ``nki.isa.nki.isa.nc_matmul``
  * ``nki.isa.nki.isa.nc_n_gather`` [used to be ``nl.gather_flattened`` with free partition limited to 512]
  * ``nki.isa.rand2``
  * ``nki.isa.rand_set_state``
  * ``nki.isa.rand_get_state``
  * ``nki.isa.set_rng_seed``
  * ``nki.isa.rng``

* new ``dtypes``:

  * ``nki.language.float8_e5m2_x4``
  * ``nki.language.float4_e2m1fn_x4``
  * ``nki.language.float8_e4m3fn_x4``

* changes to existing APIs:

  * several ``nki.language`` APIs have been removed in NKI Beta 2
  * all nki.isa APIs have ``dst`` as an input param
  * all nki.isa APIs removed ``dtype`` and ``mask`` support
  * ``nki.isa.memset`` — removed ``shape`` positional arg , since we have ``dst``
  * ``nki.isa.affine_select`` — instead of ``pred``, we now take ``pattern`` and ``cmp_op`` params
  * ``nki.isa.iota`` — ``expr`` replaced with ``pattern`` and ``offset``
  * ``nki.isa.nc_stream_shuffle`` - ``src`` and ``dst`` order changed

* docs improvements:

  * restructured NKI Documentation to align with workflows
  * added :doc:`Trainium3 Architecture Guide for NKI </nki/guides/architecture/trainium3_arch>`
  * added :doc:`About Neuron Kernel Interface (NKI) </nki/get-started/about/index>`
  * added :doc:`NKI Environment Setup Guide </nki/get-started/setup-env>`
  * added :doc:`Get Started with NKI </nki/get-started/quickstart-implement-run-kernel>`
  * added :doc:`NKI Language Guide </nki/get-started/nki-language-guide>`
  * added :doc:`About the NKI Compiler </nki/deep-dives/nki-compiler>`
  * added :doc:`About NKI Beta Versions </nki/deep-dives/nki-beta-versions>`
  * added :doc:`MXFP Matrix Multiplication with NKI </nki/deep-dives/mxfp-matmul>`
  * updated :doc:`Matrix Multiplication Tutorial </nki/guides/tutorials/matrix_multiplication>`
  * updated :doc:`Profile a NKI Kernel </nki/deep-dives/use-neuron-profile>`
  * updated :doc:`NKI APIs </nki/api/index>`
  * updated :doc:`NKI Library docs </nki/library/index>`
  * removed NKI Error Guide

Known Issues
~~~~~~~~~~~~

* ``nki.isa.nki.isa.nc_matmul`` - ``is_moving_onezero`` was incorrectly named ``is_moving_zero`` in this release
* NKI ISA semantic checks are not available with Beta 2, workaround is to reference the API docs
* NKI Collectives are not available with Beta 2
* ``nki.benchmark`` and ``nki.profile`` are not available with Beta 2


----

.. _nki-2-26-0-rn:

Neuron Kernel Interface (NKI) (Beta) [2.26] (Neuron 2.26.0 Release)
--------------------------------------------------------------------

Date: 09/18/2025

Improvements
~~~~~~~~~~~~~~~

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


----

.. _nki-2-24-0-rn:

Neuron Kernel Interface (NKI) (Beta) [2.24] (Neuron 2.24.0 Release)
--------------------------------------------------------------------

Date: 06/24/2025

Improvements
~~~~~~~~~~~~~~~

* ``sqrt`` valid data range extended for accuracy improvement with wider numerical values support.
* ``nki.language.gather_flattened`` new API
* ``nki.isa.nc_match_replace8`` additional param ``dst_idx`` 
* improved docs/examples on ``nki.isa.nc_match_replace8``, ``nki.isa.nc_stream_shuffle`` 
* improved error messages


----

.. _nki-2-23-0-rn:

Neuron Kernel Interface (NKI) (Beta) [2.23] (Neuron 2.23.0 Release)
--------------------------------------------------------------------

Date: 05/20/2025

Improvements
~~~~~~~~~~~~~~~

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


----

.. _nki-2-22-0-rn:

Neuron Kernel Interface (NKI) (Beta) [2.22] (Neuron 2.22.0 Release)
--------------------------------------------------------------------

Date: 04/03/2025

Improvements
~~~~~~~~~~~~~~~

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
  * Updated :doc:`profiling guide </nki/deep-dives/use-neuron-profile>` to use ``nki.profile`` instead of ``nki.benchmark``
  * NKI ISA Activation functions table now have :ref:`valid input data ranges<tbl-act-func>` listed
  * NKI ISA Supported Math operators now have :ref:`supported engine<tbl-aluop>` listed
  * Clarify ``+=`` syntax support/limitation


----

.. _nki-2-21-0-rn:

Neuron Kernel Interface (NKI) (Beta) [2.21] (Neuron 2.21.0 Release)
--------------------------------------------------------------------

Date: 12/16/2024

Improvements
~~~~~~~~~~~~~~~

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

  * Tutorial for :doc:`SPMD usage with multiple Neuron Cores on Trn2 </nki/guides/tutorials/spmd_multiple_nc_tensor_addition>`


----

.. _nki-2-20-1-rn:

Neuron Kernel Interface (NKI) (Beta) (Neuron 2.20.1 Release)
-------------------------------------------------------------

Date: 12/03/2024

Improvements
~~~~~~~~~~~~~~~

* NKI support for Trainium2, including full integration with Neuron Compiler.
  Users can directly shard NKI kernels across multiple Neuron Cores from an SPMD launch grid.
  See :doc:`tutorial </nki/guides/tutorials/spmd_multiple_nc_tensor_addition>` for more info.
  See :doc:`Trainium2 Architecture Guide </nki/guides/architecture/trainium2_arch>` for an initial version of the architecture specification
  (more details to come in future releases).
* New calling convention in NKI kernels, where kernel output tensors are explicitly returned from the kernel instead
  of pass-by-reference. See any :doc:`NKI tutorial </nki/guides/tutorials/index>` for code examples.


----

.. _nki-2-20-0-rn:

Neuron Kernel Interface (NKI) (Beta) [2.20] (Neuron 2.20.0 Release)
--------------------------------------------------------------------

Date: 09/16/2024

Improvements
~~~~~~~~~~~~~~~

* This release includes the beta launch of the Neuron Kernel Interface (NKI) (Beta).
  NKI is a programming interface enabling developers to build optimized compute kernels
  on top of Trainium and Inferentia. NKI empowers developers to enhance deep learning models
  with new capabilities, performance optimizations, and scientific innovation.
  It natively integrates with PyTorch and JAX, providing a Python-based programming environment
  with Triton-like syntax and tile-level semantics offering a familiar programming experience
  for developers. Additionally, to enable bare-metal access precisely programming the instructions
  used by the chip, this release includes a set of NKI APIs (``nki.isa``) that directly emit
  Neuron Instruction Set Architecture (ISA) instructions in NKI kernels.




    