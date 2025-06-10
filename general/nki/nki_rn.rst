.. _nki_rn:

Neuron Kernel Interface (NKI) release notes
==============================================
.. .. contents:: Table of Contents
..    :local:

..    :depth: 2

Neuron Kernel Interface (NKI) (Beta) [2.22]
------------------------------------------------
Date: 04/03/2025

* New modules and APIs:

  * :doc:`nki.profile <api/generated/nki.profile>`
  * :doc:`nki.isa <api/nki.isa>` new APIs:
    
    * ``tensor_copy_dynamic_dst``
    * ``tensor_copy_predicated``
    * ``max8``, ``nc_find_index8``, ``nc_match_replace8``
    * ``nc_stream_shuffle``
  
  * :doc:`nki.language <api/nki.language>` new APIs: ``mod``, ``fmod``, ``reciprocal``, ``broadcast_to``, ``empty_like``

* Improvements:

  * :doc:`nki.isa.nc_matmul <api/generated/nki.isa.nc_matmul>` now supports PE tiling feature 
  * :doc:`nki.isa.activation <api/generated/nki.isa.activation>` updated to support reduce operation and :doc:`reduce commands <api/generated/nki.isa.reduce_cmd>`
  * :doc:`nki.isa.engine <api/generated/nki.isa.engine>` enum
  * ``engine`` parameter added to more ``nki.isa`` APIs that support engine selection (ie, ``tensor_scalar``, ``tensor_tensor``, ``memset``)
  * Documentation for ``nki.kernels`` have been moved to the Github: https://aws-neuron.github.io/nki-samples. 
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

  * :doc:`nki.compiler <api/nki.compiler>` module with Allocation Control and Kernel decorators,
    see guide for more info.
  * :doc:`nki.isa <api/nki.isa>`: new APIs (``activation_reduce``, ``tensor_partition_reduce``,
    ``scalar_tensor_tensor``, ``tensor_scalar_reduce``, ``tensor_copy``, 
    ``tensor_copy_dynamic_src``, ``dma_copy``), new activation functions(``identity``, 
    ``silu``, ``silu_dx``), and target query APIs (``nc_version``, ``get_nc_version``).
  * :doc:`nki.language <api/nki.language>`: new APIs (``shared_identity_matrix``, ``tan``,
    ``silu``, ``silu_dx``, ``left_shift``, ``right_shift``, ``ds``, ``spmd_dim``, ``nc``).
  * New :ref:`datatype <nl_datatypes>`: ``float8_e5m2``
  * New :doc:`kernels <api/nki.kernels>` (``allocated_fused_self_attn_for_SD_small_head_size``,
    ``allocated_fused_rms_norm_qkv``) added, kernels moved to public repository.


* Improvements:

  * Semantic analysis checks for nki.isa APIs to validate supported ops, dtypes, and tile shapes.
  * Standardized naming conventions with keyword arguments for common optional parameters.
  * Transition from function calls to kernel :ref:`decorators <nki_decorators>` (``jit``, 
    ``benchmark``, ``baremetal``, ``simulate_kernel``).

* Documentation updates:

  * New :doc:`Direct Allocation Developer Guide <nki_direct_allocation_guide>`
  * Tutorial for :doc:`SPMD usage with multiple Neuron Cores on Trn2 <tutorials/spmd_multiple_nc_tensor_addition>`

Neuron Kernel Interface (NKI) (Beta)
------------------------------------------------
Date: 12/03/2024

* NKI support for Trainium2, including full integration with Neuron Compiler.
  Users can directly shard NKI kernels across multiple Neuron Cores from an SPMD launch grid.
  See :doc:`tutorial <tutorials/spmd_multiple_nc_tensor_addition>` for more info.
  See :doc:`Trainium2 Architecture Guide <arch/trainium2_arch>` for an initial version of the architecture specification
  (more details to come in future releases).
* New calling convention in NKI kernels, where kernel output tensors are explicitly returned from the kernel instead
  of pass-by-reference. See any :doc:`NKI tutorial <tutorials>` for code examples.

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

* In addition to documentation, we've included many of the innovative kernels
  used with-in the neuron-compiler such as
  `mamba <https://github.com/aws-neuron/nki-samples/tree/main/src/nki_samples/tutorials/fused_mamba/mamba_torch.py>`_
  and `flash attention <https://github.com/aws-neuron/nki-samples/tree/main/src/nki_samples/reference/attention.py>`_
  as open-source samples in a new `nki-samples <https://github.com/aws-neuron/nki-samples/>`_
  GitHub repository. New kernel contributions are welcome via GitHub Pull-Requests as well as
  feature requests and bug reports as GitHub Issues. For more information see the
  :doc:`latest documentation <index>`.
  Included in this initial beta release is an in-depth :doc:`getting started <getting_started>`,
  :doc:`architecture <arch/trainium_inferentia2_arch>`, :doc:`profiling <neuron_profile_for_nki>`,
  and :doc:`performance guide <nki_perf_guide>`, along with multiple :doc:`tutorials <tutorials>`,
  :doc:`api reference documents <api/index>`, documented :doc:`known issues <nki_known_issues>`
  and :doc:`frequently asked questions <nki_faq>`.
