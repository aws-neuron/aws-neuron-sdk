.. _nki_rn:

Neuron Kernel Interface (NKI) release notes
==============================================
.. .. contents:: Table of Contents
..    :local:

..    :depth: 2

Neuron Kernel Interface (NKI) (Beta)
--------------------------------------

Date: 12/03/2024

* NKI support for Trainium2, including full integration with Neuron Compiler. 
  Users can directly shard NKI kernels across multiple Neuron Cores from an SPMD launch grid. 
  See :doc:`tutorial <tutorials/spmd_multiple_nc_tensor_addition>` for more info.

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
  `mamba <https://github.com/aws-neuron/nki-samples/blob/main/src/tutorials/fused_mamba/mamba_torch.py>`_
  and `flash attention <https://github.com/aws-neuron/nki-samples/blob/main/src/reference/attention.py>`_
  as open-source samples in a new `nki-samples <https://github.com/aws-neuron/nki-samples/>`_ 
  GitHub repository. New kernel contributions are welcome via GitHub Pull-Requests as well as
  feature requests and bug reports as GitHub Issues. For more information see the 
  :doc:`latest documentation <index>`.
  Included in this initial beta release is an in-depth :doc:`getting started <getting_started>`,
  :doc:`architecture <trainium_inferentia2_arch>`, :doc:`profiling <neuron_profile_for_nki>`,
  and :doc:`performance guide <nki_perf_guide>`, along with multiple :doc:`tutorials <tutorials>`,
  :doc:`api reference documents <api/index>`, documented :doc:`known issues <nki_known_issues>` 
  and :doc:`frequently asked questions <nki_faq>`.

