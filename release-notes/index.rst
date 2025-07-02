.. _neuron-whatsnew:

What's New
==========

.. _latest-neuron-release:
.. _neuron-2.24.1-whatsnew:
Neuron 2.24.1 (06/30/2025)
---------------------------

Neuron version 2.24.1 resolves an installation issue that could prevent NeuronX Distributed Training from being installed successfully.

.. _neuron-2.24.0-whatsnew:

Neuron 2.24.0 (06/24/2025)
---------------------------

Neuron version 2.24 introduces new inference capabilities including prefix caching, disaggregated inference (Beta), and context parallelization support (Beta). This release also includes NKI language enhancements and enhanced profiling visualizations for improved debugging and performance analysis. Neuron 2.24 adds support for PyTorch 2.7 and JAX 0.6, updates existing DLAMIs and DLCs, and introduces a new vLLM inference container.

.. contents:: Table of contents
   :local:
   :depth: 1

Inference
^^^^^^^^^

NxD Inference (NxDI) includes the following enhancements:

- **Prefix caching**: Improves Time To First Token (TTFT) by up to 3x when processing common shared prompts across requests.
- **Disaggregated inference (Beta)**: Uses 1P1D (1 Prefill, 1 Decode) architecture to reduce prefill-decode interference and improve goodput.
- **Context parallelism (Beta)**: Improves TTFT for longer sequence lengths by processing context encoding in parallel across multiple NeuronCores.
- **Model support**: Added beta support for Qwen 2.5 text models.
- **NxD Inference Library**: Upgraded to support PyTorch 2.7 and Transformers 4.48.

Hugging Face Optimum Neuron 0.2.0 now supports PyTorch-based NxD Core backend for LLM inference, simplifying the implementation of new PyTorch model architectures. Models including Llama 3.1-8B and Llama-3.3-70B have migrated from Transformers NeuronX to the NxD backend.

Training
^^^^^^^^

**Library Upgrades**


- **NxD Training  (NxDT) Library**: Upgraded to support PyTorch 2.7 and Transformers 4.48.
- **JAX Training Support**: Upgraded to JAX 0.6.0.

Neuron Kernel Interface (NKI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **New nki.language.gather_flattened**: Provides efficient parallel tensor element gathering.
- **Enhanced accuracy**: Improved valid range of ``nki.language.sqrt`` and ``nki.isa.activation(nl.sqrt)`` 
- **Advanced indexing**: Improved performance for ``nki.isa.nc_match_replace8``.

Neuron Tools
^^^^^^^^^^^^

**Neuron Profiler Enhancements**

- **Framework stack traces**: Maps device instructions to model source code.
- **Scratchpad memory usage visualization**: Shows tensor-level memory usage over time with HLO name association.
- **On-device collectives barriers**: Identifies synchronization overhead.
- **HBM throughput visualization**: Tracks data movement involving High Bandwidth Memory (HBM) over time.

**NCCOM-TEST Improvements**

- Added ``--report-to-json-file`` flag: Outputs results in JSON format.
- Added ``--show-input-output-size`` flag: Explicitly displays input and output sizes based on operations.

Neuron Deep Learning Containers (DLCs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Updated containers with PyTorch 2.7 support for inference and training.
- Added new inference container with NxD Inference and vLLM with FastAPI.
- JAX DLCs now support JAX 0.6.0 training.

Neuron Deep Learning AMIs (DLAMIs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Updated MultiFramework DLAMIs to include PyTorch 2.7 and JAX 0.6.0.
- Added new Single Framework DLAMIs for PyTorch 2.7 and JAX 0.6.0.


Neuron 2.24 Feature Release Notes
---------------------------------

.. list-table::
   :widths: auto
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - What's New
     - Details
     - Instances

   * - NxD Core (neuronx-distributed) 
     - * :ref:`neuronx-distributed-rn`   
     - ``Trn1`` / ``Trn1n``, ``Trn2``

   * - NxD Inference (neuronx-distributed-inference)
     - * :ref:`neuronx-distributed-inference-rn` 
     - ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

   * - NxD Training (neuronx-distributed-training)
     - * :ref:`neuronx-distributed-training-rn` 
     - ``Trn1`` / ``Trn1n``, ``Trn2``

   * - PyTorch NeuronX (torch-neuronx)
     - * :ref:`torch-neuronx-rn`
     - ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

   * - Neuron Compiler (neuronx-cc)
     - * :ref:`neuronx-cc-rn`
     - ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

   * - Neuron Kernel Interface (NKI)
     - * :ref:`nki_rn`
     - ``Inf2``, ``Trn1``/ ``Trn1n``

   * - Neuron Tools
     - * :ref:`neuron-tools-rn`
     - ``Inf1``, ``Inf2``, ``Trn1``/ ``Trn1n``

   * - Neuron Runtime
     - * :ref:`neuron-runtime-rn`
     - ``Inf1``, ``Inf2``, ``Trn1``/ ``Trn1n``

   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * :ref:`transformers-neuronx-rn` 
     - ``Inf2``, ``Trn1`` / ``Trn1n``

   * - Neuron Deep Learning AMIs (DLAMIs)
     - * :ref:`neuron-dlami-overview`
     - ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``

   * - Neuron Deep Learning Containers (DLCs)
     - * :ref:`neuron-dlc-release-notes`
     - ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``

   * - Release Announcements
     - * :ref:`announce-no-longer-support-beta-pytorch-neuroncore-placement-apis`
       * :ref:`announce-eos-block-dimension-nki`
       * :ref:`announce-eos-pytorch25`
       * :ref:`announce-eos-tensorflow-tutorial`
       * :ref:`announce-eos-tnx`
       * :ref:`announce-eos-longer-support-xla-bf16-vars`
       * :ref:`announce-eos-block-dimension-nki`
       * :ref:`announce-no-longer-support-llama-32-meta-checkpoint`
       * :ref:`announce-no-longer-support-nki-jit`
       * See more at :ref:`announcements-main`.
     - ``Inf1``, ``Inf2``, ``Trn1``/ ``Trn1n``

For detailed release artifacts, see :ref:`Release Artifacts <latest-neuron-release-artifacts>`.


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`
