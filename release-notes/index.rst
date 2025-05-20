.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:
.. _neuron-2.23.0-whatsnew:

Neuron 2.23.0 (05/20/2025)
---------------------------

.. contents:: Table of contents
   :local:
   :depth: 1

What's New
^^^^^^^^^^

With the Neuron 2.23 release, we move NxD Inference (NxDI) library out of beta. It is now recommended for all multi-chip inference use-cases. In addition, Neuron has new training capabilities, including Context Parallelism and ORPO, NKI improvements (new operators and ISA features), and new Neuron Profiler debugging and performance analysis optimizations. Finaly, Neuron now supports :ref:`PyTorch 2.6 <introduce-pytorch-2-6>` and JAX 0.5.3.

Inference: NxD Inference (NxDI) moves from beta to GA. NxDI now supports Persistent Cache to reduce compilation times, and optimizes model loading with improved weight sharding performance.

Training: NxD Training (NxDT) added Context Parallelism support (beta) for Llama models, enabling sequence lengths up to 32K. NxDT now supports model alignment, ORPO, using DPO-style datasets. NxDT has upgraded supports for 3rd party libraries, specifically: PyTorch Lightning 2.5, Transformers 4.48, and NeMo 2.1.

Neuron Kernel Interface (NKI): New support for 32-bit integer nki.language.add and nki.language.multiply on GPSIMD Engine. NKI.ISA improvements include range_select for Trainium2, fine-grained engine control, and enhanced tensor operations. New performance tuning API `no_reorder` has been added to enable user-scheduling of instructions. When combined with allocation, this enables software pipelining. Language consistency has been improved for arithmetic operators (+=, -=, /=, *=) across loop types, PSUM, and SBUF.

Neuron Profiler: Profiling performance has improved, allowing users to view profile results 5x times faster on average. New features include timeline-based error tracking and JSON error event reporting, supporting execution and OOB error detection. Additionally, this release improves multiprocess visualization with Perfetto. 

Neuron Monitoring: Added Kubernetes context information (pod_name, namespace, and container_name) to neuron monitor prometheus output, enabling resource utilization tracking by pod, namespace, and container.

Neuron DLCs: This release updates containers with PyTorch 2.6 support for inference and training. For JAX DLC, this release adds JAX 0.5.0 training support.

Neuron DLAMIs: This release updates MultiFramework AMIs to include PyTorch 2.6, JAX 0.5, and TensorFlow 2.10 and Single Framework AMIs for PyTorch 2.6 and JAX 0.5.

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
     - Trn1/Trn1n,Trn2

   * - NxD Inference (neuronx-distributed-inference)
     - * :ref:`neuronx-distributed-inference-rn` 
     - Inf2, Trn1/Trn1n,Trn2

   * - NxD Training (neuronx-distributed-training)
     - * :ref:`neuronx-distributed-training-rn` 
     - Trn1/Trn1n,Trn2

   * - PyTorch NeuronX (torch-neuronx)
     - * :ref:`torch-neuronx-rn`
     - Trn1/Trn1n,Inf2,Trn2

   * - Neuron Compiler (neuronx-cc)
     - * :ref:`neuronx-cc-rn`
     - Trn1/Trn1n,Inf2,Trn2

   * - Neuron Kernel Interface (NKI)
     - * :ref:`nki_rn`
     - Trn1/Trn1n,Inf2

   * - Neuron Tools
     - * :ref:`neuron-tools-rn`
     - Inf1,Inf2,Trn1/Trn1n,Trn2

   * - Neuron Runtime
     - * :ref:`neuron-runtime-rn`
     - Inf1,Inf2,Trn1/Trn1n,Trn2

   * - Transformers NeuronX (transformers-neuronx) for Inference
     - * :ref:`transformers-neuronx-rn` 
     - Inf2, Trn1/Trn1n

   * - Neuron Deep Learning AMIs (DLAMIs)
     - * :ref:`neuron-dlami-overview`
     - Inf1,Inf2,Trn1/Trn1n

   * - Neuron Deep Learning Containers (DLCs)
     - * :ref:`neuron-dlc-release-notes`
     - Inf1,Inf2,Trn1/Trn1n

   * - Release Annoucements
     - * :ref:`announce-eos-block-dimension-nki`
       * :ref:`announce-eos-mllama-checkpoint`
       * :ref:`announce-eos-nxdt-megatron-models`
       * :ref:`announce-eos-torch-neuronx-nki-jit`
       * :ref:`announce-eos-xla-bf`
       * :ref:`announce-no-support-jax-neuronx-features`
       * :ref:`announce-no-support-nemo-megatron`
       * :ref:`announce-no-support-tensorflow-eos`
       * :ref:`announce-u20-base-no-support`
       * :ref:`announce-tnx-maintenance`
       * :ref:`announce-eol-nxd-examples`
       * See more at :ref:`announcements-main`
     - Inf1, Inf2, Trn1/Trn1n

For detailed release artificats, see :ref:`Release Artifacts <latest-neuron-release-artifacts>`.


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`
