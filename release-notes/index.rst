.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:

.. _neuron-2.22.0-whatsnew:

Neuron 2.22.0 (04/03/2025)
---------------------------

.. contents:: Table of contents
   :local:
   :depth: 1

What's New
^^^^^^^^^^

The Neuron 2.22 release includes performance optimizations, enhancements and new capabilities across the Neuron software stack. 

For inference workloads, the NxD Inference library now supports Llama-3.2-11B model and supports multi-LoRA serving, allowing customers to load and serve multiple LoRA adapters. Flexible quantization features have been added, enabling users to specify which model layers or NxDI modules to quantize. Asynchronous inference mode has also been introduced, improving performance by overlapping Input preparation with model execution.

For training, we added LoRA supervised fine-tuning to NxD Training to enable additional model customization and adaptation.

Neuron Kernel Interface (NKI): This release adds new APIs in nki.isa, nki.language, and nki.profile. These enhancements provide customers with greater flexibility and control.

The updated Neuron Runtime includes optimizations for reduced latency and improved device memory footprint. On the tooling side, the Neuron Profiler 2.0 (beta) has added UI enhancements and new event type support.

Neuron DLCs: this release reduces DLC image size by up to 50% and enables faster build times with updated Dockerfiles structure. On the Neuron DLAMI side, new PyTorch 2.5 single framework DLAMIs have been added for Ubuntu 22.04 and Amazon Linux 2023, along with several new virtual environments within the Neuron Multi Framework DLAMIs.


More release content can be found in the table below and each component release notes.

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

   * - NeuronX Nemo Megatron for Training
     - * `neuronx-nemo-megatron github repo <https://github.com/aws-neuron/neuronx-nemo-megatron>`_  and  :ref:`neuronx-nemo-rn`
     - Trn1/Trn1n,Inf2

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
     - * :ref:`announce-eos-neuron-det`
       * :ref:`announce-eos-nxd-examples`
       * :ref:`announce-python-eos`
       * :ref:`announce-eos-pytorch-eos-113`
       * :ref:`announce-eos-pytorch-2-1`
       * :ref:`announce-u20-dlami-dlc-eos`
       * :ref:`announce-no-support-torch-neuron`
       * See more at :ref:`announcements-main`
     - Inf1, Inf2, Trn1/Trn1n


For detailed release artificats, see :ref:`Release Artifacts <latest-neuron-release-artifacts>`.


Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`
