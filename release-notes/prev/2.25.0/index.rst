.. _neuron-2-25-0-whatsnew:

.. meta::
   :description: The official release notes for the AWS Neuron SDK, version 2.25.0. Release date: 7/31/2025.

AWS Neuron SDK 2.25.0 release notes
===================================

**Date of release**: July 31, 2025

.. toctree::
   :hidden:
   :maxdepth: 1

   PyTorch support <nx-pytorch>
   JAX support <nx-jax>
   NxD Inference <nxd-inference>
   NxD Training <nxd-training>
   NxD Core <nxd-core>
   Neuron Compiler <compiler>
   Neuron Runtime <runtime>
   Developer tools <tools>
   Deep Learning AMIs <dlami>
   Deep Learning Containers <containers>
   Docs and samples <docs-and-samples>
   Release artifacts </release-notes/releasecontent>

.. contents:: In this release
   :local:
   :depth: 2

Release highlights
------------------

Neuron 2.25.0 delivers updates across several key areas: inference performance optimizations, expanded model support, enhanced profiling capabilities, improved monitoring and observability tools, framework updates, and refreshed development environments and container offerings. The release includes bug fixes across the SDK components, along with updated tutorials and documentation for new features and model deployments.


Inference Optimizations (NxD Core and NxDI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron 2.25.0 introduces performance optimizations and new capabilities including:

* Context and Data Parallel support for improved batch scaling
* Chunked Attention for improved long sequence processing
* Automatic Aliasing (Beta) for fast tensor operations
* Disaggregated Serving (Beta) improvements

Model Support (NxDI)
^^^^^^^^^^^^^^^^^^^^

Neuron 2.25.0 expands model support to include:

* Qwen3 dense models (0.6B to 32B parameters)
* Flux.1-dev model for text-to-image generation (Beta)

Monitoring and Observability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``neuron-ls`` now displays CPU and NUMA node affinity information
* ``neuron-ls`` adds NeuronCore IDs display for each Neuron Device
* ``neuron-monitor`` improves accuracy of device utilization metrics

Framework Updates
^^^^^^^^^^^^^^^^^

* JAX 0.6.1 support added, maintaining compatibility with versions 0.4.31-0.4.38 and 0.5
* vLLM support upgraded to version 0.9.x V0

Development Environment Updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron SDK updated to version 2.25.0 in:

* Deep Learning AMIs on Ubuntu 22.04 and Amazon Linux 2023
* Multi-framework DLAMI with environments for both PyTorch and JAX
* PyTorch 2.7 Single Framework DLAMI
* JAX 0.6 Single Framework DLAMI

Container Support
^^^^^^^^^^^^^^^^^

Neuron SDK updated to version 2.25.0 in:

* PyTorch 2.7 Training and Inference DLCs
* JAX 0.6 Training DLC
* vLLM 0.9.1 Inference DLC
* Neuron Device Plugin and Scheduler container images for Kubernetes integration

Component release notes
-----------------------

Select a card below to review detailed release notes for each component of the Neuron SDK version 2.25.0. These component release notes contain details on specific new and improved features, as well as breaking changes, bug fixes, and known issues for that component area of the Neuron SDK.

.. grid:: 1 1 2 2
        :gutter: 2

        .. grid-item-card:: 
                :link: neuron-2-25-0-pytorch
                :link-type: ref

                **PyTorch framework** 2.25.0 release notes
                ^^^
                Neuron features and solutions that support the PyTorch ML framework.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

        .. grid-item-card:: 
                :link: neuron-2-25-0-jax
                :link-type: ref

                **JAX framework** 2.25.0 release notes
                ^^^
                Neuron features and solutions that support the JAX ML framework.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

        .. grid-item-card:: 
                :link: neuron-2-25-0-nxd-training
                :link-type: ref

                **NxD Training** 2.25.0 release notes
                ^^^
                Neuron features and tools for LLM and agent ML model training.
                +++
                Supports: ``Trn1`` / ``Trn1n``, ``Trn2``

        .. grid-item-card:: 
                :link: neuron-2-25-0-nxd-inference
                :link-type: ref

                **NxD Inference** 2.25.0 release notes
                ^^^
                Neuron features and tools for LLM and agent ML model inference.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``
        
        .. grid-item-card::
                :link: neuron-2-25-0-nxd-core
                :link-type: ref

                **NxD Core** 2.25.0 release notes
                ^^^
                Common features and tools for Neuron-based training and inference.
                +++
                Supports: ``Trn1`` / ``Trn1n``, ``Trn2``
         
        .. grid-item-card:: 
                :link: neuron-2-25-0-compiler
                :link-type: ref

                **Neuron Compiler** 2.25.0 release notes
                ^^^
                The Neuron compiler for AWS Trainium and Inferentia, and its libraries and tools.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

        .. grid-item-card:: 
                :link: neuron-2-25-0-runtime
                :link-type: ref

                **Neuron Runtime** 2.25.0 release notes
                ^^^
                The Neuron kernel driver and C++ libraries for AWS Inferentia and Trainium instances.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``

        .. grid-item-card:: 
                :link: neuron-2-25-0-tools
                :link-type: ref

                **Neuron Developer Tools** 2.25.0 release notes
                ^^^
                Tools that support end-to-end development for AWS Neuron.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``


        .. grid-item-card:: 
                :link: neuron-2-25-0-dlami
                :link-type: ref

                **Neuron Deep Learning AWS Machine Images (DLAMIs)** 2.25.0 release notes
                ^^^
                AWS-specific machine images for building and deploying Neuron-based ML solutions.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``
 
        .. grid-item-card:: 
                :link: neuron-2-25-0-dlc
                :link-type: ref

                **Neuron Deep Learning Containers (DLCs)** 2.25.0 release notes
                ^^^
                AWS-specific container definitions for building and deploying Neuron-based ML solutions.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``

        .. grid-item-card::
                :link: neuron-2-25-0-docs-and-samples
                :link-type: ref

                **Documentation and samples** 2.25.0 release notes
                ^^^
                Changes to the Neuron docs and code samples.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n`` 

        .. grid-item-card::
                :link: latest-neuron-release-artifacts
                :link-type: ref
        
                **Neuron 2.25.0 release artifacts**
                ^^^
                The libraries and packages updated in this release.

Support announcements
---------------------

This section signals the official end-of-support or end of support for specific features, tools, and APIs.

End-of-support announcements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*An "end-of-support (EoS)" announcement is a notification that a feature, tool, or API will not be supported in the future. Plan accordingly!*

* In a future release, the Neuron Compiler default flag ``--auto-cast=matmult`` will change to ``--auto-cast=none``.

  This means the Neuron Compiler will no longer perform auto-casting and use the data types of the operators in the incoming HLO. If the current behavior is desired, users can explicitly pass the  ``--auto-cast=matmult`` and  ``--auto-cast-type=bf16`` options to the compiler.  

  **Note:** This change will not affect Neuron NxDI, NxDT, and TNx Frameworks as these are set to ``--auto-cast=none`` by default. However, Torch-Neuronx users may experience an impact and must adjust their settings if they rely on the previous auto-casting behavior.

* Starting from Neuron Release 2.24, the Hugging Face Transformers NeuronX library is deprecated and in maintenance mode. ``transformers-neuronx`` releases will now only address critical security issues. In Neuron Release 2.26, Neuron will end support for transformers-neuronx. Current users of ``transformers-neuronx`` are advised to migrate to :doc:`NeuronX Distributed Inference </libraries/nxd-inference/index>`.

* PyTorch version 2.6 will no longer be supported in a coming release.  Current users of PyTorch 2.6 are advised to upgrade to PyTorch 2.7, which is supported in this release.

* Support for Python 3.9 will end in a coming release. Currently, we support versions of Python up to 3.11. Current users of Python 3.9 are advised to upgrade to Python 3.11, which is supported in this release.

Ending support in 2.25.0
^^^^^^^^^^^^^^^^^^^^^^^^^

*Items listed here are officially no longer supported starting with Neuron 2.25.0.*

* The following tutorials are no longer supported and have been moved the to :doc:`AWS Neuron SDK doc archive </archive/index>`:
  
  * :doc:`/archive/tutorials/finetune_t5`
  * :doc:`/archive/tutorials/ssd300_demo/ssd300_demo`
  * :doc:`/archive/tutorials/megatron_gpt_pretraining`

* Neuron 2.25 is the last release supporting NxDT Megatron Models. Future Neuron releases will not include support for NxDT Megatron Models. Current users of the NxDT Megatron Models are advised to use the Hugging Face model instead by setting the ``CONF_FILE`` variable in the ``train.sh`` file to the config model you want to use.

* With version 2.25.0, Neuron no longer supports vLLM version 0.7.2. Current users of vLLM 0.7.2 are advised to upgrade to vLLM 0.9.1, which is supported in this release.

* Transformers for NeuronX is no longer supported. For more details, see :doc:`the prior announcement </about-neuron/announcements/neuron2.x/announce-intent-maintenance-tnx>`.

Previous releases
-----------------


* :doc:`Neuron 2.24.1 </release-notes/prev/rn.html#2-24-1>`
* :doc:`Neuron 2.24.0 </release-notes/prev/rn.html#2-24>`
* :doc:`Earlier releases </release-notes/prev/rn.html>`

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`
