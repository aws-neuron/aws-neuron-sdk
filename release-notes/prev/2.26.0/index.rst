.. _neuron-2-26-0-whatsnew:

.. meta::
   :description: The official release notes for the AWS Neuron SDK, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0 release notes
===================================

**Date of release**:  September 18, 2025

.. toctree::
   :hidden:
   :maxdepth: 1

   PyTorch support <nx-pytorch>
   JAX support <nx-jax>
   NxD Inference <nxd-inference>
   NxD Core <nxd-core>
   NKI <nki>
   Neuron Runtime <runtime>
   Developer tools <tools>
   Deep Learning AMIs <dlami>
   Deep Learning Containers <containers>

What's new?
-----------

**AWS Neuron SDK 2.26.0** adds support for PyTorch 2.8, JAX 0.6.2, along with support for Python 3.11, and introduces inference improvements on Trainium2 (``Trn2``). This release includes expanded model support, enhanced parallelism features, new Neuron Kernel Interface (NKI) APIs, and improved development tools for optimization and profiling.

Inference Updates
^^^^^^^^^^^^^^^^^

**NxD Inference** - Model support expands with beta releases of Llama 4 Scout and Maverick variants on ``Trn2``. The FLUX.1-dev image generation models are now available in beta on ``Trn2`` instances.

Expert parallelism is now supported in beta, enabling MoE expert distribution across multiple NeuronCores. This release introduces on-device forward pipeline execution in beta and adds sequence parallelism in MoE routers for model deployment flexibility.

.. 
   Sliding Window Attention (SWA) provides performance improvements by attending to recent tokens rather than full context. The feature includes attention sinks support and is automatically enabled for models trained with sliding window attention using the model config ``sliding_window`` attribute.

Neural Kernel Interface (NKI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New APIs enable additional optimization capabilities:

* ``gelu_apprx_sigmoid``: GELU activation with sigmoid approximation
* ``select_reduce``: Selective element copying with maximum reduction
* ``sequence_bounds``: Sequence bounds computation

API enhancements include:

* ``tile_size``: Added total_available_sbuf_size field
* ``dma_transpose``: Added axes parameter for 4D transpose.
* ``activation``: Added ``gelu_apprx_sigmoid`` operation

Developer Tools
^^^^^^^^^^^^^^^

Neuron Profiler improvements include the ability to select multiple semaphores at once to correlate pending activity with semaphore waits and increments. Additionally, system profile grouping now uses a global NeuronCore ID instead of a process local ID for visibility across distributed workloads. The Profiler also adds warnings for dropped events due to limited buffer space.

The ``nccom-test`` utility adds State Buffer support on Trn2 for collective operations, including ``all-reduce``, ``all-gather``, and ``reduce-scatter`` operations. Error reporting provides messages for invalid all-to-all collective sizes to help developers identify and resolve issues.

Deep Learning AMI and Containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Deep Learning AMI now supports PyTorch 2.8 on Amazon Linux 2023 and Ubuntu 22.04. Container updates include PyTorch 2.8.0 and Python 3.11 across all DLCs. The transformers-neuronx environment and package have been removed from PyTorch inference DLAMI/DLC.

.. contents:: In this release
   :local:
   :depth: 2

Component release notes
-----------------------

Select a card below to review detailed release notes for updated components of the Neuron SDK version 2.26.0. These component release notes contain details on specific new and improved features, as well as breaking changes, bug fixes, and known issues for that component area of the Neuron SDK.

.. grid:: 1 1 2 2
        :gutter: 2

        .. grid-item-card:: 
                :link: neuron-2-26-0-pytorch
                :link-type: ref

                **PyTorch support** 2.26.0 release notes
                ^^^
                Neuron features and solutions that support the PyTorch ML framework.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

        .. grid-item-card:: 
                :link: neuron-2-26-0-jax
                :link-type: ref

                **JAX support** 2.26.0 release notes
                ^^^
                Neuron features and solutions that support the JAX ML framework.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

        .. grid-item-card:: 
                :link: neuron-2-26-0-nxd-inference
                :link-type: ref

                **NxD Inference** 2.26.0 release notes
                ^^^
                Neuron features and tools for LLM and agent ML model inference.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``
        
        .. grid-item-card::
                :link: neuron-2-26-0-nxd-core
                :link-type: ref

                **NxD Core** 2.26.0 release notes
                ^^^
                Common features and tools for Neuron-based training and inference.
                +++
                Supports: ``Trn1`` / ``Trn1n``, ``Trn2``

        .. grid-item-card:: 
                :link: neuron-2-26-0-nki
                :link-type: ref

                **Neuron Kernel Interface (NKI)** 2.26.0 release notes
                ^^^
                Neuron's Python-based programming interface for developing and optimizing Neuron kernels.
                +++
                Supports:  ``Inf2``, ``Trn1``/ ``Trn1n``, ``Trn2``

        .. grid-item-card:: 
                :link: neuron-2-26-0-runtime
                :link-type: ref

                **Neuron Runtime** 2.26.0 release notes
                ^^^
                The Neuron kernel driver and C++ libraries for AWS Inferentia and Trainium instances.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

        .. grid-item-card:: 
                :link: neuron-2-26-0-tools
                :link-type: ref

                **Neuron Developer Tools** 2.26.0 release notes
                ^^^
                Tools that support end-to-end development for AWS Neuron.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

        .. grid-item-card:: 
                :link: neuron-2-26-0-dlami
                :link-type: ref

                **Neuron Deep Learning AWS Machine Images (DLAMIs)** 2.26.0 release notes
                ^^^
                AWS-specific machine images for building and deploying Neuron-based ML solutions.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``
 
        .. grid-item-card:: 
                :link: neuron-2-26-0-dlc
                :link-type: ref

                **Neuron Deep Learning Containers (DLCs)** 2.26.0 release notes
                ^^^
                AWS-specific container definitions for building and deploying Neuron-based ML solutions.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``

        .. grid-item-card::
                :link: latest-neuron-release-artifacts
                :link-type: ref
        
                Neuron 2.26.0 release artifacts
                ^^^
                The libraries and packages updated in this release.

Support announcements
---------------------

This section signals the official end-of-support or end of support for specific features, tools, and APIs.

End-of-support announcements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*An "end-of-support (EoS)" announcement is a notification that a feature, tool, or API will not be supported in the future. Plan accordingly!*

* The Neuron Compiler default for the ``--auto-cast`` option will change from ``--auto-cast=matmult`` to ``--auto-cast=none`` in a future release.
* The Beta versions of the :ref:`PyTorch NeuronCore Placement APIs <torch_neuron_core_placement_guide>` are no longer supported with this release.

* Neuron version 2.26.0 is the last release supporting ``parallel_model_trace``. This NxD Inference function will be deprecated in the next version of the Neuron SDK in favor of the ``ModelBuilder.trace()`` method, which provides a more robust and flexible approach for tracing and compiling models for Neuron devices,  enabling more advanced features such as weight layout optimization support, as well as other quality-of-life and stability improvements for SPMD tracing.

  For customers directly invoking ``parallel_model_trace``, they can now use ModelBuilderV2 APIs. For more details on these APIS, see :ref:`nxd-core-model-builder-v2`. For customers that are directly using models in NxDI, there is  no impact since NxDI models are already built on MBv1 which has no issues.

Ending support in 2.26.0
^^^^^^^^^^^^^^^^^^^^^^^^

*" End-of-support" means that AWS Neuron no longer supports the feature, tool, or API indicated in the note as of this release.*

* End-of-support for the Transformers NeuronX library starts with the 2.26.0 release of the AWS Neuron SDK. As a result, the PyTorch inference Deep Learning Container (DLC) will no longer include the ``transformers-neuronx`` package and Neuron no longer provides the ``transformers_neuronx`` virtual environment in both single and multi-framework DLAMIs. For more details, see :ref:`announce-eos-tnx`.
* Starting with Neuron Release 2.26, Neuron driver versions above 2.21 will only support non-Inf1 instances (such as ``Trn1``, ``Inf2``, or other instance types). For ``Inf1`` instance users, Neuron driver versions less than 2.21 will remain supported with regular security patches.
* The Beta versions of the :ref:`PyTorch NeuronCore Placement APIs <torch_neuron_core_placement_guide>` are no longer supported with this release.

Known issues: Samples
^^^^^^^^^^^^^^^^^^^^^

* When running the `UNet training sample <https://github.com/aws-neuron/aws-neuron-samples-staging/blob/master/torch-neuronx/training/unet_image_segmentation/unet.ipynb>`_ with the Neuron compiler, you may encounter this error: `Estimated peak HBM usage exceeds 16GB.`
  
  * To work around this error, include the function ``conv_wrap`` in your model. (You can find a usable example of this function in the `UNet sample model code <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/unet_image_segmentation/model.py>`_.) Then, define a custom backward pass for your model following the instructions and example in `the Pytorch documentation <https://docs.pytorch.org/docs/stable/notes/extending.html>`_. The UNet sample also illustrates how this is done for the convolution layers in UNet.

Previous releases
-----------------

* :doc:`Neuron 2.25.0 </release-notes/prev/2.25.0/>`
* :doc:`Neuron 2.24.0 </release-notes/prev/rn.html#2-24>`
* :doc:`Earlier releases </release-notes/prev/rn.html>`

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`
