.. _neuron-2-27-0-whatsnew:
.. _latest-neuron-release:

.. meta::
   :description: The official release notes for the AWS Neuron SDK, version 2.27.0. Release date: 12/19/2025
   :date-modified: 01/14/2026

Neuron 2.27.0 Component Release Notes
=====================================

.. important:: Neuron 2.27.1 patch release
        **January 14, 2026**
        
        A patch release, Neuron version 2.27.1, is available that includes a fix for an issue with Llama 4 models found in Neuron SDK version 2.27.0. For details, see :doc:`the Neuron SDK v2.27.1 release note </release-notes/2.27.1>`.

----

**On December 19, 2025, AWS Neuron released the 2.27.0 version of the Neuron SDK**. 

This page provides detailed component release notes for the Neuron SDK 2.27.0. For a an overview of the release content, see :ref:`What's New in AWS Neuron <whats-new-2025-12-19-v2_27>`.

**Update for Neuron 2.27.1**: A patch release, Neuron version 2.27.1, is available that includes a fix for an issue with Llama 4 models found in Neuron SDK version 2.27.0. For details, see :doc:`the Neuron SDK v2.27.1 release note </release-notes/2.27.1>`.

Select a card below to review detailed release notes for each component of the Neuron SDK version 2.27.0. These component release notes contain details on specific new and improved features, as well as breaking changes, bug fixes, and known issues for that component area of the Neuron SDK.

.. grid:: 1 
        :gutter: 2

        .. grid-item-card::
                :link: latest-neuron-release-artifacts
                :link-type: ref
                :class-card: sd-border-1
        
                **Neuron 2.27.0 release artifacts**
                ^^^
                The libraries and packages updated in this Neuron release.

.. grid:: 1 1 2 2
        :gutter: 2

        .. grid-item-card:: 
                :link: neuron-2-27-0-pytorch
                :link-type: ref

                **PyTorch support** 2.27.0 release notes
                ^^^
                Neuron features and solutions that support the PyTorch ML framework.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``, ``Trn3``

        .. grid-item-card:: 
                :link: neuron-2-27-0-nxd-inference
                :link-type: ref

                **NxD Inference** 2.27.0 release notes
                ^^^
                Neuron features and tools for LLM and agent ML model inference.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``, ``Trn3``
         
        .. grid-item-card:: 
                :link: neuron-2-27-0-compiler
                :link-type: ref

                **Neuron Compiler** 2.27.0 release notes
                ^^^
                The Neuron compiler for AWS Trainium and Inferentia, and its libraries and tools.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``, ``Trn3``

        .. grid-item-card:: 
                :link: neuron-2-27-0-nki
                :link-type: ref

                **Neuron Kernel Interface (NKI)** 2.27.0 release notes
                ^^^
                Neuron's Python-based programming interface for developing and optimizing Neuron kernels.
                +++
                Supports: ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``, ``Trn3``

        .. grid-item-card::
                :link: neuron-2-27-0-nkilib
                :link-type: ref

                **NKI Library (NKI-Lib)** 2.27.0 release notes
                ^^^
                A collection of pre-optimized Neuron kernels for common model operations.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``, ``Trn3``

        .. grid-item-card:: 
                :link: neuron-2-27-0-runtime
                :link-type: ref

                **Neuron Runtime** 2.27.0 release notes
                ^^^
                The Neuron kernel driver and C++ libraries for AWS Inferentia and Trainium instances.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``, ``Trn3``

        .. grid-item-card:: 
                :link: neuron-2-27-0-tools
                :link-type: ref

                **Neuron Developer Tools** 2.27.0 release notes
                ^^^
                Tools that support end-to-end development for AWS Neuron.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``, ``Trn3``

        .. grid-item-card:: 
                :link: neuron-2-27-0-dlami
                :link-type: ref

                **Neuron Deep Learning AWS Machine Images (DLAMIs)** 2.27.0 release notes
                ^^^
                AWS-specific machine images for building and deploying Neuron-based ML solutions.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``, ``Trn3``
 
        .. grid-item-card:: 
                :link: neuron-2-27-0-dlc
                :link-type: ref

                **Neuron Deep Learning Containers (DLCs)** 2.27.0 release notes
                ^^^
                AWS-specific container definitions for building and deploying Neuron-based ML solutions.
                +++
                Supports: ``Inf1``, ``Inf2``, ``Trn1`` / ``Trn1n``, ``Trn2``, ``Trn3``


NxD Core and NxD Training Updates for 2.27
------------------------------------------

Neuron support for PyTorch 2.9 will be the last to include NeuronX Distributed Training (NxDT), NxD Core training APIs, and PyTorch/XLA for training. Starting with Neuron support for PyTorch 2.10, these components will no longer be supported.

Existing NxDT/NxD Core users should stay on PyTorch 2.9 until ready to migrate to native PyTorch on Neuron (starting PyTorch 2.10). Customers are recommended to use native PyTorch with standard distributed primitives (DTensor, FSDP, DDP) and TorchTitan starting with Neuron 2.28 and PyTorch 2.10. A migration guide will be published in Neuron 2.28.

Software maintenance announcements
----------------------------------

This section signals the official end-of-support or end of support for specific features, tools, and APIs. For the full set of Neuron release announcements, see :doc:`/about-neuron/announcements/index`.

Known issues: Samples
---------------------

* When running the `UNet training sample <https://github.com/aws-neuron/aws-neuron-samples-staging/blob/master/torch-neuronx/training/unet_image_segmentation/unet.ipynb>`_ with the Neuron compiler, you may encounter this error: `Estimated peak HBM usage exceeds 16GB.`
  
  * To work around this error, include the function ``conv_wrap`` in your model. (You can find a usable example of this function in the `UNet sample model code <https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/training/unet_image_segmentation/model.py>`_.) Then, define a custom backward pass for your model following the instructions and example in `the PyTorch documentation <https://docs.pytorch.org/docs/stable/notes/extending.html>`_. The UNet sample also illustrates how this is done for the convolution layers in UNet.

Previous releases
-----------------

* `Neuron 2.26.0 </release-notes/prev/2.26.0/>`_
* `Neuron 2.25.0 </release-notes/prev/2.25.0/>`_
* `Neuron 2.24.0 </release-notes/prev/rn.html#2-24>`_
* `Earlier releases </release-notes/prev/rn.html>`_

