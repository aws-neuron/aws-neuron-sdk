.. _neuron-whatsnew:

What's New
==========

.. contents:: Table of contents
   :local:
   :depth: 1

.. _latest-neuron-release:

.. _neuron-2.21.0.beta-whatsnew:

Neuron 2.21.0 Beta (12/03/2024)
-------------------------------

.. note::
  This release (Neuron 2.21 Beta) was only tested with Trn2 instances. The next release (Neuron 2.21) will support all instances (Inf1, Inf2, Trn1, and Trn2).

  For access to this release (Neuron 2.21 Beta), please contact your account manager.

This release (Neuron 2.21 beta) introduces support for :ref:`AWS Trainium2 <trainium2-arch>` and :ref:`Trn2 instances <aws-trn2-arch>`, including the trn2.48xlarge instance type and u-trn2 UltraServer. The release showcases Llama 3.1 405B model inference using NxD Inference on a single trn2.48xlarge instance, and FUJI 70B model training using the AXLearn library across eight trn2.48xlarge instances.

:ref:`NxD Inference <nxdi-index>`, a new PyTorch-based library for deploying large language models and multi-modality models, is introduced in this release. It integrates with vLLM and enables PyTorch model onboarding with minimal code changes. The release also adds support for `AXLearn <https://github.com/apple/axlearn>`_ training for JAX models.

The new :ref:`Neuron Profiler 2.0 <neuron-profiler-2-0-guide>` introduced in this release offers system and device-level profiling, timeline annotations, and container integration. The profiler supports distributed workloads and provides trace export capabilities for Perfetto visualization.

The documentation has been updated to include architectural details about :ref:`Trainium2 <trainium2-arch>` and :ref:`NeuronCore-v3 <neuroncores-v3-arch>`, along with specifications and topology information for the trn2.48xlarge instance type and u-trn2 UltraServer.

:ref:`Use Q Developer <amazon-q-dev>` as your Neuron Expert for general technical guidance and to jumpstart your NKI kernel development.

.. note::
  For the latest release that supports Trn1, Inf2 and Inf1 instances, please see :ref:`Neuron Release 2.20.2 <neuron-2.20.0-whatsnew>`


Release Artifacts
-----------------



.. _latest-neuron-release-artifacts:

Previous Releases
-----------------

* :ref:`prev-rn`
* :ref:`pre-release-content`
* :ref:`prev-n1-rn`
