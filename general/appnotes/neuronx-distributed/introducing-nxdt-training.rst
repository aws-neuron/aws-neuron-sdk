.. _introduce-nxd-training:

Introducing NxD Training
===================================================

.. contents:: Table of contents
   :local:
   :depth: 2


What are we introducing?
------------------------

Starting with the Neuron 2.20 release, we are introducing NxD Training. 
In doing so, we are expanding NeuronX Distributed library (previously called NxD that will now be called NxD Core) to 
NxD Training with data science/engineering modules, and end to end examples. NxD Training is a PyTorch based 
distributed training library that enables customers to train large-scale models. Some key distributed strategies 
supported by NxD Training include 3D-parallelism (data parallelism, tensor parallelism and pipeline parallelism) and 
ZeRO-1 (where optimizer states are partitioned across workers). 
 
NxD Training supports model training workflows like pretraining, supervised finetuning (SFT) and parameter efficient 
finetuning (PEFT) using Low-Rank Adapter (LoRA) techniques [#f1]_. For developers, NxD Training offers both API level access 
through NxD Core and PyTorch Lightning and an intuitive interface via YAML based configuration files. NxD Training 
offers a flexible approach that enables customers to leverage only the functionalities that align with their unique 
workflows and seamlessly integrate their machine learning training software at the appropriate level within NxD Training, 
ensuring a user experience tailored to their specific requirements. This is a beta preview version of NxD Training  
and feedback from the developer community is strongly encouraged for upcoming releases.



.. _how-nxd-core-user-affected:

I currently use NeuronX Distributed (NxD Core). How does NxD Training release affect me?
---------------------------------------------------------------------------------------------------------------

Existing NxD Core customers can continue to use NxD Core APIs available under NxD Training. If workflows based on NxD Core 
meet your needs, you do not need to do anything different with NxD Training’s introduction. NxD Core APIs and 
functionalities for NxD Core continue to be available to you as before. You can choose to 
:ref:`install NxD Core only <neuronx_distributed_setup>` and skip all subsequent installation steps for 
NxD Training. However, NxD Training has additional support for YAML based configuration, a model hub and integration with 
PyTorch Lightning. If these capabilities are of interest to you, you may choose to evaluate and start using NxD Training. 

.. _should_nnm_usage_continue:

Should the current Neuron NeMo Megatron (NNM) users continue to use NNM?
------------------------------------------------------------------------------------------------

NxD Training offers same capabilities as Neuron NeMo Megatron (NNM). Additionally, NNM 
will go into maintenance mode in the next release. If you are currently using NNM, the introduction of NxD Training 
toolkit means that you should start evaluating NxD Training for your training needs. With its YAML interface, NxD 
Training is very close in terms of usability to NNM and NeMo. Migrating from NNM to NxD Training  
should involve a relatively minor effort and instructions for doing so are provided 
:ref:`here <nxdt_developer_guide_migration_nnm_nxdt>`.

.. _what_to_use_as_new_user:

I am new to Neuron and have training workloads, what toolkits or libraries should I use?
----------------------------------------------------------------------------------------

If you are starting with Neuron and looking for solutions to your model pretraining or finetuning needs, then NxD Training 
is the recommended toolkit for you. Please start from :ref:`NxD Training page <nxdt>` for overview, 
installation and usage instructions.


Additional Resources
------------------------

Multiple NxD Training resources on getting started, using it and getting required support are listed below. If you encounter issues 
or have product related questions, please refer to FAQs and troubleshooting guides. Additionally, please feel free to reach out to us 
using resources in Support section.

:ref:`How to get started <neuron-quickstart>`

:ref:`Release notes <neuron-2.19.0-whatsnew>`

:ref:`Main section <nxdt>`

:ref:`Troubleshooting <nxdt_known_issues>` 

:ref:`Support <neuron_support>`

.. [#f1] Supported through NxD Core.