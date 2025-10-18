.. _neuron-architecture-index:

.. meta::
   :description: Explore the hardware architecture of AWS Neuron instances, including EC2 Trn and Inf instance types, AWS Inferentia and Trainium chips, and NeuronCore processing units. Learn about system specifications, memory hierarchies, interconnect topologies, and architectural considerations for machine learning workloads.
   :date-modified: 2025-10-03

AWS Neuron architecture guides
==============================

Review and understand the hardware architecture of AWS Neuron instances, including AWS Elastic Compute Cloud (EC2) ``Trn`` and ``Inf`` instance types, AWS Inferentia and Trainium chips, and NeuronCore processing units. The documentation covers system specifications, memory hierarchies, interconnect topologies, and architectural considerations for machine learning workloads.

.. toctree::
      :maxdepth: 1
      :caption: Hardware architecture

      Trn/Inf Instances </about-neuron/arch/neuron-hardware/neuron-instances>
      Amazon EC2 AI Chips </about-neuron/arch/neuron-hardware/neuron-devices>
      NeuronCores </about-neuron/arch/neuron-hardware/neuroncores-arch>

AWS Neuron hardware architecture
--------------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card::
      :link: neuroninstances-arch
      :link-type: ref

      **Trainium and Inferentia architecture overview**
      ^^^
      Architecture overview of EC2 ``Trn`` and ``Inf`` instances with system specifications and connectivity details

   .. grid-item-card::
      :link: neurondevices-arch
      :link-type: ref

      **NeuronDevices architecture**
      ^^^
      Deep dive into AWS Inferentia and Trainium chip architecture and capabilities

   .. grid-item-card::
      :link: neuroncores-arch
      :link-type: ref

      **NeuronCores architecture**
      ^^^
      Detailed NeuronCore architecture including versions, features, and performance characteristics

.. toctree::
   :hidden:
   :maxdepth: 1

   neuron-features/index
   neuron-hardware/index
   glossary