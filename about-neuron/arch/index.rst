.. _neuron-architecture-index:

.. meta::
   :description: Explore the hardware architecture of AWS Neuron instances, including EC2 Trn and Inf instance types, AWS Inferentia and Trainium chips, and NeuronCore processing units. Learn about system specifications, memory hierarchies, interconnect topologies, and architectural considerations for machine learning workloads.
   :date-modified: 2025-10-03

AWS Neuron architecture guides
==============================

Review and understand the hardware architecture of AWS Neuron instances, including AWS Elastic Compute Cloud (EC2) ``Trn`` and ``Inf`` instance types, AWS Inferentia and Trainium chips, and NeuronCore processing units. The documentation covers system specifications, memory hierarchies, interconnect topologies, and architectural considerations for machine learning workloads.

About Neuron Hardware
----------------------

AWS Neuron hardware consists of custom-designed machine learning accelerators optimized for deep learning workloads. This section covers the architecture and capabilities of AWS Inferentia and Trainium chips, their NeuronCore processing units, and the EC2 instances that host them.

Trainium Architecture
----------------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: AWS Trainium3
      :link: neuron-hardware/trainium3
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Third-generation training accelerator chip

   .. grid-item-card:: AWS Trainium2
      :link: neuron-hardware/trainium2
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Second-generation training accelerator chip

   .. grid-item-card:: AWS Trainium
      :link: neuron-hardware/trainium
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      First-generation training accelerator chip

Inferentia Architecture
------------------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: AWS Inferentia2
      :link: neuron-hardware/inferentia2
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Second-generation inference accelerator chip

   .. grid-item-card:: AWS Inferentia
      :link: neuron-hardware/inferentia
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      First-generation inference accelerator chip

NeuronCore Architecture
------------------------

NeuronCores are fully-independent heterogenous compute-units that power Tranium, Tranium2, Inferentia, and Inferentia2 chips.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: NeuronCore v4
      :link: neuron-hardware/neuron-core-v4
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Processing unit architecture for Trainium3

   .. grid-item-card:: NeuronCore v3
      :link: neuron-hardware/neuron-core-v3
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Processing unit architecture for Trainium2

   .. grid-item-card:: NeuronCore v2
      :link: neuron-hardware/neuron-core-v2
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Processing unit architecture for Inferentia2 and Trainium


   .. grid-item-card:: NeuronCore v1
      :link: neuron-hardware/neuron-core-v1
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Processing unit architecture for Inferentia

Neuron AWS EC2 Platform Architecture
-------------------------------------

Overviews of the AWS Inf and Trn instance and UltraServer architectures.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Inf1 Architecture
      :link: neuron-hardware/inf1-arch
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Inf1 instance architecture and specifications

   .. grid-item-card:: Inf2 Architecture
      :link: neuron-hardware/inf2-arch
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Inf2 instance architecture and specifications

   .. grid-item-card:: Trn1 Architecture
      :link: neuron-hardware/trn1-arch
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Trn1 instance architecture and specifications

   .. grid-item-card:: Trn2 Architecture
      :link: neuron-hardware/trn2-arch
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Trn2 instance architecture and specifications

   .. grid-item-card:: Trn3 Architecture
      :link: neuron-hardware/trn3-arch
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Trn3 instance architecture and specifications


.. toctree::
   :maxdepth: 1
   :hidden:

   AWS Inferentia <neuron-hardware/inferentia>
   AWS Inferentia2 <neuron-hardware/inferentia2>
   AWS Trainium <neuron-hardware/trainium>
   AWS Trainium2 <neuron-hardware/trainium2>
   AWS Trainium3 <neuron-hardware/trainium3>
   NeuronCore v1 <neuron-hardware/neuron-core-v1>
   NeuronCore v2 <neuron-hardware/neuron-core-v2>
   NeuronCore v3 <neuron-hardware/neuron-core-v3>
   NeuronCore v4 <neuron-hardware/neuron-core-v4>
   Inf1 Architecture <neuron-hardware/inf1-arch>
   Inf2 Architecture <neuron-hardware/inf2-arch>
   Trn1 Architecture <neuron-hardware/trn1-arch>
   Trn2 Architecture <neuron-hardware/trn2-arch>
