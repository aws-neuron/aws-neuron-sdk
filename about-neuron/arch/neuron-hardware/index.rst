.. _neuron-hardware-index:

.. meta::
   :description: About AWS Neuron ML chips, including Trainium and Inferentia.
   :date-modified: 10/03/2025

Neuron Hardware
===============

AWS Neuron hardware consists of custom-designed machine learning accelerators optimized for deep learning workloads. This section covers the architecture and capabilities of AWS Inferentia and Trainium chips, their NeuronCore processing units, and the EC2 instances that host them.


AWS Trainium Hardware
----------------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: AWS Trainium
      :link: trainium
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      First-generation training accelerator chip

   .. grid-item-card:: AWS Trainium2
      :link: trainium2
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Second-generation training accelerator chip

AWS Inferentia Hardware
-------------------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: AWS Inferentia
      :link: inferentia
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      First-generation inference accelerator chip

   .. grid-item-card:: AWS Inferentia2
      :link: inferentia2
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Second-generation inference accelerator chip

NeuronCore Architecture
------------------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: NeuronCore v1
      :link: neuron-core-v1
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Processing unit architecture for Inferentia

   .. grid-item-card:: NeuronCore v2
      :link: neuron-core-v2
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Processing unit architecture for Inferentia2 and Trainium

   .. grid-item-card:: NeuronCore v3
      :link: neuron-core-v3
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Processing unit architecture for Trainium2

   .. grid-item-card:: NeuronCores Architecture
      :link: neuroncores-arch
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Overview of NeuronCore processing units

   .. grid-item-card:: Neuron Devices
      :link: neuron-devices
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Device management and configuration

AWS EC2 Trn/Inf Platform Architecture
--------------------------------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Neuron Instances
      :link: neuron-instances
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      EC2 instance types with Neuron accelerators

   .. grid-item-card:: Inf1 Architecture
      :link: inf1-arch
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Inf1 instance architecture and specifications

   .. grid-item-card:: Inf2 Architecture
      :link: inf2-arch
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Inf2 instance architecture and specifications

   .. grid-item-card:: Trn1 Architecture
      :link: trn1-arch
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Trn1 instance architecture and specifications

   .. grid-item-card:: Trn2 Architecture
      :link: trn2-arch
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Trn2 instance architecture and specifications

.. toctree::
   :maxdepth: 1
   :hidden:

   AWS Inferentia <inferentia>
   AWS Inferentia2 <inferentia2>
   AWS Trainium <trainium>
   AWS Trainium2 <trainium2>
   NeuronCore v1 <neuron-core-v1>
   NeuronCore v2 <neuron-core-v2>
   NeuronCore v3 <neuron-core-v3>
   NeuronCores Architecture <neuroncores-arch>
   Neuron Devices <neuron-devices>
   Neuron Instances <neuron-instances>
   Inf1 Architecture <inf1-arch>
   Inf2 Architecture <inf2-arch>
   Trn1 Architecture <trn1-arch>
   Trn2 Architecture <trn2-arch>