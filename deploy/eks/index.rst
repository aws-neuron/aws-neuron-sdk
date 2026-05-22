.. meta::
   :description: Deploy Neuron workloads on Amazon EKS with device plugins, Helm charts, Dynamic Resource Allocation, and topology-aware scheduling.
   :keywords: EKS, Kubernetes, Neuron, device plugin, DRA, Helm, scheduling, inference, training, Trainium, Inferentia
   :date-modified: 04/20/2026

.. _deploy-eks:
.. _eks_flow:

Amazon EKS
==========

Deploy containerized Neuron workloads on Amazon Elastic Kubernetes Service. EKS provides managed Kubernetes with Neuron device plugins, topology-aware scheduling, health monitoring, and Dynamic Resource Allocation for Trainium and Inferentia instances.

Get started
-----------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Set up EKS for Neuron
      :link: /deploy/eks/setup
      :link-type: doc
      :class-card: sd-border-1

      Create an EKS cluster with Neuron nodes, install the device plugin, configure the scheduler extension, and verify resource allocation.

   .. grid-item-card:: Neuron Helm chart
      :link: /deploy/eks/helm-chart
      :link-type: doc
      :class-card: sd-border-1

      Install device plugins, scheduler extensions, node problem detector, and DRA driver with a single Helm command.

Run workloads
--------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Run inference on EKS
      :link: /deploy/eks/inference
      :link-type: doc
      :class-card: sd-border-1

      Deploy inference containers on EKS using Neuron Deep Learning Containers on Inferentia and Trainium instances.

   .. grid-item-card:: Run training on EKS
      :link: /deploy/eks/training
      :link-type: doc
      :class-card: sd-border-1

      Deploy distributed training workloads on EKS with Trainium instances and Neuron DLCs.

Advanced topics
----------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Dynamic Resource Allocation (DRA)
      :link: /deploy/eks/dra
      :link-type: doc
      :class-card: sd-border-1

      Use Kubernetes DRA for attribute-based device selection and topology-aware allocation on K8s 1.34+.

   .. grid-item-card:: Neuron UltraServer Operator (Beta)
      :link: /deploy/eks/ultraserver-operator
      :link-type: doc
      :class-card: sd-border-1

      Topology-aware provisioning and lifecycle management of Neuron UltraServer workloads on EKS, built on the Neuron DRA driver.

   .. grid-item-card:: Schedule MPI jobs on UltraServers
      :link: /deploy/eks/ultraserver
      :link-type: doc
      :class-card: sd-border-1

      Run MPI jobs across Trn2 UltraServer nodes in EKS for multi-node inference and training.

   .. grid-item-card:: EKS prerequisites
      :link: /deploy/eks/prerequisites
      :link-type: doc
      :class-card: sd-border-1

      Detailed prerequisites for setting up an EKS cluster with Neuron support.

.. toctree::
   :maxdepth: 1
   :hidden:

   Set up EKS for Neuron </deploy/eks/setup>
   Neuron Helm Chart </deploy/eks/helm-chart>
   Inference on EKS </deploy/eks/inference>
   Training on EKS </deploy/eks/training>
   Dynamic Resource Allocation </deploy/eks/dra>
   Neuron UltraServer Operator </deploy/eks/ultraserver-operator>
   UltraServer MPI Jobs </deploy/eks/ultraserver>
   EKS Prerequisites </deploy/eks/prerequisites>
