.. meta::
   :description: Neuron infrastructure components for Kubernetes including device plugins, scheduler extensions, monitoring, health detection, and Dynamic Resource Allocation.
   :keywords: Neuron, Kubernetes, device plugin, scheduler, DRA, monitoring, node problem detector, Helm, infrastructure, EKS
   :date-modified: 04/20/2026

.. _deploy-infrastructure:

Neuron infrastructure components
=================================

Neuron provides infrastructure components for managing Neuron hardware in containerized and Kubernetes environments. These components handle device discovery, scheduling, health monitoring, and resource allocation.

Overview
--------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Neuron plugins overview
      :link: /deploy/infrastructure/plugins
      :link-type: doc
      :class-card: sd-border-1

      Overview of all Neuron infrastructure components: device plugin, scheduler extension, node problem detector, monitor, and DRA driver.

Installation
------------

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Neuron Helm chart
      :link: /deploy/eks/helm-chart
      :link-type: doc
      :class-card: sd-border-1

      Install all Neuron infrastructure components with a single Helm command. The recommended installation method for EKS.

Scheduling and device management
----------------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Scheduler extension
      :link: /deploy/infrastructure/scheduler
      :link-type: doc
      :class-card: sd-border-1

      Topology-aware scheduling for optimal Neuron device allocation in Kubernetes.

   .. grid-item-card:: Scheduler flow diagram
      :link: /deploy/infrastructure/scheduler-flow
      :link-type: doc
      :class-card: sd-border-1

      Visual diagram of how the Neuron scheduler extension integrates with Kubernetes components.

Monitoring and health
----------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Neuron monitor
      :link: /deploy/infrastructure/monitoring
      :link-type: doc
      :class-card: sd-border-1

      Collect and expose Neuron device metrics with Prometheus integration for observability and alerting.

   .. grid-item-card:: Node problem detector and recovery
      :link: /deploy/infrastructure/problem-detector
      :link-type: doc
      :class-card: sd-border-1

      Detect hardware failures and trigger automatic node replacement for Neuron devices.

   .. grid-item-card:: NPD permissions (IRSA)
      :link: /deploy/infrastructure/problem-detector-irsa
      :link-type: doc
      :class-card: sd-border-1

      Configure IAM roles for service accounts to grant the node problem detector necessary permissions.

.. toctree::
   :maxdepth: 1
   :hidden:

   Plugins Overview </deploy/infrastructure/plugins>
   Scheduler Extension </deploy/infrastructure/scheduler>
   Scheduler Flow </deploy/infrastructure/scheduler-flow>
   Neuron Monitor </deploy/infrastructure/monitoring>
   Node Problem Detector </deploy/infrastructure/problem-detector>
   NPD Permissions (IRSA) </deploy/infrastructure/problem-detector-irsa>
