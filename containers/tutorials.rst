.. meta::
   :description: Comprehensive tutorials for deploying AWS Neuron SDK in containers with Docker and Kubernetes. Learn to build Neuron containers, configure EKS clusters, deploy device plugins, and set up monitoring for Trainium and Inferentia instances.
   :keywords: Neuron containers, Docker, Kubernetes, EKS, Trainium, Inferentia, device plugin, scheduler, monitoring, tutorials, AWS, machine learning

Containers - Tutorials
=======================

Learn how to deploy and manage AWS Neuron workloads in containerized environments. These tutorials cover everything from building Docker containers with Neuron support to deploying production-ready Kubernetes clusters with device plugins, schedulers, and monitoring solutions. Whether you're running inference or training workloads on AWS Trainium or Inferentia instances, these step-by-step guides will help you configure your container infrastructure for optimal performance and reliability.

.. toctree::
    :maxdepth: 1
    :hidden:
    
    Inference </containers/tutorials/inference/index>
    Training </containers/tutorials/training/index>
    /containers/tutorials/tutorial-docker-env-setup
    /containers/tutorials/build-run-neuron-container
    /containers/tutorials/tutorial-oci-hook
    /containers/tutorials/k8s-setup
    /containers/tutorials/k8s-neuron-helm-chart
    /containers/tutorials/k8s-neuron-scheduler-flow
    /containers/tutorials/k8s-neuron-monitor
    /containers/tutorials/k8s-neuron-problem-detector-and-recovery
    /containers/tutorials/k8s-neuron-problem-detector-and-recovery-irsa


General Container Tutorials
----------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Docker Environment Setup
      :link: /containers/tutorials/tutorial-docker-env-setup
      :link-type: doc

      Configure Docker on Amazon Linux 2023 to expose Inferentia and Trainium devices to containers. Install Neuron drivers, runtime, and configure the Docker daemon for Neuron device access.

   .. grid-item-card:: Build and Run Neuron Containers
      :link: /containers/tutorials/build-run-neuron-container
      :link-type: doc

      Learn how to build Docker images with Neuron support using provided Dockerfiles and run containerized applications on Inf1 and Trn1 instances with proper device exposure.

   .. grid-item-card:: Docker Neuron OCI Hook Setup
      :link: /containers/tutorials/tutorial-oci-hook
      :link-type: doc

      Install and configure the Neuron OCI hook to enable the AWS_NEURON_VISIBLE_DEVICES environment variable for exposing all Neuron devices to containers without explicit device flags.

Kubernetes Setup and Configuration
-----------------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Kubernetes Environment Setup
      :link: /containers/tutorials/k8s-setup
      :link-type: doc

      Complete guide to setting up Kubernetes for Neuron, including EKS cluster creation with Trainium nodes, device plugin installation, scheduler extension setup, and resource allocation configuration.

   .. grid-item-card:: Neuron Helm Chart
      :link: /containers/tutorials/k8s-neuron-helm-chart
      :link-type: doc

      Simplify Neuron infrastructure deployment with the unified Helm chart that installs device plugins, scheduler extensions, node problem detector, and DRA driver in a single command.

Kubernetes Device Management
-----------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Scheduler Flow Diagram
      :link: /containers/tutorials/k8s-neuron-scheduler-flow
      :link-type: doc

      Visual diagram showing how the Neuron Scheduler Extension integrates with Kubernetes components to schedule Pods with Neuron resource requests.

Kubernetes Monitoring and Recovery
-----------------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Neuron Monitor
      :link: /containers/tutorials/k8s-neuron-monitor
      :link-type: doc

      Deploy Neuron Monitor to collect and expose metrics from Neuron devices and runtime. Integrate with Prometheus for observability, performance tracking, and troubleshooting.

   .. grid-item-card:: Node Problem Detector and Recovery
      :link: /containers/tutorials/k8s-neuron-problem-detector-and-recovery
      :link-type: doc

      Monitor Neuron device health and automatically remediate issues by detecting hardware failures, driver problems, and runtime errors. Enable automatic node replacement for faulty hardware.

   .. grid-item-card:: NPD Permissions (IRSA)
      :link: /containers/tutorials/k8s-neuron-problem-detector-and-recovery-irsa
      :link-type: doc

      Configure IAM roles for service accounts (IRSA) to grant the Neuron Node Problem Detector necessary permissions for Auto Scaling group operations and CloudWatch metrics.


Training and Inference Container Tutorials
------------------------------------------    
.. tab-set:: 

    .. tab-item:: Training

        .. include:: /containers/tutorials/training/index.txt

.. tab-set:: 

    .. tab-item:: Inference
    
         .. include:: /containers/tutorials/inference/index.txt