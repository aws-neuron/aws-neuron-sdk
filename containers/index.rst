.. meta::
   :description: AWS Neuron Deep Learning Containers (DLCs) are pre-configured Docker images for training and serving models on AWS Trainium and Inferentia instances with the Neuron SDK.
   :keywords: Neuron Containers, Deep Learning Containers, DLC, Docker, Kubernetes, EKS, ECS, AWS Neuron, Trainium, Inferentia, vLLM, Container Deployment
   :date-modified: 01/22/2026

.. _neuron_containers:

Neuron Containers
=================

This section contains the technical documentation for using AWS Neuron Deep Learning Containers (DLCs) and containerized deployments on Inferentia and Trainium instances.

.. toctree::
    :maxdepth: 1
    :hidden:

    Getting Started </containers/getting-started>
    Locate Neuron DLC Images </containers/locate-neuron-dlc-image>
    Customize DLC </containers/dlc-then-customize-devflow>
    Neuron Plugins </containers/neuron-plugins>
    Tutorials </containers/tutorials>
    How-To Guides </containers/how-to/how-to-ultraserver>
    FAQ </containers/faq>
    DRA </containers/neuron-dra>
    Release Notes </release-notes/components/containers>

What are Neuron Deep Learning Containers?
------------------------------------------

AWS Neuron Deep Learning Containers (DLCs) are a set of pre-configured Docker images for training and serving models on AWS Trainium and Inferentia instances using the AWS Neuron SDK. Each DLC is optimized for specific ML frameworks and comes with all Neuron components pre-installed, enabling you to quickly deploy containerized workloads without manual setup.

With Neuron DLCs, developers can:

* Deploy production-ready containers with pre-installed Neuron SDK and ML frameworks
* Use containers across multiple deployment platforms including EC2, EKS, ECS, and SageMaker
* Customize DLCs to fit specific project requirements
* Leverage Neuron plugins for better observability and fault tolerance
* Run distributed training and inference workloads with vLLM integration
* Schedule MPI jobs on Trn2 UltraServers for improved performance

Neuron DLCs support popular ML frameworks including PyTorch, TensorFlow, and JAX, and are available for both training and inference workloads on Inf1, Inf2, Trn1, Trn1n, and Trn2 instances.

.. admonition:: Neuron DRA for Kubernetes

   Neuron has released support for Dynamic Resource Allocation (DRA) with Kubernetes. :doc:`Read more about it here </containers/neuron-dra>`.

Quickstarts
-----------

.. grid:: 1 1 2 2
    :gutter: 3
    
    .. grid-item-card:: Quickstart: Deploy a DLC with vLLM
        :link: quickstart_vllm_dlc_deploy
        :link-type: ref
        :class-card: sd-rounded-3
        
        Get started by configuring and deploying a Deep Learning Container with vLLM for inference. Time to complete: ~30 minutes.

    .. grid-item-card:: Quickstart: Build a Custom Neuron Container
        :link: containers-getting-started
        :link-type: ref
        :class-card: sd-rounded-3
        
        Learn how to build a custom Neuron container using Docker for training or inference workloads.

Neuron Containers Documentation
--------------------------------

.. grid:: 1 1 2 2
    :gutter: 3
    
    .. grid-item-card:: Getting Started
        :link: containers-getting-started
        :link-type: ref
        :class-card: sd-rounded-3
        
        Step-by-step guide for building Neuron containers using Docker, including driver installation and container setup.

    .. grid-item-card:: Locate Neuron DLC Images
        :link: locate-neuron-dlc-image
        :link-type: ref
        :class-card: sd-rounded-3
        
        Find the right pre-configured Deep Learning Container image for your ML framework and instance type.

    .. grid-item-card:: Customize Neuron DLC
        :link: containers-dlc-then-customize-devflow
        :link-type: ref
        :class-card: sd-rounded-3
        
        Learn how to customize Neuron Deep Learning Containers to fit your specific project requirements.

    .. grid-item-card:: Neuron Plugins
        :link: neuron-container-plugins
        :link-type: ref
        :class-card: sd-rounded-3
        
        Explore Neuron plugins for containerized environments, providing better observability and fault tolerance.

    .. grid-item-card:: Tutorials
        :link: /containers/tutorials
        :link-type: doc
        :class-card: sd-rounded-3
        
        Hands-on tutorials for deploying containers on EC2, EKS, ECS, and other platforms with various configurations.

    .. grid-item-card:: How-To: Schedule MPI Jobs on UltraServers
        :link: containers-how-to-ultraserver
        :link-type: ref
        :class-card: sd-rounded-3
        
        Learn how to schedule MPI jobs to run on Neuron UltraServers in EKS for improved performance.

    .. grid-item-card:: FAQ & Troubleshooting
        :link: container-faq
        :link-type: ref
        :class-card: sd-rounded-3
        
        Frequently asked questions and solutions for common issues with Neuron containers.

    .. grid-item-card:: Neuron Containers Release Notes
        :link: /release-notes/components/containers
        :link-type: doc
        :class-card: sd-rounded-3
        
        Review the latest updates, new DLC images, and improvements in Neuron container releases.
