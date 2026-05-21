.. meta::
   :description: Choose between Deep Learning AMIs, Deep Learning Containers, and custom Docker images for deploying Neuron workloads on AWS.
   :keywords: DLAMI, DLC, Docker, Neuron, containers, AMI, pre-configured environments, Trainium, Inferentia
   :date-modified: 04/20/2026

.. _deploy-environments:

Pre-configured environments
============================

AWS Neuron provides pre-configured environments so you can start running workloads without manual SDK installation. Choose the option that best fits your deployment model: a Deep Learning AMI for EC2-based development, a Deep Learning Container for orchestrated deployments, or a custom Docker build for full control.

Which environment is right for you?
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 35 15 20 15

   * - Option
     - Best for
     - Setup time
     - Customization
     - Deployment targets
   * - **Deep Learning AMI**
     - EC2 development, quick prototyping, Jupyter notebooks
     - ~5 minutes
     - Pre-configured virtual environments
     - EC2
   * - **Deep Learning Container**
     - Production deployments, orchestrated workloads
     - ~10 minutes
     - Container-based, framework-specific images
     - EKS, ECS, Batch, EC2
   * - **Custom Docker**
     - Full control, CI/CD pipelines, custom dependencies
     - ~30 minutes
     - Complete flexibility
     - Any

Get started
-----------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Deep Learning AMIs
      :link: /deploy/environments/dlami
      :link-type: doc
      :class-card: sd-border-1

      Pre-configured EC2 machine images with Neuron SDK, frameworks, and virtual environments. Available in multi-framework, single-framework, and base variants.

   .. grid-item-card:: Deep Learning Container images
      :link: /deploy/environments/dlc-images
      :link-type: doc
      :class-card: sd-border-1

      Find the right pre-built Docker image for your framework and workload type. Includes PyTorch, JAX, and vLLM inference containers.

   .. grid-item-card:: Customize a Deep Learning Container
      :link: /deploy/environments/customize-dlc
      :link-type: doc
      :class-card: sd-border-1

      Extend a Neuron DLC with additional packages or modify published Dockerfiles to fit your project.

   .. grid-item-card:: Build and run Neuron containers with Docker
      :link: /deploy/environments/docker-setup
      :link-type: doc
      :class-card: sd-border-1

      Install Neuron drivers, configure Docker, and build custom containers from scratch on EC2 instances.

Quickstarts
-----------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Quickstart: Deploy a DLC with vLLM
      :link: /deploy/environments/quickstart-deploy-dlc
      :link-type: doc
      :class-card: sd-border-1

      Configure and deploy a Deep Learning Container with vLLM for inference. ~30 minutes.

   .. grid-item-card:: Quickstart: PyTorch inference with DLC
      :link: /deploy/environments/quickstart-pytorch-inference-dlc
      :link-type: doc
      :class-card: sd-border-1

      Run PyTorch inference using a pre-built Neuron DLC on EC2.

.. toctree::
   :maxdepth: 1
   :hidden:

   Deep Learning AMIs </deploy/environments/dlami>
   DLC Images </deploy/environments/dlc-images>
   Customize DLC </deploy/environments/customize-dlc>
   Docker Setup </deploy/environments/docker-setup>
   Quickstart: Deploy DLC with vLLM </deploy/environments/quickstart-deploy-dlc>
   Quickstart: PyTorch Inference DLC </deploy/environments/quickstart-pytorch-inference-dlc>
