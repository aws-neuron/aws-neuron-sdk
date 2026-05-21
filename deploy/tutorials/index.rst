.. meta::
   :description: Tutorials for deploying AWS Neuron workloads in containers with Docker and Kubernetes, including environment setup, container builds, and OCI hooks.
   :keywords: Neuron containers, Docker, tutorials, OCI hook, container build, Trainium, Inferentia, deployment
   :date-modified: 04/20/2026

.. _deploy-tutorials:

Container tutorials
====================

Step-by-step tutorials for building and running Neuron containers. These guides cover Docker environment setup, container builds, OCI hook configuration, and framework-specific inference and training examples.

General container tutorials
----------------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Docker environment setup
      :link: /deploy/tutorials/docker-env-setup
      :link-type: doc
      :class-card: sd-border-1

      Configure Docker on Amazon Linux 2023 to expose Inferentia and Trainium devices to containers.

   .. grid-item-card:: Build and run Neuron containers
      :link: /deploy/tutorials/build-run-container
      :link-type: doc
      :class-card: sd-border-1

      Build Docker images with Neuron support and run containerized applications on Inf1 and Trn1 instances.

   .. grid-item-card:: Docker Neuron OCI hook setup
      :link: /deploy/tutorials/oci-hook
      :link-type: doc
      :class-card: sd-border-1

      Install and configure the Neuron OCI hook to expose all Neuron devices to containers using an environment variable.

Inference tutorials
--------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Inference container tutorial
      :link: /deploy/tutorials/inference/tutorial-infer
      :link-type: doc
      :class-card: sd-border-1

      Run inference in a Neuron container with a pre-built DLC image.

   .. grid-item-card:: Deploy ResNet-50 on Kubernetes
      :link: /deploy/tutorials/inference/k8s-rn50-demo
      :link-type: doc
      :class-card: sd-border-1

      Deploy a ResNet-50 inference service on an EKS cluster with Inferentia.

Training tutorials
-------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Training container tutorial
      :link: /deploy/tutorials/training/tutorial-training
      :link-type: doc
      :class-card: sd-border-1

      Run training in a Neuron container with a pre-built DLC image on Trainium.

   .. grid-item-card:: Deploy MLP training on Kubernetes
      :link: /deploy/eks/training
      :link-type: doc
      :class-card: sd-border-1

      Deploy an MLP training job on an EKS cluster with Trainium.

.. toctree::
   :maxdepth: 1
   :hidden:

   Docker Environment Setup </deploy/tutorials/docker-env-setup>
   Build and Run Containers </deploy/tutorials/build-run-container>
   OCI Hook Setup </deploy/tutorials/oci-hook>
   Inference Tutorials </deploy/tutorials/inference/index>
   Training Tutorials </deploy/tutorials/training/index>
