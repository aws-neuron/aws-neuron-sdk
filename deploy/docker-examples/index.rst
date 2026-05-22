.. meta::
   :description: Example Dockerfiles for building Neuron containers for inference and training workloads on Trainium and Inferentia.
   :keywords: Dockerfile, Neuron, Docker, containers, inference, training, examples, Trainium, Inferentia
   :date-modified: 04/20/2026

.. _deploy-docker-examples:

Docker examples
================

This section provides example Dockerfiles and supporting code for building Neuron containers. Use these as starting points for your own container builds, or reference them when customizing a Deep Learning Container.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Inference Dockerfiles
      :link: /deploy/docker-examples/inference/Dockerfile-inference-dlc
      :link-type: doc
      :class-card: sd-border-1

      Dockerfiles for building inference containers, including DLC-based, TorchServe, and TensorFlow Serving examples.

   .. grid-item-card:: Training Dockerfiles
      :link: /deploy/docker-examples/training/Dockerfile-trainium-dlc
      :link-type: doc
      :class-card: sd-border-1

      Dockerfiles for building training containers on Trainium, including a complete MLP training example.

   .. grid-item-card:: Device plugin Dockerfile
      :class-card: sd-border-1

      The ``Dockerfile.device-plugin`` in this directory shows how to build the Neuron device plugin container for Kubernetes.

Related resources
-----------------

- :doc:`/deploy/environments/docker-setup` — Set up Docker on EC2 for Neuron
- :doc:`/deploy/environments/customize-dlc` — Customize a pre-built Deep Learning Container
- :doc:`/deploy/environments/dlc-images` — Find pre-built DLC images in ECR

.. toctree::
   :maxdepth: 1
   :hidden:

   Inference DLC Dockerfile </deploy/docker-examples/inference/Dockerfile-inference-dlc>
   Libmode Dockerfile </deploy/docker-examples/inference/Dockerfile-libmode>
   TF Serving Dockerfile </deploy/docker-examples/inference/Dockerfile-tf-serving>
   TorchServe Config </deploy/docker-examples/inference/config-properties>
   Libmode Entrypoint </deploy/docker-examples/inference/dockerd-libmode-entrypoint>
   TorchServe Neuron </deploy/docker-examples/inference/torchserve-neuron>
   Training DLC Dockerfile </deploy/docker-examples/training/Dockerfile-trainium-dlc>
   MLP Training Example </deploy/docker-examples/training/mlp>
   v1 App+RT Different Containers </deploy/docker-examples/v1/inference/Dockerfile-app-rt-diff>
   v1 App+RT Same Container </deploy/docker-examples/v1/inference/Dockerfile-app-rt-same>
   v1 Neuron Runtime </deploy/docker-examples/v1/inference/Dockerfile-neuron-rtd>
   v1 Torch Neuron </deploy/docker-examples/v1/inference/Dockerfile-torch-neuron>
   v1 Entrypoint </deploy/docker-examples/v1/inference/dockerd-entrypoint-app-rt-same>
