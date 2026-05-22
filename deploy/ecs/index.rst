.. meta::
   :description: Deploy Neuron inference and training workloads on Amazon ECS with Deep Learning Containers and node problem detection.
   :keywords: ECS, Neuron, containers, inference, training, Trainium, Inferentia, node problem detector
   :date-modified: 04/20/2026

.. _deploy-ecs:
.. _ecs_flow:

Amazon ECS
==========

Run containerized Neuron workloads on Amazon Elastic Container Service. ECS provides task-based container orchestration for inference and training on Inferentia and Trainium instances, with support for Neuron node problem detection and recovery.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Run inference on ECS
      :link: /deploy/ecs/inference
      :link-type: doc
      :class-card: sd-border-1

      Deploy inference containers on ECS using Neuron Deep Learning Containers on Inferentia instances.

   .. grid-item-card:: Run training on ECS
      :link: /deploy/ecs/training
      :link-type: doc
      :class-card: sd-border-1

      Deploy training containers on ECS using Neuron DLCs on Trainium instances.

   .. grid-item-card:: Node problem detector for ECS
      :link: /deploy/ecs/npd
      :link-type: doc
      :class-card: sd-border-1

      Monitor Neuron device health and automatically remediate issues on ECS clusters.

.. toctree::
   :maxdepth: 1
   :hidden:

   Inference on ECS </deploy/ecs/inference>
   Training on ECS </deploy/ecs/training>
   Node Problem Detector </deploy/ecs/npd>
