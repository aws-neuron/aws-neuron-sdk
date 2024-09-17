.. _ecs_flow:

Amazon ECS
==========

.. toctree::
    :maxdepth: 1

    /general/devflows/plugins/npd-ecs-flows
    /general/devflows/inference/dlc-then-ecs-devflow
    /general/devflows/training/dlc-then-ecs-devflow

In this section, you'll find resources to help you use Neuron with ECS cluster, deploying inference and training workloads on Inferentia and Trainium ECS clusters.


Using Neuron Node Problem Detector Plugin with ECS
--------------------------------------------------

Neuron node problem detector and recovery plugin enhances resiliency by detecting and remediating errors.
To get started with using Neuron node problem detector plugin and recovery plugin on an ECS cluster, please refer to :ref:`ecs-neuron-problem-detector-and-recovery`.


Running Inference workload
--------------------------

This guide walks you through the end-to-end process of building and running a Docker container with your model and deploying it on an ECS cluster with Inferentia instances.
For running machine learning inference workloads on Amazon ECS using AWS Deep Learning Containers, please refer to :ref:`inference-dlc-then-ecs-devflow`.


Running Training workload
-------------------------

This guide walks you through the end-to-end process of building and running a Docker container with your model and deploying it on an ECS cluster with Trainium instances.
For running machine learning training workloads on Amazon ECS using AWS Deep Learning Containers, please refer to :ref:`training-dlc-then-ecs-devflow`.
