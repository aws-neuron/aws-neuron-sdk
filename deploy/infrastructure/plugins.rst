.. _neuron-container-plugins:

Neuron Plugins for Containerized Environments
=============================================

This section provides an overview of the Neuron infrastructure components for containerized environments. For detailed setup instructions, see :ref:`tutorial-k8s-env-setup-for-neuron`.

Neuron Device Plugin
--------------------

Exposes Neuron hardware resources to Kubernetes as schedulable resources (``aws.amazon.com/neuron`` and ``aws.amazon.com/neuroncore``). The device plugin discovers Neuron devices on each node, advertises them to the scheduler, and manages allocation to Pods with exclusive access.

Neuron Scheduler Extension
---------------------------

Provides topology-aware scheduling for optimal Neuron device allocation. It considers device connectivity and placement to ensure efficient utilization. This component is optional and most beneficial for workloads requesting specific subsets of Neuron devices or cores.

Neuron Node Problem Detector and Recovery
------------------------------------------

Monitors Neuron device health and detects hardware and software errors. When unrecoverable issues occur, it can mark nodes as unhealthy and trigger node replacement. It also publishes CloudWatch metrics under the ``NeuronHealthCheck`` namespace for monitoring.

For ECS environments, see :ref:`ecs-neuron-problem-detector-and-recovery`.

Neuron Monitor
--------------

Collects and exposes metrics from Neuron devices including hardware utilization, performance counters, memory usage, and device health. Supports integration with observability platforms like Prometheus for monitoring and alerting.

Neuron Dynamic Resource Allocation (DRA) Driver
-----------------------------------------------

Manages Neuron hardware resources in a Kubernetes environment. It integration with Kubernetes Dynamic Resource Allocation (DRA) framework to advertise Neuron devices and their attributes. This feature cannot be used alongside Neuron device plugin for nodes of the same cluster. For more information on Neuron DRA driver, please refer to :ref:`neuron-dra`

