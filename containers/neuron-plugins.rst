.. _neuron-container-plugins:

Neuron Plugins for Containerized Environments
=============================================

This section summarizes various neuron infrastructure artifacts for containerized environments. 

* Neuron Node Problem Detector - This plugin enhances resiliency by detecting and remediating errors. For detailed instructions on running this plugin in EKS environment, please refer to :ref:`tutorial-k8s-env-setup-for-neuron` To leverage this plugin on ECS, please refer to :ref:`ecs-neuron-problem-detector-and-recovery`

* Neuron Device Plugin - The Neuron device plugin manages Neuron hardware resources in a Kubernetes environment. It integrates with the Kubernetes device plugin framework to advertise and manage Neuron resources, making them available for use by Pods. For more information on using Neuron with Kubernetes, please refer to :ref:`tutorial-k8s-env-setup-for-neuron`

* Neuron Scheduler Extension - Neuron scheduler extension is a Kubernetes artifact which helps with optimal allocation of neuron cores. Installating scheduler extension is optional if a workload pod consumes all neuron resources on a node. For more information on using Neuron with Kubernetes, please refer to :ref:`tutorial-k8s-env-setup-for-neuron`

