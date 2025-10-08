.. _eks_flow:

Amazon EKS
==========

.. toctree::
    :maxdepth: 1

    /containers/kubernetes-getting-started
    /devflows/inference/dlc-then-eks-devflow
    /containers/tutorials/training/k8s_mlp_train_demo


In this section, you'll find resources to help you use Neuron with EKS cluster, deploying inference and training workloads on Inferentia and Trainium EKS clusters.


EKS Setup
------------

This guide covers setting up the Neuron device plugin, scheduler extension, node problem detector, and monitoring plugins.
These components enable efficient resource utilization, monitoring, and resilience when using Inferentia and Trainium instances for inference and training workloads on Kubernetes clusters.
To get started with using AWS Neuron and setting up the required plugins on an EKS cluster, please refer to :ref:`tutorial-k8s-env-setup-for-neuron`.


Running Inference workload
--------------------------

This guide walks you through the end-to-end process of building and running a Docker container with your model and deploying it on an EKS cluster with Inferentia instances.
For running machine learning inference workloads on Amazon EKS using AWS Deep Learning Containers, please refer to :ref:`dlc-then-eks-devflow`.


Running Training workload
-------------------------

This guide walks you through the end-to-end process of building and running a Docker container with your model and deploying it on an EKS cluster with Trainium instances.
For running machine learning training workloads on Amazon EKS using AWS Deep Learning Containers, please refer to :ref:`example-deploy-mlp-train-pod`.
