.. _tutorial-k8s-env-setup-for-neuron:

Kubernetes environment setup for Neuron
=======================================

Introduction
------------

Customers that use Kubernetes can conveniently integrate Inf1/Trn1 instances into their workflows. This tutorial will go through deploying the neuron device plugin daemonset and also how to allocate neuron cores or devices to application pods.

.. dropdown:: Prerequisite
      :class-title: sphinx-design-class-title-small
      :class-body: sphinx-design-class-body-small
      :animate: fade-in

      Before deploying Neuron components, create an EKS cluster and add Neuron-enabled nodes. For the full, step-by-step cluster and node group setup, see :ref:`k8s-prerequisite`.

.. dropdown:: Deploy Neuron Device Plugin
      :class-title: sphinx-design-class-title-small
      :class-body: sphinx-design-class-body-small
      :animate: fade-in

      .. include:: /deploy/includes/k8s-neuron-device-plugin.rst

.. dropdown:: Deploy Neuron Scheduler Extension
      :class-title: sphinx-design-class-title-small
      :class-body: sphinx-design-class-body-small
      :animate: fade-in

      The Neuron scheduler extension provides topology-aware scheduling for workloads that request more than one Neuron core or device. For deployment steps and troubleshooting, see :ref:`neuron-scheduler-extension`.
