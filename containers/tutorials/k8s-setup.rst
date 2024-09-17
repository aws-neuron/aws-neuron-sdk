.. _tutorial-k8s-env-setup-for-neuron-to-remove:

Kubernetes environment setup for Neuron
=======================================

Introduction
------------

Customers that use Kubernetes can conveniently integrate Inf1/Trn1 instances into their workflows. This tutorial will go through deploying the neuron device plugin daemonset and also how to allocate neuron cores or devices to application pods.

.. dropdown:: Prerequisite
      :class-title: sphinx-design-class-title-small
      :class-body: sphinx-design-class-body-small
      :animate: fade-in

      .. include:: /containers/tutorials/k8s-prerequisite.rst

.. dropdown:: Deploy Neuron Device Plugin
      :class-title: sphinx-design-class-title-small
      :class-body: sphinx-design-class-body-small
      :animate: fade-in

      .. include:: /containers/tutorials/k8s-neuron-device-plugin.rst

.. dropdown:: Deploy Neuron Scheduler Extension
      :class-title: sphinx-design-class-title-small
      :class-body: sphinx-design-class-body-small
      :animate: fade-in

      .. include:: /containers/tutorials/k8s-neuron-scheduler.rst
