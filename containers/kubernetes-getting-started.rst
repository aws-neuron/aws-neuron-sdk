.. _kubernetes-getting-started:

Using Neuron with Amazon EKS
=============================

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _tutorial-k8s-env-setup-for-neuron:

EKS Setup for Neuron
--------------------

Customers that use Kubernetes can conveniently integrate Inf/Trn instances into their workflows. This section provides step-by-step instructions for setting up an EKS cluster with Neuron support.

Prerequisites
~~~~~~~~~~~~~

.. include:: /containers/tutorials/k8s-prerequisite.rst

Neuron Helm Chart
~~~~~~~~~~~~~~~~~

.. include:: /containers/tutorials/k8s-neuron-helm-chart.rst

.. _k8s-neuron-device-plugin:

Neuron Device Plugin
~~~~~~~~~~~~~~~~~~~~

.. include:: /containers/tutorials/k8s-neuron-device-plugin.rst

.. _neuron_scheduler:

Neuron Scheduler Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: /containers/tutorials/k8s-neuron-scheduler.rst

Neuron Node Problem Detector and Recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: /containers/tutorials/k8s-neuron-problem-detector-and-recovery-irsa.rst

.. include:: /containers/tutorials/k8s-neuron-problem-detector-and-recovery.rst

Neuron Monitor Daemonset
~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: /containers/tutorials/k8s-neuron-monitor.rst
