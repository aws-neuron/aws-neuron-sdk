.. _k8s-neuron-helm-chart:

The Neuron Helm Chart simplifies the deployment and management of Neuron infrastructure components on Kubernetes clusters. It provides a unified installation method for all essential Neuron components, streamlining the setup process and ensuring consistent configuration across your cluster.

Components Included
^^^^^^^^^^^^^^^^^^^

The Neuron Helm Chart includes the following components:

* Neuron Device Plugin
* Neuron Scheduler Extension
* :ref:`Neuron Node Problem Detector and Recovery <k8s-neuron-problem-detector-and-recovery>`
* Neuron DRA (Dynamic Resource Allocation) Driver. Refer to :ref:`neuron-dra`.

Installation
^^^^^^^^^^^^

To install the Neuron Helm Chart:

.. code:: bash

    helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart

For detailed information on configuration options, advanced deployment scenarios, and troubleshooting, please refer to the official Neuron Helm Charts repository: https://github.com/aws-neuron/neuron-helm-charts/
