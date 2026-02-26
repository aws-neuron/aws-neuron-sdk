.. _k8s-neuron-problem-detector-and-recovery:

Deploy Neuron Node Problem Detector and Recovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Neuron Node Problem Detector and Recovery is a critical resiliency component that continuously monitors the health of Neuron devices on each Kubernetes node by detecting hardware and software errors such as device failures, driver problems, and runtime errors. It integrates with the Kubernetes Node Problem Detector framework to report Neuron-specific conditions. When unrecoverable issues are detected, it can automatically remediate problems by marking nodes as unhealthy and triggering node replacement to prevent workload scheduling on faulty hardware. The component can also publish CloudWatch metrics under the ``NeuronHealthCheck`` namespace for monitoring and alerting purposes.

**Requirements**

Before deploying the Neuron Node Problem Detector and Recovery, ensure the following requirements are met:

* **Neuron Driver:** Version 2.15 or later
* **Neuron Runtime:** SDK 2.18 or later
* **Prerequisites:** All prerequisites for Kubernetes containers and the Neuron Node Problem Detector must be satisfied

**Installation**

Install the Neuron Node Problem Detector and Recovery as a DaemonSet using Helm:

.. note::

    The installation pulls the container image from the upstream Node Problem Detector repository at ``registry.k8s.io/node-problem-detector``.

.. code:: bash

    helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart

**Enable Node Recovery**

By default, the Neuron Node Problem Detector runs in **monitor-only mode**. To enable automatic node recovery functionality:

.. code:: bash

    helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
        --set "npd.nodeRecovery.enabled=true"

**Verify Installation**

Verify that the Node Problem Detector pods are running:

.. code:: bash

    kubectl get pod -n neuron-healthcheck-system

Expected output (example with 4 nodes in cluster):

.. code:: bash

    NAME                          READY   STATUS    RESTARTS   AGE
    node-problem-detector-7qcrj   1/1     Running   0          59s
    node-problem-detector-j45t5   1/1     Running   0          59s
    node-problem-detector-mr2cl   1/1     Running   0          59s
    node-problem-detector-vpjtk   1/1     Running   0          59s

**Monitoring and Metrics**

When an unrecoverable error occurs, the Neuron Node Problem Detector:

* Publishes metrics to CloudWatch under the ``NeuronHealthCheck`` namespace
* Updates the node's ``NodeCondition``, which can be viewed using:

  .. code:: bash

      kubectl describe node <node-name>
