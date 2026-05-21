.. _k8s-neuron-monitor:

Neuron Monitor is a monitoring solution that collects and exposes metrics from Neuron devices and the Neuron runtime. It provides visibility into hardware utilization, performance counters, memory usage, and device health status. The monitor can export metrics in formats compatible with popular observability platforms like Prometheus, enabling integration with existing monitoring and alerting infrastructure. This allows operators to track Neuron device performance, identify bottlenecks, and troubleshoot issues in production environments.

For detailed information about Neuron Monitor, see the `Neuron Monitor User Guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-monitor-user-guide.html>`_.

.. note::

    Neuron Monitor does not currently support environments using the Neuron DRA (Dynamic Resource Allocation) Driver.

Deploy Neuron Monitor DaemonSet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Step 1: Download the Configuration**

Download the Neuron Monitor YAML file: :download:`k8s-neuron-monitor-daemonset.yml </src/k8/k8s-neuron-monitor-daemonset.yml>`

**Step 2: Apply the Configuration**

Apply the Neuron Monitor YAML to create a DaemonSet on the cluster:

.. code:: bash

    kubectl apply -f k8s-neuron-monitor-daemonset.yml

**Step 3: Verify Installation**

Verify that the Neuron Monitor DaemonSet is running:

.. code:: bash

    kubectl get ds neuron-monitor --namespace neuron-monitor

Expected output (example with 2 nodes in cluster):

.. code:: bash

    NAME             DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
    neuron-monitor   2         2         2       2            2           <none>          27h

**Step 4: Get Pod Names**

Retrieve the Neuron Monitor pod names:

.. code:: bash

    kubectl get pods --namespace neuron-monitor

Expected output:

.. code:: bash

    NAME                   READY   STATUS    RESTARTS   AGE
    neuron-monitor-slsxf   1/1     Running   0          17m
    neuron-monitor-wc4f5   1/1     Running   0          17m

**Step 5: Verify Prometheus Endpoint**

Verify that the Prometheus metrics endpoint is available:

.. code:: bash

    kubectl exec neuron-monitor-wc4f5 --namespace neuron-monitor -- wget -q --output-document - http://127.0.0.1:8000

Expected output (sample metrics):

.. code:: bash

    # HELP python_gc_objects_collected_total Objects collected during gc
    # TYPE python_gc_objects_collected_total counter
    python_gc_objects_collected_total{generation="0"} 362.0
    python_gc_objects_collected_total{generation="1"} 0.0
    python_gc_objects_collected_total{generation="2"} 0.0
    # HELP python_gc_objects_uncollectable_total Uncollectable objects found during GC
    # TYPE python_gc_objects_uncollectable_total counter
