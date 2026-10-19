.. meta::
   :description: Troubleshoot common Neuron Observer installation and export issues on Amazon EKS, and review FAQs and public beta limitations.
   :keywords: Neuron Observer, troubleshooting, FAQ, EKS, beta limitations, CloudWatch, Prometheus, AMP
   :date-modified: 10/15/2026
   :content-type-template: _content-types/reference-troubleshooting.rst

.. _neuron-observer-troubleshooting-faq:

============================================
Troubleshooting and FAQs: Neuron Observer
============================================

Overview
--------

Use this page to troubleshoot Neuron Observer installation and export issues on Amazon EKS. It covers pods that do not start, a daemon that does not become ready, metrics that do not reach your observability backend, and known public beta limitations.

.. note::

   Neuron Observer is in public beta. APIs, metric names, and installation steps can change before general availability.

.. contents:: On this page
   :local:
   :depth: 2

Common issues
-------------

Installation and startup
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Resolution
   * - Pods stuck in ``Pending``
     - - Verify nodes have the ``neuron.amazonaws.com/present: "true"`` label.
       - Confirm the Neuron device plugin is installed and labeling nodes.
   * - Observer pod ``CrashLoopBackOff``
     - Check the current and previous container logs: ``kubectl logs -n neuron-observer <pod>`` and ``kubectl logs -n neuron-observer <pod> --previous``. A missing ``/dev/neuron0`` does not prevent the daemon from starting.
   * - Readiness probe stays 503 (not ready)
     - **Possible causes:**

       - The daemon has not completed its first collection cycle.
       - The collection task stopped before it could update the metric store.

       **Solutions:**

       - Wait for the first collection cycle to complete, then retry.
       - If the readiness failure persists, check the daemon logs for collection-task errors. Readiness depends on completion of the first collection cycle, not on device discovery or driver state.
   * - ``/healthz`` or ``/readyz`` returns an empty body
     - This response is expected. The endpoints return only an HTTP status code with no response body. ``/healthz`` returns 200 while the server is running; ``/readyz`` returns 200 after the first collection cycle and 503 before it. Use ``nobs status``, or run ``kubectl port-forward -n neuron-observer <pod> 7001:7001`` and check the status from another terminal with ``curl -s -o /dev/null -w '%{http_code}' http://localhost:7001/readyz``.

Metrics and export
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Resolution
   * - No metrics at the daemon's ``/metrics`` endpoint
     - - Run ``kubectl port-forward -n neuron-observer <pod> 7001:7001``, then query ``http://localhost:7001/metrics`` from another terminal.
       - Verify the daemon completed its first collection cycle: ``kubectl exec -n neuron-observer <pod> -- nobs status``.
       - Verify ``instance_id`` can be resolved. The daemon resolves it from IMDS, then falls back to ``NODE_NAME`` and ``HOSTNAME``. If none is available, the daemon suppresses metric exports and writes a warning to its log.
   * - CloudWatch: no metrics under the ``NeuronObserver`` namespace
     - - Confirm the Pod Identity association exists: ``aws eks list-pod-identity-associations --cluster-name <c> --namespace neuron-observer --service-account neuron-observer-otel``.
       - Confirm the IAM role has the required CloudWatch Metrics and CloudWatch Logs permissions. Missing permissions appear as ``AccessDenied`` in the collector logs.
       - Confirm ``observer.daemon.config.export.enabled=true`` and ``observer.otelDaemonCollector.pipelines.cloudwatch.enabled=true``.
       - Confirm the region is set: ``observer.otelDaemonCollector.awsRegion=<REGION>``.
       - Check the collector logs: ``kubectl logs -n neuron-observer <collector-pod>``.
   * - AMP: ``403 Forbidden`` in collector logs
     - **Possible causes:**

       - The Pod Identity association is missing or points at a role without ``aps:RemoteWrite`` on the workspace ARN.
       - ``observer.otelDaemonCollector.pipelines.amp.endpoint`` is malformed. The endpoint must end in ``/api/v1/remote_write``.
       - ``observer.otelDaemonCollector.awsRegion`` does not match the AMP workspace region.
   * - Prometheus: no targets discovered
     - - If you use the Prometheus Operator, verify the ``PodMonitor`` exists and its ``release`` label matches your Prometheus selector.
       - If you do not use the Prometheus Operator, verify the additional scrape configuration targets the daemon's port 7001 and ``/metrics`` path.
       - The additional scrape configuration file must be a bare YAML list that starts with ``- job_name:``, not a ``scrape_configs:`` mapping.

Access and image pull
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symptom
     - Resolution
   * - Grafana login fails
     - Retrieve the admin password with ``kubectl --namespace monitoring get secrets prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo``.
   * - Cannot pull image
     - - Run ``kubectl describe pod -n neuron-observer <pod>`` and inspect the image-pull events.
       - Confirm the image is ``public.ecr.aws/neuron/neuron-observer:0.2.0``.
       - Confirm the node can reach Amazon ECR Public. The public image does not require an ECR repository access grant or ECR permissions on the node role.

Diagnostic steps
----------------

Use the ``nobs`` CLI to inspect a node and confirm the daemon's view of the hardware:

.. code-block:: bash

   # Daemon liveness / readiness and driver versions
   kubectl exec -n neuron-observer <pod> -- nobs status

   # Discovered devices and their health verdict
   kubectl exec -n neuron-observer <pod> -- nobs devices

   # Current metrics
   kubectl exec -n neuron-observer <pod> -- nobs metrics --scope device

Frequently asked questions
--------------------------

General
~~~~~~~

**What does Neuron Observer do?**
   Neuron Observer collects accelerator and system telemetry for AWS Trainium clusters. It reports device health and hardware faults for each node.

**Which instances does Neuron Observer support?**
   In the public beta, Neuron Observer is supported on Trainium 2 (Trn2) and Trainium 3 (Trn3) instances running on Amazon EKS.

**What does Neuron Observer not monitor?**
   Neuron Observer monitors accelerator and system health. It does not diagnose application-level issues such as model convergence, data pipeline stalls, or hyperparameter problems.

Neuron Observer and neuron-monitor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Do I need to migrate from neuron-monitor?**
   Not during the public beta. ``neuron-monitor`` remains fully supported, and you can run both tools at the same time. For more information about ``neuron-monitor``, see the :ref:`neuron-monitor user guide <neuron-monitor-ug>`.

**Can I run neuron-monitor and Neuron Observer at the same time?**
   Yes. You can run them at the same time during the public beta.

Deployment and architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Does the Observer daemon affect my training workload?**
   The daemon runs in a separate DaemonSet pod. If the daemon stops, the training workload continues, but Observer stops collecting and publishing node signals until the daemon recovers.

Metrics and export
~~~~~~~~~~~~~~~~~~~

**How do I consume Neuron Observer signals?**
   You can consume Neuron Observer signals through Prometheus and Grafana, Amazon Managed Service for Prometheus (AMP), Amazon CloudWatch, the API, and the ``nobs`` CLI.

**Which export backends does Neuron Observer support?**
   Prometheus scrapes the daemon's native ``/metrics`` endpoint. The daemon pushes metrics through an ADOT collector to Amazon Managed Service for Prometheus (AMP) and Amazon CloudWatch. You can use these destinations together.

**How do I check that a node is healthy before scheduling a workload?**
   In the daemon pod, run the active health check with ``nobs health run``. From outside the pod, run ``kubectl exec -n neuron-observer <pod> -- nobs health run``. The health check runs a matrix multiplication test on each visible Logical NeuronCore (LNC), an all-reduce test over NeuronLink, and an all-reduce test over EFA when an EFA fabric is available. The daemon rejects the request if another process holds any NeuronCore locks.

**What device health states does Neuron Observer report?**
   Neuron Observer reports three device health states: ``healthy``, ``unhealthy``, and ``unknown``. For the full definitions, see :doc:`/tools/neuron-observer/metrics-fault-reference`.

Beta limitations
----------------

- Neuron Observer monitors accelerator and system health. It does not diagnose application-level issues (model convergence, data pipeline stalls, hyperparameter problems).
- ``neuron-monitor`` and Neuron Observer can run at the same time during the public beta.

Related topics
--------------

- :doc:`/tools/neuron-observer/install-configure` — installation and export pipeline options.
- :doc:`/tools/neuron-observer/configuration-reference` — the Helm chart configuration values, their types, and defaults.
- :doc:`/tools/neuron-observer/get-started` — quickstart from installation to first metrics.
- :doc:`/tools/neuron-observer/metrics-fault-reference` — metrics and health signals.
