.. meta::
   :description: Neuron Observer collects device health, hardware fault, and performance metrics from AWS Trainium clusters.
   :keywords: AWS Neuron, Neuron Observer, Trainium, observability, cluster health, hardware faults, metrics, nobs, EKS
   :date-modified: 10/15/2026
   :content-type-template: _content-types/conceptual-overview.rst

.. _neuron-observer-index:

========================
What is Neuron Observer?
========================

Neuron Observer collects accelerator and system telemetry from AWS Trainium clusters. It reports device health, hardware faults, and performance metrics for each node.

.. note::

   Neuron Observer is in public beta for the Neuron 2.33 release. It is supported on Trainium 2 (Trn2) and Trainium 3 (Trn3) instances only. APIs, metric names, and installation steps can change before general availability. During the beta, ``neuron-monitor`` remains fully supported and no migration is required. For more information about ``neuron-monitor``, see the :ref:`neuron-monitor user guide <neuron-monitor-ug>`.

Applies to
----------

- Cluster health monitoring for training and inference workloads on Trainium.
- Trainium 2 (Trn2) and Trainium 3 (Trn3) instances running on Amazon EKS.

Core capabilities
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Capability
     - Question it answers
     - Example output / detail
   * - Health state
     - Is this device healthy?
     - 3-state classification: ``healthy``, ``unhealthy``, or ``unknown``.
   * - Fault localization
     - Which node and device are affected?
     - Device health and fault events identify the affected node and device.
   * - Performance telemetry
     - What accelerator and system signals are available for investigating performance?
     - Utilization, latency, EFA, ECC, runtime, and device metrics.
   * - Lower-level API access
     - Can I access lower-level signals for custom scheduling, debugging, or research?
     - Snapshot API access to lower-level metrics and signals in addition to the default exported metric set.

How does Neuron Observer work?
------------------------------

Neuron Observer runs a daemon on each Trainium node. The daemon updates device health when it receives driver health events and collects other telemetry at a configurable interval. Prometheus can scrape metrics and health gauges from the daemon, or the daemon can send them through OpenTelemetry. The API exposes fault events.

On a single node, the daemon collects telemetry from the Neuron driver, sysfs, procfs, and runtime data, stores the latest snapshot, and exposes it through the HTTP API, the ``nobs`` CLI, and an OpenTelemetry exporter that pushes to a co-located ADOT collector. The ADOT collector sends the metrics to your observability backends.

The daemon runs in a separate DaemonSet pod. If the daemon stops, the training workload continues, but Observer stops collecting and publishing node signals until the daemon recovers.

How you consume Neuron Observer signals
----------------------------------------

Use Prometheus and Grafana, Amazon Managed Service for Prometheus (AMP), or Amazon CloudWatch to consume Neuron Observer signals. Use the API and CLI to integrate health signals into a scheduler, control plane, or debugging workflow.

Further reading
---------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Get Started
      :link: /tools/neuron-observer/get-started
      :link-type: doc
      :class-card: sd-border-1

      Install Neuron Observer on your Trainium cluster and view your first metrics.

   .. grid-item-card:: Install and Configure
      :link: /tools/neuron-observer/install-configure
      :link-type: doc
      :class-card: sd-border-1

      Install and configure Neuron Observer on Amazon EKS, with export pipeline options.

   .. grid-item-card:: Configuration Reference
      :link: /tools/neuron-observer/configuration-reference
      :link-type: doc
      :class-card: sd-border-1

      Look up the Neuron Observer Helm chart values, their types, and defaults.

   .. grid-item-card:: Metrics and Faults
      :link: /tools/neuron-observer/metrics-fault-reference
      :link-type: doc
      :class-card: sd-border-1

      Reference for the metrics and fault events Neuron Observer reports.

   .. grid-item-card:: Troubleshooting and FAQs
      :link: /tools/neuron-observer/troubleshooting-faq
      :link-type: doc
      :class-card: sd-border-1

      Common issues, FAQs, and public beta limitations.

.. toctree::
   :maxdepth: 1
   :hidden:

   Get Started </tools/neuron-observer/get-started>
   Install and Configure </tools/neuron-observer/install-configure>
   Configuration Reference </tools/neuron-observer/configuration-reference>
   Metrics and Faults </tools/neuron-observer/metrics-fault-reference>
   Troubleshooting and FAQs </tools/neuron-observer/troubleshooting-faq>
