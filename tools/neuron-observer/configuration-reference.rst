.. meta::
   :description: Reference for the Neuron Observer Helm chart configuration values under observer.* — daemon deployment and runtime, Prometheus scraping, Grafana dashboards, AMP, OTel export, and namespace.
   :keywords: Neuron Observer, configuration, reference, Helm, values.yaml, observer, daemon, Prometheus, Grafana, AMP, OTel, namespace, Trainium
   :date-modified: 10/15/2026
   :content-type-template: _content-types/reference-general.rst

.. _neuron-observer-configuration-reference:

=========================================
Reference: Neuron Observer configuration
=========================================

Use this page to look up the type, default, and purpose of Neuron Observer Helm chart values commonly configured during the public beta. All values live under ``observer.*`` in the shared chart's ``values.yaml``, including fields that the chart writes to mounted component configuration files.

.. note::

   Neuron Observer is in public beta. APIs, metric names, and installation steps can change before general availability.

Overview
--------

For installation examples that use the ``observer.*`` values in these tables, see :ref:`neuron-observer-install-configure`.

Daemon image and scheduling
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 12 18 36

   * - Value
     - Type
     - Default
     - Purpose
   * - ``observer.enabled``
     - bool
     - false
     - Set to ``true`` to render any Observer resource.
   * - ``observer.image.repository``
     - string
     - ``public.ecr.aws/neuron/neuron-observer``
     - Container image repository.
   * - ``observer.image.tag``
     - string
     - ``0.2.0``
     - Container image tag.
   * - ``observer.image.pullPolicy``
     - string
     - ``IfNotPresent``
     - Kubernetes image pull policy.
   * - ``observer.daemon.nodeSelector``
     - map
     - ``neuron.amazonaws.com/present: "true"``
     - Schedules the daemon only on nodes labeled as containing Neuron devices.
   * - ``observer.daemon.hostNetwork``
     - bool
     - true
     - Runs the daemon in the host network namespace so it can read EFA/RDMA hardware counters. The chart mounts host ``/proc`` separately; it does not enable ``hostPID``.
   * - ``observer.daemon.tolerations``
     - list
     - ``[{key: aws.amazon.com/neuron, operator: Exists, effect: NoSchedule}]``
     - Toleration for the Neuron taint. Extend it if your nodes have extra taints.

Prometheus scraping
-------------------

The daemon always serves Prometheus metrics from ``/metrics`` on its API port. This native pull endpoint does not require ``observer.daemon.config.export.enabled`` or an OpenTelemetry collector.

.. list-table::
   :header-rows: 1
   :widths: 44 10 14 32

   * - Value
     - Type
     - Default
     - Purpose
   * - ``observer.monitoring.releaseLabel``
     - string
     - ``monitoring``
     - Value of the ``release`` label added to Prometheus Operator resources. It must match the selector used by your Prometheus installation. Set it to an empty string to omit the label.
   * - ``observer.monitoring.additionalLabels``
     - map
     - ``{}``
     - Labels added to all Observer Prometheus Operator resources.
   * - ``observer.monitoring.podMonitor.enabled``
     - bool or null
     - null
     - Controls creation of a ``PodMonitor``. By default, the chart creates one only when the Prometheus Operator CRD is available. Set explicitly to ``true`` or ``false`` to override detection.
   * - ``observer.monitoring.podMonitor.interval``
     - duration
     - ``30s``
     - Prometheus scrape interval.
   * - ``observer.monitoring.podMonitor.scrapeTimeout``
     - duration or null
     - unset
     - Optional Prometheus scrape timeout.
   * - ``observer.monitoring.podMonitor.releaseLabel``
     - string or null
     - unset
     - Overrides ``observer.monitoring.releaseLabel`` for the ``PodMonitor``.
   * - ``observer.monitoring.podMonitor.additionalLabels``
     - map
     - ``{}``
     - Additional labels added only to the ``PodMonitor``.

Grafana dashboards
------------------

Enable the dashboard bundle to create ``ConfigMap`` resources for the cluster overview, node diagnostics, and device health dashboards. A Grafana dashboard sidecar container discovers these resources.

.. list-table::
   :header-rows: 1
   :widths: 44 10 14 32

   * - Value
     - Type
     - Default
     - Purpose
   * - ``observer.monitoring.grafanaDashboards.enabled``
     - bool
     - false
     - Creates the three dashboard ``ConfigMap`` resources when set to ``true``.
   * - ``observer.monitoring.grafanaDashboards.namespace``
     - string
     - ``monitoring``
     - Namespace where the chart creates the dashboard ``ConfigMap`` resources. The namespace must already exist.
   * - ``observer.monitoring.grafanaDashboards.label``
     - string
     - ``grafana_dashboard``
     - Label key used by the Grafana dashboard sidecar to discover the ``ConfigMap`` resources.
   * - ``observer.monitoring.grafanaDashboards.labelValue``
     - string
     - ``"1"``
     - Value assigned to the dashboard discovery label.
   * - ``observer.monitoring.grafanaDashboards.additionalLabels``
     - map
     - ``{}``
     - Additional labels added to each dashboard ``ConfigMap``.
   * - ``observer.monitoring.grafanaDashboards.annotations``
     - map
     - ``{}``
     - Annotations added to each dashboard ``ConfigMap``.

Daemon runtime configuration
----------------------------

The chart mounts these settings at ``/etc/neuron-observer/config.yaml``.

.. list-table::
   :header-rows: 1
   :widths: 40 10 14 36

   * - Value
     - Type
     - Default
     - Purpose
   * - ``observer.daemon.config.collection.interval_secs``
     - int
     - 5
     - Seconds between hardware collection cycles. Must be greater than zero.
   * - ``observer.daemon.config.collection.proc_path``
     - string
     - ``/host-proc``
     - Absolute path to the procfs tree used for host and process metrics. The chart mounts the node's ``/proc`` at this path.
   * - ``observer.daemon.config.collection.sources.device``
     - bool
     - true
     - Reads runtime device statistics and ioctl data through ``/dev/neuronN``.
   * - ``observer.daemon.config.collection.sources.sysfs``
     - bool
     - true
     - Reads Neuron health, device information, and ECC data from sysfs.
   * - ``observer.daemon.config.collection.sources.system``
     - bool
     - true
     - Reads the configured procfs tree for host CPU, memory, and process metrics.
   * - ``observer.daemon.config.collection.sources.imds``
     - bool
     - true
     - Queries IMDS for ``instance_id``, ``instance_type``, and other instance metadata.
   * - ``observer.daemon.config.collection.sources.efa``
     - bool
     - true
     - Reads EFA network counters from ``/sys/class/infiniband``.
   * - ``observer.daemon.config.collection.latency_stream.enabled``
     - bool
     - true
     - Enables background sampling used to produce per-core latency percentile metrics.
   * - ``observer.daemon.config.collection.latency_stream.poll_interval_us``
     - int
     - 50
     - Microseconds between latency-stream samples. Must be greater than zero.
   * - ``observer.daemon.config.emulation.enabled``
     - bool
     - false
     - Replays a built-in snapshot instead of collecting from hardware. Intended for testing and development.
   * - ``observer.daemon.config.export.enabled``
     - bool
     - false
     - Enables OTLP push from the daemon. If ``export.endpoint`` is unset, the exporter uses the standard OpenTelemetry endpoint environment variables or the SDK default. This setting is independent of the native Prometheus ``/metrics`` endpoint.
   * - ``observer.daemon.config.export.interval_secs``
     - int
     - 5
     - Seconds between OTLP exports. Must be greater than zero.
   * - ``observer.daemon.config.export.endpoint``
     - string or null
     - unset
     - Exact OTLP/HTTP metrics endpoint. A non-empty value overrides the standard OpenTelemetry endpoint environment variables and must include ``/v1/metrics``.
   * - ``observer.daemon.config.export.mapping``
     - map or null
     - unset
     - Metric export mapping. Omit this value to use the built-in default, specify ``preset: monitor-compat`` to use the compatibility preset, or specify ``path: <absolute-path>`` to replace the default with a custom mapping file.
   * - ``observer.daemon.config.logging.level``
     - string
     - ``info``
     - Tracing filter directive. Use a level such as ``trace``, ``debug``, ``info``, ``warn``, ``error``, or ``off``, or use a target-specific ``EnvFilter`` directive.
   * - ``observer.daemon.config.logging.format``
     - string
     - ``json``
     - ``json`` or ``text``.
   * - ``observer.daemon.config.daemon.listen_address``
     - string
     - ``0.0.0.0``
     - IPv4 or IPv6 address on which the daemon API listens. Use ``0.0.0.0`` to listen on all IPv4 interfaces or ``::`` to listen on all IPv6 interfaces.
   * - ``observer.daemon.config.daemon.port``
     - int
     - 7001
     - TCP port for the daemon API and native Prometheus metrics. Must be greater than zero.
   * - ``observer.daemon.config.daemon.shutdown_timeout_secs``
     - int
     - 5
     - Grace period for in-flight requests after shutdown begins. Must be greater than zero.
   * - ``observer.daemon.config.events.health_buffer_capacity``
     - int
     - 4096
     - Maximum number of fault events retained in memory for replay to reconnecting clients.
   * - ``observer.daemon.config.events.heartbeat_interval_ms``
     - int
     - 1000
     - Milliseconds between heartbeat frames on live event-stream connections.

OpenTelemetry export
--------------------

When daemon export is enabled, the chart creates a separate daemon-mode OpenTelemetry collector.

These values configure the daemon collector:

.. list-table::
   :header-rows: 1
   :widths: 46 8 12 34

   * - Value
     - Type
     - Default
     - Purpose
   * - ``observer.otelDaemonCollector.pipelines.cloudwatch.enabled``
     - bool
     - false
     - Pushes metrics to Amazon CloudWatch through the EMF exporter.
   * - ``observer.otelDaemonCollector.pipelines.cloudwatch.namespace``
     - string
     - ``NeuronObserver``
     - CloudWatch metric namespace.
   * - ``observer.otelDaemonCollector.pipelines.cloudwatch.logGroupName``
     - string
     - ``/metrics/neuron-observer``
     - CloudWatch Logs log group used by the EMF exporter.
   * - ``observer.otelDaemonCollector.pipelines.cloudwatch.logStreamName``
     - string
     - ``otel-metrics``
     - CloudWatch Logs stream used by the EMF exporter.
   * - ``observer.otelDaemonCollector.pipelines.cloudwatch.storageResolution``
     - int
     - 1
     - CloudWatch metric storage resolution in seconds.
   * - ``observer.otelDaemonCollector.pipelines.cloudwatch.metricDeclarationsOverride``
     - list or null
     - null
     - Replaces the chart's default EMF metric declarations when set.
   * - ``observer.otelDaemonCollector.pipelines.amp.enabled``
     - bool
     - false
     - Pushes metrics to Amazon Managed Service for Prometheus (AMP) through Prometheus remote write.
   * - ``observer.otelDaemonCollector.pipelines.amp.endpoint``
     - string
     - ``""``
     - AMP workspace ``.../api/v1/remote_write`` endpoint. Required when this pipeline is enabled.
   * - ``observer.otelDaemonCollector.pipelines.otlp.enabled``
     - bool
     - false
     - Forwards metrics to another OTLP HTTP endpoint.
   * - ``observer.otelDaemonCollector.pipelines.otlp.endpoint``
     - string
     - ``""``
     - Destination OTLP HTTP endpoint. Required when this pipeline is enabled.
   * - ``observer.otelDaemonCollector.awsRegion``
     - string
     - ``""``
     - AWS Region used by the CloudWatch and AMP exporters. Required when either pipeline is enabled.

Namespace
---------

.. list-table::
   :header-rows: 1
   :widths: 40 10 14 36

   * - Value
     - Type
     - Default
     - Purpose
   * - ``observer.namespace.create``
     - bool
     - false
     - Creates the ``neuron-observer`` namespace when set to ``true``. When this value is ``false`` (the default), create the namespace before installation or use Helm's ``--create-namespace`` option.

Related reference
-----------------

- :doc:`/tools/neuron-observer/metrics-fault-reference` — the metrics and fault categories Neuron Observer reports.

Learn more
----------

- :doc:`/tools/neuron-observer/install-configure` — install and configure Neuron Observer on Amazon EKS.
- :doc:`/tools/neuron-observer/get-started` — a quickstart from installation to first metrics.
- :doc:`/tools/neuron-observer/troubleshooting-faq` — common issues, FAQs, and beta limitations.
