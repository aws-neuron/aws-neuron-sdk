.. meta::
    :description: release notes for Neuron Observer across all AWS Neuron SDK versions.
    :keywords: neuron observer, observability, metrics, monitoring, telemetry, cluster health, nobs, release notes, aws neuron sdk
    :date-modified: 10/19/2026

.. _observer_rn:

Component Release Notes for Neuron Observer
============================================

**Latest version (in 2.33.0)**: Neuron Observer 0.2.0 (Public Beta)

The release notes for **Neuron Observer**, a low-overhead observability tool for AWS
Trainium clusters. Read these notes for the changes, improvements, and bug fixes in each
AWS Neuron SDK release.

.. note::

   Neuron Observer is in **public beta** starting with the Neuron 2.33.0 release. It is
   supported on Trainium 2 (trn2) and Trainium 3 (trn3) instances. APIs, metric
   names, and installation steps can change before general availability. During the
   beta, ``neuron-monitor`` remains fully supported and no migration is required.

----

.. _observer-2-33-0-rn:

Neuron Observer (Neuron 2.33.0 Release)
---------------------------------------

Date of Release: 10/19/2026

Neuron Observer is now available in public beta. It provides low-overhead health and telemetry
signals, detects issues in sub-second time, and pinpoints the affected node or device—from a
single host to a full cluster.

This release includes:

- A per-node daemon that collects Trainium health signals and telemetry locally.
- The ``nobs`` command-line client to check node and device health (for example, ``nobs devices``).
- Device health state (healthy, unhealthy, unknown) with signals localized to the node and device.
- Health signal and telemetry coverage includes hardware errors such as HBM uncorrectable errors (UEs), NeuronLink health, EFA network health, and resource utilization metrics.
- An on-demand health check that runs a small workload on each Logical NeuronCore and verifies the result, so you can validate a node before you place a workload.
- A native Prometheus ``/metrics`` endpoint on the daemon, always available, that a Prometheus or Amazon Managed Service for Prometheus (AMP) scraper reads directly without a collector.
- Optional OpenTelemetry (OTLP) metrics export, off by default, that pushes to an OpenTelemetry collector for delivery to Amazon CloudWatch, AMP, or Prometheus.
- A Helm chart for deployment on Amazon EKS Trainium node groups.

For setup and usage, see the :doc:`Neuron Observer documentation </tools/neuron-observer/index>`.

----
