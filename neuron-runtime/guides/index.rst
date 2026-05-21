.. meta::
   :description: Neuron Runtime how-to guides covering the libnrt developer workflow, async API migration, and runtime configuration on AWS Trainium and Inferentia.
   :date-modified: 2026-05-11
   :keywords: Neuron, Neuron Runtime, libnrt, NRT, async APIs, runtime configuration

.. _neuron-runtime-guides:

=================================
Neuron Runtime How-To Guides
=================================

Task-focused guides for developers working directly with the Neuron Runtime
(``libnrt``). Use these when you are building a custom framework on top of
the runtime, migrating an existing C/C++ application to the explicit async
APIs, or tuning runtime behavior through environment variables. If you are
using Neuron through PyTorch, JAX, or TensorFlow, the framework handles most
of the runtime interaction for you — see the :ref:`neuron_runtime` overview
for where these guides fit in the larger runtime surface area.

.. toctree::
    :maxdepth: 1
    :hidden:

    Developer guide <nrt-developer-guide>
    Migrate to the explicit async APIs <how-to-migrate-async-apis>
    Configuration guide <configuration-guide>

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: Runtime developer guide
        :link: nrt-api-guide
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Build a C/C++ application against ``libnrt`` directly. Covers the
        runtime architecture, driver and library installation, NEFF loading,
        tensor staging, execution, and the collective communication library
        used for distributed workloads.

    .. grid-item-card:: Migrate to the Explicit Async APIs
        :link: nrt-migrate-to-explicit-async
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Move a C/C++ application off the legacy
        ``NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS`` implicit async mode
        and onto the ``nrta_*`` explicit async APIs. Covers scheduling with
        sequence numbers, polling and event-based completion tracking,
        per-request error handling, and queue backpressure.

    .. grid-item-card:: Runtime Configuration Guide
        :link: nrt-configuration
        :link-type: ref
        :class-header: sd-bg-primary sd-text-white

        Configure the Neuron Runtime through environment variables. Covers
        NeuronCore visibility and allocation, execution timeouts, logging
        verbosity, core dump behavior, and other runtime knobs you set
        before launching your application.

Related topics
--------------

* :ref:`neuron-runtime` — Neuron Runtime overview and landing page
* :ref:`neuron-runtime-explore-home` — deep dives into runtime internals
  (NEFF files, device memory, collectives, core dumps)
* :doc:`/neuron-runtime/api/index` — Neuron Runtime API reference
* :ref:`nrt-troubleshooting` — troubleshooting runtime issues on Inf1 and Trn1
