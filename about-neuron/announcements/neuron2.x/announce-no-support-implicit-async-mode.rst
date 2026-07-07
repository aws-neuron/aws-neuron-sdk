.. post:: June 28, 2026
    :language: en
    :tags: announce-no-longer-support, neuron-runtime, async

.. _announce-no-support-implicit-async-mode:

Neuron no longer supports implicit asynchronous mode starting with Neuron 2.31.0
---------------------------------------------------------------------------------

Starting with Neuron SDK 2.31.0, implicit asynchronous execution mode has been removed from the Neuron Runtime. The ``NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS`` environment variable no longer enables asynchronous execution and now emits a deprecation warning.

Customers using the implicit request async APIs must migrate their code to the explicit Neuron Runtime async APIs. For details on the new APIs, see :doc:`/neuron-runtime/api/nrt-async-api-overview`.
