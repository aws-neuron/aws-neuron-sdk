.. post:: May 21, 2026
    :language: en
    :tags: announce-end-of-support, neuron-runtime, async

.. _announce-eos-implicit-async-mode:

Announcing end-of-support for implicit asynchronous mode in Neuron Runtime
---------------------------------------------------------------------------

A future release of the Neuron SDK will remove support for implicit asynchronous
mode, including the ``NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS`` environment
variable.

Customers using the implicit request async APIs must migrate their code and
calls to the new Neuron Runtime async APIs. For details on the new APIs, see
:doc:`/neuron-runtime/api/nrt-async-api-overview`.
