.. meta::
   :description: Best practices for the Neuron Runtime async APIs: choose between sync and async APIs, keep execution queues full, and handle execution errors.
   :date-modified: 2026-10-13

.. _nrt-async-api-best-practices:

##############
Best Practices
##############

Sync vs Async APIs
==================

With the introduction of the explicit async APIs, the Neuron Runtime provides users with a choice between synchronous APIs and asynchronous APIs. Choosing the right approach
depends on your workload requirements and performance goals.

When to Use Synchronous APIs
----------------------------

Synchronous APIs are appropriate when:

* **Prototyping or debugging** — Blocking behavior simplifies reasoning about execution order and makes it easier to isolate issues.
* **Simple, sequential workloads** — If your application processes one request at a time without pipelining, the added complexity of async APIs may not provide meaningful
  benefit.

When to Use Asynchronous APIs
-----------------------------

Asynchronous APIs are recommended when:

* **Maximizing device utilization** — Async APIs allow you to queue future execution requests while the device processes current work, eliminating idle time between operations.
* **Pipelining across Execution Units** — Async APIs enable the overlapping of work between different Execution Units, allowing for customizable pipelining schemes, reducing
  Execution Unit idle time.
* **Overlapping device work with CPU work** — Non-blocking APIs free the CPU to perform other tasks (e.g., preprocessing, request management) while the device processes requests.

Maximizing Device Utilization
=============================

To maximize device utilization, applications should keep execution unit queues saturated with work at all times. Rather than waiting for each request to complete before submitting
the next request, use the schedule APIs to queue multiple requests ahead of execution—this ensures the device always has work ready to execute when the current operation finishes.
Monitor queue depth using completion APIs like ``nrta_get_sequence`` to track how many requests remain in flight, and submit new work as completions occur to maintain a steady pipeline.
Avoid letting the queue drain completely, as this creates idle gaps while the CPU prepares and submits the next request. A good rule of thumb is to keep at least 2-3 requests
queued per execution unit to absorb any variability in CPU scheduling or request preparation time. For workloads that span multiple execution units, submit work to each unit
as soon as the data dependencies are satisfied—this allows compute, communication, and data transfer operations to overlap, further improving overall device utilization.

Handling Execution Errors
=========================

Request Error Handling
----------------------

When using asynchronous APIs, errors may not surface until after the
schedule call returns—the device could encounter a failure mid-execution
while the application continues to submit new work. To detect these
failures, all the schedule APIs accept an ``NRT_STATUS*`` parameter,
that will be populated with the result of the request execution upon
request completion. Applications should track where they allocated this
status, and check this status after each scheduled request to detect
execution failures.

See :doc:`nrt-async-api-examples` for an example.

Execution Unit Unrecoverable
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In rare cases, an execution unit may enter a fatal failure state due to
a non-recoverable error such as a timeout or detectable hardware issue.
Once in this state, the execution unit can no longer process requests —
all subsequent schedule calls will return
``NRT_EXEC_UNIT_UNRECOVERABLE``.

This is a terminal state; the execution unit cannot be restored without
*at least* reinitializing Runtime, most likely by terminating and
relaunching the application. With the worst errors, reloading the driver
or rebooting the machine will be needed. Applications should monitor for
this return code and implement appropriate recovery logic, such as
releasing resources, notifying upstream services, and relaunching their
application.
