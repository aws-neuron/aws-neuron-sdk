.. meta::
   :description: Why and when to use the AWS Neuron Runtime explicit async APIs (nrta_*) to keep Neuron device execution units busy.
   :date-modified: 2026-10-13

.. _nrt-async-api-overview:

====================================================
Neuron Runtime Async APIs: Motivation and Overview
====================================================

Introduction
============

Achieving maximum utilization of AWS Neuron Devices requires applications to execute work asynchronously—submitting future execution requests while the device is still processing
previous ones. The Neuron Runtime (NRT) Async APIs provide explicit, fine-grained control over asynchronous operations, enabling developers to fully optimize their workloads for
Neuron hardware.

Neuron Device Execution Units
=============================

The Neuron Runtime exposes the Neuron device as a collection of specialized, independent processing blocks called execution units. Each execution unit can process
operations asynchronously, enabling parallel execution across multiple units.

Currently there are 3 types of Execution Units:

+------------------+-----------------------------------------------------------------------------------+
| Execution Unit   | Purpose                                                                           |
+==================+===================================================================================+
| Neuron Core XU   | Executes compiled models or kernels                                               |
+------------------+-----------------------------------------------------------------------------------+
| Collectives XU   | Runs standalone collective operations (all-gather, reduce-scatter, all-reduce)    |
|                  | outside of a compiled model/kernel                                                |
+------------------+-----------------------------------------------------------------------------------+
| Tensor Op XU     | Transfers data between host and Neuron Devices                                    |
+------------------+-----------------------------------------------------------------------------------+

And each neuron core has multiple execution units of each type *(PENDING
API for getting number of queues)*: 

+------------------+------------+
| Execution Unit   | Queues/NC  |
+==================+============+
| Neuron Core XU   | 1          |
+------------------+------------+
| Collectives XU   | 3          |
+------------------+------------+
| Tensor Op XU     | 2          |
+------------------+------------+

In general, an individual Execution Unit on the device is uniquely
identified by :math:`(NeuronCore\times XUType\times QueueID)`

This abstraction along with the Explicit Async APIs, provide
applications the control necessary to overlap compute, communication,
and data movement operations.

(Legacy) Async Execution Mode vs (New) Async APIs
=================================================

Previously, the Neuron Runtime supported an Async Execution Mode which allowed for the asynchronous submission of model/kernel executions. When this mode is enabled, calls to
``nrt_execute`` return immediately, allowing the calling thread to prepare the next execution while the device processes the current one. To maintain tensor consistency, tensor
read/write operations automatically block while tensors are in use by pending executions.

While this flow works, the implicit nature of the implementation limits both the flexibility and control available to applications.

**Limited Flexibility:** The current async model ties execution and data operations together in ways that prevent efficient pipelining. For example, reading tensor
data from the device blocks until all pending executions complete, preventing applications from overlapping data transfers with ongoing Neuron Core computation.

**Limited Control:** The current APIs do not expose asynchronous control for all execution units, limiting applications from making optimal scheduling decisions. Without
fine-grained, asynchronous control over each execution unit, applications cannot implement scheduling strategies that maximize overlap between compute, communication, and
data movement operations.

Async APIs
==========

The Async APIs directly address the limitations of the implicit async implementation through two core design choices:

* **Explicit completion primitives** — Instead of relying on implicit blocking behavior to ensure consistency, the new APIs provide explicit mechanisms for tracking request
  completion. This gives applications full control over synchronization and enables efficient polling patterns that keep execution units saturated with work.
* **All execution units can run asynchronously** — Unlike the current model where execution and tensor operations are coupled, the new APIs allow the Neuron Core, Collectives,
  and Tensor execution units to operate independently and in parallel. This enables applications to schedule compute, communication, and data movement operations concurrently,
  achieving true overlap between these different types of work.

Together, these design choices give applications the flexibility to implement custom scheduling strategies and the control needed to make optimal decisions about when to overlap work,
when to synchronize, and how to maximize device utilization.

Key Benefits
------------

* **Higher device utilization** — Pipeline work across multiple devices without idle cycles
* **Compute/communication/data transfer overlap** — Schedule independent operations in parallel
* **Greater optimization flexibility** — Build custom execution strategies tailored to your specific workload

What are the Async APIs
=======================

The Explicit Async APIs (prefixed with ``nrta``) are organized into two main categories:

* **Schedule APIs** (``nrta_execute_schedule``, ``nrta_cc_schedule``, ``nrta_tensor_read``, ``nrta_tensor_write``, ``nrta_tensor_copy``) — enqueue work to an execution unit and return a sequence number for tracking.
* **Completion APIs** (``nrta_get_sequence``, ``nrta_is_completed``) — enable applications to monitor execution unit progress and check for request completion.

Together, these categories enable a workflow where applications continuously submit work and monitor completions—keeping execution units busy and
maximizing device utilization.

See `nrt_async.h <https://github.com/aws-neuron/aws-neuron-sdk/blob/master/src/libnrt/include/nrt/nrt_async.h>`_ for more details.

Summary
=======

The Neuron Runtime Async APIs give developers explicit control over asynchronous execution on Neuron hardware. Whether you're building advanced inference pipelines or
implementing eager mode workloads that demand responsive kernel scheduling, these APIs unlock optimization opportunities by exposing non-blocking interfaces for all
execution units.
