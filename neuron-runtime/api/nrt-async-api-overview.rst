Neuron Runtime Async APIs: Overview
===================================

.. note::

   The Neuron Runtime Async APIs are currently in early release and may change across Neuron versions.

Introduction
------------

Achieving maximum utilization of AWS Neuron Devices requires applications to execute work asynchronously—submitting future execution requests while the device is still processing
previous ones. The Neuron Runtime (NRT) Async APIs provide explicit, fine-grained control over asynchronous operations, enabling developers to fully optimize their workloads for
Neuron hardware.

Neuron Device Execution Units
-----------------------------

The Neuron Runtime exposes the Neuron device as a collection of specialized, independent processing blocks called execution units. Each execution unit can process
operations asynchronously, enabling parallel execution across multiple units.

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

This abstraction along with the Explicit Async APIs, provide applications the control necessary to overlap compute, communication, and data movement operations.

Async Execution Mode vs Async APIs
----------------------------------

Previously, the Neuron Runtime supported an Async Execution Mode which allowed for the asynchronous submission of model/kernel executions. When this mode is enabled, calls to
``nrt_execute`` return immediately, allowing the calling thread to prepare the next execution while the device processes the current one. To maintain tensor consistency, tensor
read/write operations automatically block while tensors are in use by pending executions.

While this flow works, the implicit nature of the implementation limits both the flexibility and control available to applications.

**Limited Flexibility:** The current async model ties execution and data operations together in ways that prevent efficient pipelining. For example, reading tensor
data from the device blocks until all pending executions complete, preventing applications from overlapping data transfers with ongoing Neuron Core computation.

**Limited Control:** The current APIs do not expose asynchronous control for all execution units, limiting applications from making optimal scheduling decisions. Without
fine-grained, asynchronous control over each execution unit, applications cannot implement scheduling strategies that maximize overlap between compute, communication, and
data movement operations.

Explicit Async APIs
-------------------

The Explicit Async APIs directly address the limitations of the implicit async implementation through two core design choices:

* **Explicit completion primitives** — Instead of relying on implicit blocking behavior to ensure consistency, the new APIs provide explicit mechanisms for tracking request
  completion. This gives applications full control over synchronization and enables efficient polling patterns that keep execution units saturated with work.
* **All execution units can run asynchronously** — Unlike the current model where execution and tensor operations are coupled, the new APIs allow the Neuron Core, Collectives,
  and Tensor Copy execution units to operate independently and in parallel. This enables applications to schedule compute, communication, and data movement operations concurrently,
  achieving true overlap between these different types of work.

Together, these design choices give applications the flexibility to implement custom scheduling strategies and the control needed to make optimal decisions about when to overlap work,
when to synchronize, and how to maximize device utilization.

Key Benefits
^^^^^^^^^^^^

* **Higher device utilization** — Pipeline work across multiple devices without idle cycles
* **Compute/communication/data transfer overlap** — Schedule independent operations in parallel
* **Greater optimization flexibility** — Build custom execution strategies tailored to your specific workload

What are the Async APIs
-----------------------

The Explicit Async APIs (prefixed with ``nrta``) are organized into three main categories:

* **Schedule APIs** (``nrta_execute_schedule``, ``nrta_cc_schedule``, etc.) — enqueue work to an execution unit and return a sequence number for tracking
* **Completion APIs** (``nrta_get_sequence``, ``nrta_is_completed``, etc.) — enable applications to monitor execution unit progress and check for request completion
* **Error APIs** (``nrta_error_tracker_get_list``) — allow applications to detect and retrieve errors that occurred during asynchronous execution

Together, these categories enable a workflow where applications continuously submit work, monitor completions, and handle errors—keeping execution units busy and
maximizing device utilization.

See :doc:`nrt_async.h </neuron-runtime/api/nrt_async>` for more details.

Summary
-------

The Neuron Runtime Async APIs give developers explicit control over asynchronous execution on Neuron hardware. Whether you're building advanced inference pipelines or
implementing eager mode workloads that demand responsive kernel scheduling, these APIs unlock optimization opportunities by exposing non-blocking interfaces for all
execution units.
