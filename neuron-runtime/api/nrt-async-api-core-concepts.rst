.. meta::
   :description: How the Neuron Runtime explicit async APIs (nrta_*) queue requests, track completion, and report errors, and what your application must synchronize.
   :date-modified: 2026-10-13

.. _nrt-async-api-core-concepts:

=============================================
Neuron Runtime Async APIs: Core Concepts
=============================================

This page explains how the Neuron Runtime (NRT) explicit async APIs (``nrta_*``, declared in ``nrt_async.h``) behave: how requests are queued, how
completion is tracked, how errors are reported, and what you must synchronize yourself. Read it before writing an application against these APIs.

For motivation and background, see :doc:`nrt-async-api-overview`. For recommended usage patterns, see :doc:`nrt-async-api-best-practices` and
:doc:`nrt-async-api-examples`. For function signatures, see :doc:`nrt_async`.

Summary
=======

* Work is submitted to an **execution unit (XU)** queue. Each XU runs its requests one at a time, in submit order.
* Every accepted request gets a **sequence ID**. You track completion by sequence ID, by polling or with an **eventfd**.
* Scheduling calls do not block. If a queue is full they return ``NRT_QUEUE_FULL`` and change nothing.
* A request's final status is written to the ``ret`` pointer you pass at schedule time, before the request is reported complete.
* Dependencies are either **explicit** or **implicit**. Requests on the same XU queue are implicitly ordered, so no synchronization is needed. For data produced on a different XU or queue, you must explicitly wait for the producer to complete before scheduling its consumer.
* A fatal error permanently *degrades* an XU. Plan for recovery by restarting the application.

Execution units and queues
==========================

Async requests are submitted to an execution unit (XU), an independent processing block of a Neuron Core (see :doc:`nrt-async-api-overview`).
An XU is identified by ``(logical Neuron Core, XU type, queue)``. The ``lnc`` argument of the ``nrta_*`` APIs is the logical Neuron Core index:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - XU
     - ``nrta_xu_t``
     - Queues per logical core
   * - Tensor Op
     - ``NRTA_XU_TENSOR_OP``
     - 2
   * - Compute
     - ``NRTA_XU_COMPUTE``
     - 1
   * - Collectives
     - ``NRTA_XU_COLLECTIVES``
     - 1

Each XU queue has its own sequence number space and health status. Requests on different XUs run in parallel; this is what lets you overlap data
movement, compute and communication.

Sequence IDs and completion behavior
====================================

Every request accepted by a schedule API is assigned a sequence ID, returned through a ``nrta_seq_t`` output argument. A sequence ID is a 64-bit
value. The upper 16 bits identify the XU the request was submitted to, and the lower 48 bits are a counter that is specific to that XU:

.. code-block:: text

    63            48 47                             0
   +----------------+--------------------------------+
   |   XU id (16)   |     sequence number (48)       |
   +----------------+--------------------------------+

Because every XU keeps its own counter, sequence numbers are not unique across XUs on their own; the XU id in the upper bits is what makes the full
64-bit value unique. This also means sequence IDs should only be compared with other sequence IDs from the same XU.

Sequence numbers increase monotonically in submit order. When several threads submit to the same XU, the order is the order in which the runtime
accepted the calls, so an application that needs a specific order across threads must serialize its submissions.

Requests on an XU also complete in the order they were submitted. This is what makes sequence IDs useful for tracking completion: the runtime only needs
to keep track of the most recently completed sequence ID for each XU, and every request with a lower or equal ID on that XU is complete.

Use these APIs to check progress:

* ``nrta_get_sequence()`` returns the most recently completed sequence ID for an XU.
* ``nrta_is_completed(seq, &done)`` reports whether a specific request has completed.
* Completion of sequence N implies completion of every earlier sequence on the same XU, so waiting on the last request of a batch is enough to know the
  whole batch is done.

A request that fails during execution still completes. Its final status is written to the ``NRT_STATUS *ret`` passed at schedule time *before* the request
is reported as completed, so the status is safe to read as soon as completion is observed. Completion therefore means "the runtime is finished with
this request", not "the request succeeded"; check the returned status to tell the two apart.

The following sketch shows the pattern: schedule, wait, then read the status.

.. code-block:: c

   NRT_STATUS status = NRT_SUCCESS;   // must stay valid until the request completes
   nrta_seq_t seq = {};

   NRT_STATUS rc = nrta_execute_schedule(model, inputs, outputs, 0, &status, &seq);
   if (rc != NRT_SUCCESS) {
       // Not accepted (for example NRT_QUEUE_FULL): no sequence was assigned.
   }

   bool done = false;
   while (!done) {
       nrta_is_completed(seq, &done);   // or wait on an eventfd instead of spinning
   }
   if (status != NRT_SUCCESS) {
       // Accepted but failed during execution.
   }

See :doc:`nrt-async-api-examples` for complete examples and :doc:`nrt_async` for exact signatures.

Queue full behavior
===================

Every schedule API returns ``NRT_QUEUE_FULL`` when the target XU cannot accept another request. This is a clean failure: nothing was enqueued, no
sequence was consumed, and the ``ret`` and ``req_sequence`` outputs are not modified. Retry once a request on that XU has completed. Rather than
sleep-looping, wait on an XU completion event (see `Event FD notifications`_) or poll ``nrta_get_sequence()``.

* **Compute XU:** the limit is the number of requests that are scheduled and not yet completed, including the one currently running. The default is 8
  and the maximum is 63; it is set with ``NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS``.
* **Tensor Op XU:** each queue accepts a large number of outstanding requests (64K).
* **Collectives XU:** ``nrta_cc_schedule`` can also return ``NRT_QUEUE_FULL``; handle it the same way as for other XUs.

Request error behavior
======================

Errors are reported in one of two places.

1. **At schedule time, through the return value.** The request was not accepted and no sequence was assigned. Typical statuses: ``NRT_INVALID``
   (NULL arguments, bad core or queue, mismatched tensors), ``NRT_QUEUE_FULL``, ``NRT_RESOURCE``, ``NRT_CLOSED`` (the runtime is closing),
   ``NRT_EXEC_UNIT_UNRECOVERABLE`` (the XU is degraded).
2. **At execution time, through** ``*ret``. The request was accepted and has a sequence. When it completes, its final status is written through the
   ``NRT_STATUS *ret`` given at schedule time. ``ret`` may be NULL, but then a failure is not observable.

Rules to follow:

* **Keep** ``ret`` **valid until the request completes.** The runtime writes ``*ret`` asynchronously while the request is in flight.
* **A failed request does not cancel the ones behind it.** Each queued request completes with its own status.
* **Non-fatal and fatal failures.** Invalid parameters, resource exhaustion and driver queue-full failures are non-fatal: the XU stays usable.
  Timeouts, hardware errors and unrecoverable driver errors are fatal and degrade the XU. Fatal failures are propagated to all other enqueued
  requests.

Data dependencies
=================

Between XUs/XU Queues: managed by the application
-------------------------------------------------

The runtime does not track data dependencies between XU queues, and the schedule calls take no dependency arguments. If a request consumes data produced
by a request on a different XU or queue, wait for the producer's sequence to complete before scheduling the consumer. For example, an
``nrta_tensor_write`` that feeds an ``nrta_execute_schedule`` must be observed complete first. The same holds for execution → read,
collective → execution, and so on. Use ``nrta_is_completed()``, ``nrta_get_sequence()`` or an event fd to wait.

Scheduling a consumer early and hoping the producer finishes first is a race: the request can begin executing as soon as it reaches the head of its queue.

Within one XU queue: ordering is implicit
-----------------------------------------

Requests on the same XU queue run one at a time in submit order, so a consumer submitted after its producer on the same queue needs no explicit wait.

* **Compute:** there is one queue per logical core, so executions that consume each other's outputs need no synchronization.
* **Tensor Op:** reads, writes, copies, casts, compute and FMA requests on the same queue are ordered with each other.

Event FD notifications
======================

Instead of polling, applications can register an eventfd file descriptor that the runtime signals by writing ``1`` to it (as an ``eventfd`` write) when work completes.

.. list-table::
   :header-rows: 1
   :widths: 40 35 25

   * - API
     - Signaled when
     - Lifetime
   * - ``nrta_event_register_seq_id_completion(seq, fd)``
     - The given sequence completes
     - One notification for that sequence
   * - ``nrta_event_register_xu_completion(lnc, xu, queue, fd)``
     - **Any** request on that XU completes
     - Until deregistered

Behavior:

* **The runtime does not own the fd.** It never closes it. The fd must stay valid until it has been signaled or deregistered. An invalid fd is
  rejected with ``NRT_INVALID``.
* **The fd must not block on write.** Use ``eventfd(0, EFD_NONBLOCK)``. A blocking fd whose counter is saturated could stall completion processing for
  the affected core. If a non-blocking write fails with ``EAGAIN``, that notification is lost.
* **Registering replaces.** Registering again for the same sequence, or for the same XU, replaces the previous fd. One fd exists per sequence and per
  XU; fan out in the application if several consumers need the signal.
* **A negative fd deregisters.** Deregistering an already completed sequence succeeds and does nothing.
* **Registering on an already completed sequence signals immediately**, from the calling thread. This makes it safe to register right after
  scheduling, with no window in which a completion can be missed.
* **Sequences that were never issued** return ``NRT_INVALID``.
* **Ordering.** The notification is sent after the completion is visible and after ``*ret`` has been written. On wake-up, ``nrta_is_completed(seq)``
  is true and ``*ret`` holds the final status.
* **XU-level fds count completions.** The XU fd is written once per completed request, so a burst of N completions adds N to the eventfd counter, and a
  single ``read`` returns the total. After waking, call ``nrta_get_sequence()`` to see how far the XU has progressed instead of assuming one completion per wake-up.
* **Degraded XUs still notify.** Requests that complete with an error on a degraded XU still signal their fds.

.. tip::

   Reuse a small set of eventfds rather than creating one per request.

Degraded XUs
============

An XU that hits a fatal error becomes *degraded*. Degradation is permanent for that XU.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - State
     - Meaning
   * - Healthy
     - Normal operation
   * - Closing
     - The runtime is being closed. New requests fail with ``NRT_CLOSED``; requests already queued finish
   * - Degraded
     - A fatal error occurred. New requests fail with ``NRT_EXEC_UNIT_UNRECOVERABLE``

**What degrades an XU**

* Compute: a fatal error while executing a model (timeout, HW error, and SW Queue Overflow). Problems detected before anything reaches the
  hardware (invalid arguments, resource preparation failures) fail only that call and do not degrade the XU.
* Tensor Op: a request that completes with a fatal status (timeout, hardware error, unrecoverable driver error), or a partially posted request.

**What happens once an XU is degraded**

* New schedule calls fail immediately with ``NRT_EXEC_UNIT_UNRECOVERABLE``.
* Requests that were already queued still complete, so their sequences advance, their event fds are signaled and their queue slots are released:

  * *Compute:* they are skipped and complete with ``NRT_EXEC_UNIT_UNRECOVERABLE``. The request that hit the fatal error keeps its original error status.
  * *Tensor Op:* the requests were already handed to the hardware and finish with whatever status the driver reports.

* A degraded XU cannot be recovered without at least re-initializing the runtime, and in the worst cases reloading the driver or rebooting the
  instance. When you see ``NRT_EXEC_UNIT_UNRECOVERABLE``:

  1. drain the remaining in-flight work
  2. release resources
  3. notify upstream services
  4. restart the application and/or reload the driver

**Scope.** Health is tracked per XU. A degraded Tensor Op queue does not by itself mark the other Tensor Op queue or the Compute XU on that core as
degraded, but an underlying hardware fault often causes failures elsewhere soon after, so check every XU's results independently.

Threading
=========

* Any thread may submit to an XU. Sequence order across threads is the order in which calls are accepted; the application is responsible for any
  ordering it needs between threads.
* The runtime completes requests and sends event fd notifications from its own internal threads, not from application threads. Do not assume a
  notification arrives on a particular thread.

Related topics
==============

* :doc:`nrt-async-api-overview`
* :doc:`nrt-async-api-best-practices`
* :doc:`nrt-async-api-examples`
* :doc:`nrt_async`
* :doc:`nrt_status`
