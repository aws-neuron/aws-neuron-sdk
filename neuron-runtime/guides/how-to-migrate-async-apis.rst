.. meta::
   :description: Migrate from implicit async execution mode (NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS) to the explicit nrta_* async APIs in Neuron Runtime
   :date_updated: 2026-05-15

.. _nrt-migrate-to-explicit-async:

========================================================================
How to migrate from implicit async mode to explicit async APIs
========================================================================

Task overview
-------------

This topic discusses how to migrate from the legacy implicit async execution mode
(``NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS``) to the explicit async APIs (``nrta_*``)
using the AWS Neuron SDK. The explicit async APIs provide fine-grained control over
asynchronous execution, enabling higher device utilization through independent scheduling
of compute, communication, and data transfer operations.

Prerequisites
-------------

- **Neuron SDK version:** Neuron SDK 2.29 or later with explicit async API support.
- **Familiarity with the synchronous NRT APIs:** You should already have a working application using ``nrt_execute``.
- **Header inclusion:** Ensure your project includes ``nrt/nrt_async.h``.

.. note::

   If you are using Neuron exclusively through a framework (PyTorch Neuron, JAX Neuron,
   TensorFlow Neuron, etc.), the only action needed is to **update the framework to a
   version that supports the explicit async APIs**. The framework handles the runtime
   interaction on your behalf, and no application-level code changes are required.

Instructions
------------

**1:** Remove the implicit async environment variable

The legacy implicit async mode is controlled by the environment variable
``NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS``. Remove it from your environment,
launch scripts, and configuration files.

.. code-block:: bash

   # Remove from your environment
   unset NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS

With the explicit async APIs, inflight request depth is managed directly by your
application logic rather than a global environment variable.

**2:** Replace ``nrt_execute`` with ``nrta_execute_schedule``

The implicit mode made ``nrt_execute`` non-blocking. With explicit async, use
``nrta_execute_schedule`` which returns immediately and provides a sequence number
for tracking completion.

.. code-block:: c

   // Before (implicit async): nrt_execute blocks or returns immediately depending on env var
   NRT_STATUS ret = nrt_execute(model, input_set, output_set);

   // After (explicit async): always non-blocking, returns a sequence number
   NRT_STATUS exec_ret;
   nrta_seq_t seq;
   NRT_STATUS ret = nrta_execute_schedule(model, input_set, output_set, 0, &exec_ret, &seq);
   if (ret == NRT_QUEUE_FULL) {
       // Queue is full — wait for completions before retrying
   }

.. note::

   The ``exec_ret`` parameter is populated with the execution result **after** the request
   completes on the device. You must retain this variable until completion is confirmed.

**3:** Add explicit completion tracking

Replace any implicit synchronization (which previously happened inside ``nrt_tensor_read``/``nrt_tensor_write``
or at queue-full boundaries) with explicit completion checks.

**Option A: Polling**

.. code-block:: c

   // Poll until a specific request completes
   bool is_completed = false;
   while (!is_completed) {
       nrta_is_completed(seq, &is_completed);
       usleep(1);
   }

   // Or check the last completed sequence on an execution unit
   nrta_seq_t completed_seq;
   nrta_get_sequence(lnc, NRTA_XU_COMPUTE, 0, &completed_seq);
   if (completed_seq >= last_submitted_seq) {
       // All submitted work is done
   }

**Option B: Event-based (recommended for production)**

.. code-block:: c

   #include <sys/eventfd.h>
   #include <poll.h>

   // Register an eventfd to be signaled on completion
   int efd = eventfd(0, EFD_NONBLOCK);
   nrta_event_register_seq_id_completion(seq, efd);

   // Wait for signal via poll/epoll/select
   struct pollfd pfd = { .fd = efd, .events = POLLIN };
   poll(&pfd, 1, timeout_ms);

   close(efd);

**4:** Implement per-request error handling

In implicit async mode, errors surfaced at the next blocking call. With explicit async,
each scheduled request has its own ``NRT_STATUS*`` that is written upon completion.

.. code-block:: c

   static const int NUM_REQUESTS = 8;
   NRT_STATUS exec_rets[NUM_REQUESTS];
   nrta_seq_t req_seqs[NUM_REQUESTS];

   // Submit multiple requests
   for (int i = 0; i < NUM_REQUESTS; i++) {
       NRT_STATUS ret = nrta_execute_schedule(model, inputs, outputs, 0,
                                              &exec_rets[i], &req_seqs[i]);
       if (ret != NRT_SUCCESS) break;
   }

   // check completion
   ...

   // After completion, check each request's status
   for (int i = 0; i < NUM_REQUESTS; i++) {
       if (exec_rets[i] != NRT_SUCCESS) {
           fprintf(stderr, "Request %d failed: %d\n", i, exec_rets[i]);
       }
   }

.. note::

   If any schedule call returns ``NRT_EXEC_UNIT_UNRECOVERABLE``, the execution unit has
   entered a fatal state. The application must reinitialize the runtime (typically by
   restarting the process).

**5:** Handle queue backpressure (``NRT_QUEUE_FULL``)

The implicit mode used ``NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS`` to cap concurrency.
With explicit async, the runtime signals backpressure by returning ``NRT_QUEUE_FULL`` from
``nrta_execute_schedule`` when the execution unit's queue cannot accept more requests.
When this happens, your application must wait for at least one in-flight request to
complete, then retry the schedule call.

.. code-block:: c

   NRT_STATUS exec_ret;
   nrta_seq_t seq;
   NRT_STATUS ret = nrta_execute_schedule(model, inputs, outputs, 0, &exec_ret, &seq);

   if (ret == NRT_QUEUE_FULL) {
       // Wait for at least one completion before retrying
       nrta_seq_t completed;
       do {
           nrta_get_sequence(lnc, NRTA_XU_COMPUTE, 0, &completed);
       } while (completed < last_known_completed + 1);

       // Retry the schedule
       ret = nrta_execute_schedule(model, inputs, outputs, 0, &exec_ret, &seq);
   }

Confirm your work
-----------------

To confirm you have successfully migrated to the explicit async APIs:

1. Verify the environment variable is no longer set:

.. code-block:: bash

   echo $NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS
   # Should be empty

2. Run your application and confirm that ``nrta_execute_schedule`` returns ``NRT_SUCCESS``
   and sequence numbers increment monotonically.

3. Verify that completion tracking works by checking that ``nrta_is_completed`` returns
   ``true`` after execution finishes, and that per-request ``NRT_STATUS`` values are
   ``NRT_SUCCESS``.

Common issues
-------------

Uh oh! Did you encounter an error or other issue while working through this task? Here are some commonly encountered issues and how to address them.

.. rubric:: NRT_QUEUE_FULL returned from schedule calls

- **Possible solution**: Your application is submitting work faster than the device can process it. Implement backpressure by waiting for completions (via polling or events) before submitting new requests. Reduce your inflight request cap.

.. rubric:: Stale data read from tensors

- **Possible solution**: You are reading tensor data before the execution that produces it has completed. Use ``nrta_is_completed`` or ``nrta_get_sequence`` to confirm the producing execution has finished before scheduling a tensor read.

.. rubric:: NRT_EXEC_UNIT_UNRECOVERABLE

- **Possible solution**: The execution unit has entered a fatal state due to a hardware error or timeout. Terminate and relaunch your application. In severe cases, the Neuron driver may need to be reloaded.

.. rubric:: exec_ret status is not populated

- **Possible solution**: The ``NRT_STATUS*`` passed to schedule calls is only written upon request **completion**, not at schedule time. Ensure you are checking the status only after confirming completion via ``nrta_is_completed``, ``nrta_get_sequence``, or an event notification.

Related information
-------------------

- `Neuron Runtime Async API Overview <nrt_async_api_overview.md>`_ - Motivation and design of the explicit async APIs
- `Async API Best Practices <nrt_async_api_best_practices.md>`_ - Guidelines for maximizing device utilization
- `Async API Usage Examples <nrt_async_api_examples.md>`_ - Code examples for scheduling, polling, events, and error handling
