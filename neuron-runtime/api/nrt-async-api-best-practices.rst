==============================
NRT Async APIs: Best Practices
==============================

.. note::

   The Neuron Runtime Async APIs are currently in early release and may change across Neuron versions.

.. contents:: Table of Contents
   :local:
   :depth: 3

Sync vs Async APIs
------------------

With the introduction of the explicit async APIs, the Neuron Runtime provides users with a choice between synchronous APIs and asynchronous APIs. Choosing the right approach
depends on your workload requirements and performance goals.

When to Use Synchronous APIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Synchronous APIs are appropriate when:

* **Prototyping or debugging** — Blocking behavior simplifies reasoning about execution order and makes it easier to isolate issues.
* **Simple, sequential workloads** — If your application processes one request at a time without pipelining, the added complexity of async APIs may not provide meaningful
  benefit.

When to Use Asynchronous APIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Asynchronous APIs are recommended when:

* **Maximizing device utilization** — Async APIs allow you to queue future execution requests while the device processes current work, eliminating idle time between operations.
* **Pipelining across Execution Units** — Async APIs enable the overlapping of work between different Execution Units, allowing for customizable pipelining schemes, reducing
  Execution Unit idle time.
* **Overlapping device work with CPU work** — Non-blocking APIs free the CPU to perform other tasks (e.g., preprocessing, request management) while the device processes requests.

Maximizing Device Utilization
-----------------------------

To maximize device utilization, applications should keep execution unit queues saturated with work at all times. Rather than waiting for each request to complete before submitting
the next request, use the schedule APIs to queue multiple requests ahead of execution—this ensures the device always has work ready to execute when the current operation finishes.
Monitor queue depth using completion APIs like ``nrta_get_sequence`` to track how many requests remain in flight, and submit new work as completions occur to maintain a steady pipeline.
Avoid letting the queue drain completely, as this creates idle gaps while the CPU prepares and submits the next request. A good rule of thumb is to keep at least 2-3 requests
queued per execution unit to absorb any variability in CPU scheduling or request preparation time. For workloads that span multiple execution units, submit work to each unit
as soon as the data dependencies are satisfied—this allows compute, communication, and data transfer operations to overlap, further improving overall device utilization.

Handling Execution Errors
-------------------------

Request Error Handling
^^^^^^^^^^^^^^^^^^^^^^

When using asynchronous APIs, errors may not surface until after the schedule call returns—the device could encounter a failure mid-execution while the application continues to submit
new work. To detect these failures, the runtime provides the ``nrta_error_tracker_t`` data structure for tracking errors during asynchronous execution.

The error tracking workflow is as follows:

1. **Create an error tracker** — Call ``nrta_error_tracker_create`` to allocate an error tracker for a specific logical Neuron Core.
2. **Pass the tracker to schedule calls** — When submitting requests via schedule APIs, pass the error tracker as a parameter. The runtime will record any errors that occur during
   processing of those requests.
3. **Check for errors after completion** — After confirming requests have completed, call ``nrta_error_tracker_get_list`` to retrieve a list of errors along with the sequence numbers
   of the failed requests.

A recommended pattern is to check for errors after each batch of completions: first call a completion API like ``nrta_get_sequence`` to determine which requests have finished, then call
``nrta_error_tracker_get_list`` to identify any failures. This approach ensures errors are detected promptly while maintaining the performance benefits of asynchronous execution.

**Example**

.. code-block:: c

    int lnc = 0;
    nrta_error_tracker_t *error_tracker = NULL;

    // create error tracker
    nrta_error_tracker_create(lnc, &error_tracker);

    // submit execution requests
    nrta_seq_t req_seq = {};
    for (int req = 0; req < 8; req++) {
        nrta_seq_t seq = {};
        NRT_STATUS ret = nrta_execute_schedule(model, inputs, outputs, 0, error_tracker, &seq);
        if (ret != NRT_SUCCESS) {
            if (ret == NRT_QUEUE_FULL) {
                break;
            }
            // handle other errors
            ...
        } else {
            req_seq = seq;
        }
    }

    // wait for completion
    nrta_seq_t completed_seq = {};
    while (true) {
        nrta_get_sequence(lnc, NRTA_XU_COMPUTE, &completed_seq);
        if (completed_seq >= req_seq) {
            break;
        }
        usleep(1);
    }

    // check for execution errors
    const nrta_error_t *err_list = NULL;
    size_t err_count = 0;
    nrta_error_tracker_get_list(error_tracker, &err_list, &err_count);
    for (int err_idx = 0; err_idx < err_count; err_idx++) {
        fprintf(stderr, "Request [%x] completed with error %lu\n", 
                err_list[err_idx].seq_id, err_list[err_idx].error_code);
    }

    // cleanup
    nrta_error_tracker_destroy(error_tracker);

Execution Unit Degradation
^^^^^^^^^^^^^^^^^^^^^^^^^^

In rare cases, an execution unit may enter a degraded state due to a non-recoverable error such as a timeout or detectable hardware issue. Once degraded, the execution unit can no longer
process requests—all subsequent schedule calls will return ``NRT_EXEC_UNIT_UNRECOVERABLE``. This is a terminal state; the execution unit cannot be restored without terminating and relaunching
the application or rebooting the machine. Applications should monitor for this return code and implement appropriate recovery logic, such as releasing resources, notifying upstream services,
and relaunching their application.
