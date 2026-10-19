.. meta::
   :description: C code examples for the Neuron Runtime async APIs (nrta_*), including how to schedule execution and wait for completion.
   :date-modified: 2026-10-13

.. _nrt-async-api-examples:

Async API Usage Examples
========================

Schedule
--------

.. code:: c

   NRT_STATUS exec_ret;

   NRT_STATUS ret = nrta_execute_schedule(model, inputs, outputs, 0, &exec_ret, &seq);
   if (ret != NRT_SUCCESS) {
       if (ret == NRT_QUEUE_FULL) {
           // or handle retries if desired
           break;
       }
       // handle other errors
       ...
   } else {
       req_seqs[req] = seq;
   }

Wait for Completion via polling
-------------------------------

Here’s an example for polling for completion:

.. code:: c

   nrta_seq_t last_req_seq;
   nrta_seq_t completed_seq = {};
   while (true) {
       nrta_get_sequence(lnc, NRTA_XU_COMPUTE, &completed_seq);
       if (completed_seq >= last_req_seq) {
           break;
       }
       usleep(1);
   }

In a similar vein, we can also poll with
``nrta_is_completed(last_req_seq, &is_completed)``

.. _nrta-error-handling:

Error Handling
--------------

We need to maintain an array/vector to track the execution return
statuses.

.. code:: c

   static const int NUM_EXECS = 8;
   int lnc = 0;
   NRT_STATUS exec_rets[NUM_EXECS];

   // submit execution requests
   nrta_seq_t req_seqs[NUM_EXECS];
   for (int req = 0; req < NUM_EXECS; req++) {
       nrta_seq_t seq = {};
       NRT_STATUS ret = nrta_execute_schedule(model, inputs, outputs, 0, &exec_rets[req], &seq);
       if (ret != NRT_SUCCESS) {
           if (ret == NRT_QUEUE_FULL) {
               // or handle retries if desired
               break;
           }
           // handle other errors
           ...
       } else {
           req_seqs[req] = seq;
       }
   }

   // check for execution errors
   int error_count = 0;
   for (int req = 0; req < NUM_EXECS; req++) {
       if (exec_rets[req] != NRT_SUCCESS) {
           fprintf(stderr, "Request [%x] completed with error %lu\n", req_seqs[req], exec_rets[req]);
           error_count++;
       }
   }
   if (error_count > 0) {
       ...
   }

Finding Number of Pending Executions
------------------------------------

While this is susceptible to some races, here’s an example of how to
estimate the outstanding requests:

.. code:: c

   nrta_seq_t last_completed = {};
   const int compute_queue = 0; // Compute XU only has 1 queue
   nrta_get_sequence(lnc, NRTA_XU_COMPUTE, compute_queue, &last_completed);

   // sanity check: the two sequence ids should be from the same XU
   assert(NRTA_SEQ_GET_XU_ID(last_submitted) == NRTA_SEQ_GET_XU_ID(last_completed));
   // the sequence id is a monotone and sequential value for each XU
   return last_submitted - last_completed;
