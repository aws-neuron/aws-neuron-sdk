.. _llm-inference-benchmarking:

LLM Inference Benchmarking guide
================================

This guide gives an overview of the metrics that are tracked for LLM Inference and guidelines in using LLMPerf library
to benchmark for LLM Inference.

.. contents:: Table of contents
   :local:
   :depth: 2


.. _llm_inference_metrics:

LLM Inference metrics
---------------------
Following are the essential metrics for monitoring LLM inference server performance.

.. list-table::
   :widths: 20 70 
   :header-rows: 1
   :align: left
   :class: table-smaller-font-size

   * - Metric
     - Description

   * - Time To First Token (TTFT) 
     - Average time taken for the LLM to process the prompt and output the first output token to the user. This is typically measured in milli seconds.
  
   * - Time per Output Token (TPOT) 
     - Average time taken for LLM to generate an output token for an inference request. This is typically measured in milli seconds. This metric is also referred as Inter Token Latency (ITL) or Per Token Latency(PTL)
  
   * - End-to-End Response Latency
     - Time taken for the LLM to generate the entire response, including all output tokens. This metric is computed as  
       end-to-end latency = (TTFT) + (TPOT) * (Number of output tokens).
 
   * - Output Token Throughput
     - Number of output tokens generated per second by the inference server across all concurrent users and requests.


.. _llm_perf_patch_changes:

Using LLMPerf to benchmark LLM Inference performance
----------------------------------------------------

`LLMPerf <https://github.com/ray-project/llmperf>`_ is an open source library to benchmark LLM Inference performance. However, there are few changes that need to be applied to LLMPerf
to accurately benchmark and reproduce the metrics that are published by Neuron.


All the changes outlined below are provided as a patch file that you can easily download and apply.
We will work in upstreaming these changes to public LLMPerf in the future. 

Using the relevant HF tokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In public LLMPerf, ``hf-internal-testing`` tokenizer is used by default for all the models that can impact accuracy of performance.
Instead, there is a change to pass the tokenizer config of the model from Hugging Face which is being benchmarked for Neuron.

Excluding TTFT in TPOT calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
LLMPerf includes TTFT in Time per Output Token(or Inter Token Latency) calculation. As TPOT and TTFT are two different metrics, a change is done to LLMPerf
to exclude TTFT from TPOT calculation to keep it consistent with how other industry standard performance benchmarks are done.


Following are the instructions to apply the patch to the LLMPerf library.


* Step 1: Get the Neuron git patch file

  Download the ``neuron_perf.patch`` :download:`file </src/benchmark/helper_scripts/neuron_perf.patch>` into the ``llmperf`` directory. 

* Step 2: Apply the git patch

  Run ``git apply neuron_perf.patch``. Confirm changes with ``git diff``.


Benchmarking Data parallel inference with multiple model copies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To measure performance with data parallel inference by using multiple model copies, we need to make additional changes to LLMPerf by applying the following patch:

* Step 1: Get the Neuron git patch file for data parallel inference

  Download the ``llmperf_dp.patch`` :download:`file </src/benchmark/helper_scripts/llmperf_dp.patch>` into the ``llmperf`` directory. 

* Step 2: Apply the git patch

  Run ``git apply llmperf_dp.patch``. Confirm changes with ``git diff``.

This patch enables data parallelism by allowing requests to be distributed across multiple model server endpoints. When multiple addresses are specified in OPENAI_API_BASE (e.g. "http://server1;http://server2;http://server3"), each request will be routed to a different server either randomly or in round-robin fashion, allowing concurrent processing across multiple model servers.
