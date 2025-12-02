.. _llm-inference-benchmarking:

LLM Inference benchmarking guide
================================

This guide gives an overview of the metrics that are tracked for LLM Inference and guidelines in using LLMPerf library
to benchmark for LLM Inference.

.. contents:: Table of contents
   :local:
   :depth: 2


.. _llm_inference_metrics:

LLM Inference metrics
---------------------
Following are the essential metrics for monitoring LLM Inference server performance.

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


All the changes outlined below are provided as a patch file. 

.. note::

  Patches need to be applied in order because they might modify the same files.

Step 1: Install LLMPerf from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    python3 -m venv llmperf-env
    source llmperf-env/bin/activate

    git clone https://github.com/ray-project/llmperf.git ~/llmperf
    cd ~/llmperf
    pip install -e .


Step 2: Patch custom Tokenizer and updated TPOT metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In public LLMPerf, ``hf-internal-testing`` tokenizer is used for all models which leads to incorrect
performance metrics due to counting more or less tokens than were actually processed by the model
on the server. Instead, we use the tokenizer of the model that is being benchmarked. 

LLMPerf includes TTFT in Time per Output Token(or Inter Token Latency) calculation. As TPOT and TTFT are two different metrics, a change is done to LLMPerf
to exclude TTFT from TPOT calculation to keep it consistent with how other industry standard performance benchmarks are done.

Follow these instructions to apply the patch to the LLMPerf library.

* Download the ``neuron_perf.patch`` :download:`file </src/benchmark/helper_scripts/neuron_perf.patch>` into the ``llmperf`` directory. 
* Run ``git apply neuron_perf.patch``. Confirm changes with ``git diff``.


Step 3: Patch data parallel benchmarking with multiple model endpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To measure performance with data parallel inference using multiple model copies, 
we allow users to provide multiple semicolon separated endpoints via `OPENAI_API_BASE` 
(e.g. "export OPENAI_API_BASE=http://server1;http://server2;http://server3") for
the OpenAI chat completion client. By default, the patch uses round-robin to route
requests.

* Download the ``llmperf_dp.patch`` :download:`file </src/benchmark/helper_scripts/llmperf_dp.patch>` into the ``llmperf`` directory. 
* Run ``git apply llmperf_dp.patch``. Confirm changes with ``git diff``.


Step 4: Patch reasoning model support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To measure LLM Inference performance of reasoning models, we need to patch LLMPerf to measure TTFT up to the 
first reasoning token instead of the first answer token.

* Download the ``llmperf_reasoning.patch`` :download:`file </src/benchmark/helper_scripts/llmperf_reasoning.patch>` into the ``llmperf`` directory. 
* Run ``git apply llmperf_reasoning.patch``. Confirm changes with ``git diff``.
