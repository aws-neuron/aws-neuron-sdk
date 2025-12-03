.. meta::
    :description: Tutorial for deploying GPT-OSS 120B on Trainium3 instances using NeuronX Distributed (NxD) Inference with vLLM.
    :keywords: GPT-OSS 120B, Trainium3, NeuronX Distributed Inference, NxD Inference, vLLM, Large Language Models, LLM Deployment, Tensor Parallelism, Data Parallelism, Speculative Decoding, Neuron SDK
    :date-modified: 12/02/2025

.. _nxdi-trn3-gpt-oss-120b-tutorial:

Tutorial: GPT-OSS 120B on Trn3 instances [BETA]
=======================================================================

NeuronX Distributed (NxD) Inference allows you to deploy GPT-OSS 120B on
Trn3 instances for high-performance inference. This tutorial provides a step-by-step
guide to deploy GPT-OSS 120B on a Trn3 instance using tensor parallelism, 
data parallelism and optimized kernels for efficient inference at scale.

.. contents:: Table of contents
   :local:
   :depth: 2

Prerequisites
-------------

As a prerequisite, this tutorial requires that you have a Trn3 instance
created from a Deep Learning AMI which is current private[Beta] 
that has the Neuron SDK with support for GPT-OSS 120B on Trn3 instances pre-installed.

.. note::

    Please contact us to get access to the private Deep Learning AMI for 2.27 Beta release
    that has all the necessary artifacts for you to run this tutorial on Trn3 instance.


The Deep Learning AMI contains the following:

* Neuron system dependencies
* Python virtual environment with Neuron SDK and vLLM v0.11.1 in :code:`~/neuronx_gpt_oss_120b_in_vllm_venv`
* vLLM startup script at :code:`~/start_vllm_server.sh`
* GPT-OSS 120B and EAGLE3 draft model checkpoints in :code:`/mnt/inference/models/`


Performance Optimizations
-------------------------

The model is configured to run with data parallelism i.e. 8 independent vLLM endpoints per Trn3
instance each using tensor parallelism with :code:`tp_degree=8` and :code:`LNC=2`. Furthermore, we use the
following performance optimizations:

* speculative decoding using EAGLE3 with speculation length 5
* optimized NKI kernels for attention, MoE, sampling
* support for MXFP4 compute in MoE (Trn3 only)

For more information see:

* :ref:`moe-inference-deep-dive`
* :ref:`logical-neuroncore-config`
* :ref:`trainium3-arch`


Step 1: Launch vLLM server
--------------------------

Use the included script to launch a vLLM server on the instance.

::

    source ./neuronx_gpt_oss_120b_in_vllm_venv/bin/activate
    bash start_vllm_server.sh


During first start up, the model will be compiled and serialized. Subsequent startups will
directly load from the serialized model (:ref:`nxdi-vllm-v1-serialization`). You should see output indicating the server is 
ready:


::

    INFO:     Started server process
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8000



The setup is intended to be used with data parallelism and supports running 8 copies on
one Trn3 instance. You will need to provide a unique port (:code:`--port 8000`) and update the visible 
NeuronCores range (:code:`export NEURON_RT_VISIBLE_CORES=0-7`) for each copy. If you want to start multiple
servers concurrently without loading from a serialized model, you also need to provide each with
a unique compiler working directory by setting the :code:`BASE_COMPILE_WORK_DIR` environment variable.
Please refer to :ref:`/libraries/nxd-inference/tutorials/trn2-llama3.3-70b-dp-tutorial.ipynb` for more information.

Currently, (NxD) Inference supports EAGLE3 heads with the same hidden size and vocabulary size as the target model
and follow the Llama3 dense architecture. It must contain the following layers:
:code:`fc, hidden_norm, input_layernorm, attention, mlp, lm_head and embed_tokens`. Any other EAGLE3 head 
architecture needs to be brought up as a new model.


Step 2: Test inference with sample requests
--------------------------------------------

With the vLLM server running, open a new terminal session and test the inference endpoint.

First, verify the server is responding:

::

    curl -i http://localhost:8000/health

You should receive a :code:`HTTP/1.1 200 OK` response.

Now, send a sample inference request:

::

    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "/mnt/inference/models/gpt-oss-120b",
            "messages": [
                {"role": "user", "content": "How are you?"}
            ]
        }'

You should receive a JSON response with the generated text.


Step 3: Run performance benchmarks
-----------------------------------

We are going to use LLMPerf for benchmarking. Install LLMPerf from source and 
patch it to support data parallelism and reasoning models following :ref:`llm-inference-benchmarking`.

Then, run the benchmark with the following commands:

::

    export OPENAI_API_KEY=EMPTY

    # if you have started multiple vLLM servers, 
    # append the endpoints separated by semicolon
    # e.g. `export OPENAI_API_BASE="http://localhost:8000/v1;http://localhost:8001/v1"`
    # and adjust `--num-concurrent-requests` accordingly. You might also want to increase
    # `--max-num-completed-requests`.
    export OPENAI_API_BASE="http://0.0.0.0:8000/v1"

    python ~/llmperf/token_benchmark_ray.py \
        --model /mnt/inference/models/gpt-oss-120b \
        --mean-input-tokens 10000 \
        --stddev-input-tokens 0 \
        --mean-output-tokens 3000 \
        --stddev-output-tokens 0 \
        --num-concurrent-requests 1 \
        --results-dir "./llmperf_results/" \
        --max-num-completed-requests 50 \
        --additional-sampling-params '{"temperature": 1.0, "top_k": 1.0, "top_p": 1.0}' \
        --llm-api "openai"

Step 4: Clean up
----------------

To stop the vLLM server and free up resources:

1. Press ``Ctrl+C`` in the terminal running the vLLM server
2. Verify all processes have stopped:

::

    ps aux | grep vllm

3. If any vLLM processes are still running, terminate them using their process IDs (PIDs): ``kill -9 <PID>``.
   
You have now successfully deployed GPT-OSS 120B on a Trn3 instance using NxD Inference with vLLM!
