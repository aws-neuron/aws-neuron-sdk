.. _nxdi-sd-inference-tutorial:

Tutorial: Using Speculative Decoding (SD) to improve inference performance on Trn2 instances
============================================================================================

NeuronX Distributed Inference (NxDI) allows you to deploy large language models on
Trn2 or Trn1 instances. This tutorial provides a step-by-step guide to deploy a Qwen3-32B model
on a Trn2 instance using two configurations: one without speculative decoding and one
with Qwen3-0.6B as the draft model for speculative decoding. We use LLMPerf to measure and compare
performance between the two configurations. While this tutorial uses Qwen models for
demonstration, the approach is model-agnostic and can be applied to other supported models
(see :ref:`nxdi-model-reference`).

.. contents:: Table of contents
   :local:
   :depth: 2

Environment setup
-----------------
This tutorial requires that you have a Trn2 instance created from a Deep Learning AMI that has the Neuron SDK pre-installed.
To set up a Trn2 instance using Deep Learning AMI with pre-installed Neuron SDK,
see :ref:`nxdi-setup`.

Connect to the EC2 instance via your preferred option: EC2 Instance Connect, Session Manager, or SSH client.
For more information, see `Connect to your Linux instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-connect-methods.html>`_ in the Amazon EC2 User Guide.

Start a built-in vLLM Neuron Deep Learning Container (DLC). For more information about available containers,
see the `AWS Neuron Deep Learning Containers repository <https://github.com/aws-neuron/deep-learning-containers#vllm-inference-neuronx>`_.

For example, we use the following:

::

    docker run -d -it --privileged -v /home/ubuntu/:/home/ubuntu/ public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py311-sdk2.26.1-ubuntu22.04

Scenario 1: Run without Speculative Decoding
---------------------------------------------

Step 1: Set environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Populate the following environment variables:

::

    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    export NEURON_COMPILED_ARTIFACTS="/home/ubuntu/Qwen-32B-BS1-SL6k-TP64"
    export MODEL_ID="Qwen/Qwen3-32B"

NxDI will persist the compiled model artifacts on the EC2 instance in ``NEURON_COMPILED_ARTIFACTS`` so you can rerun the model without recompiling it. If you need to recompile it, empty the folder.

Step 2: Start the vLLM server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Invoke the model:

::

    VLLM_USE_V1=0 vllm serve $MODEL_ID \
        --tensor-parallel-size 64 \
        --max-num-seqs 1  \
        --max-model-len 6400 \
        --override-neuron-config '{"save_sharded_checkpoint": true}'  

We use ``tensor-parallel-size 64`` assuming the default Logical NeuronCore (LNC) configuration.
For more information about LNC, see `Trainium2 Architecture <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html>`_.

We also use ``max-num-seqs 1`` as a baseline. Feel free to adjust this value to your needs. We will use the same value for both scenarios.

Finally, we use ``save_sharded_checkpoint: true`` to speed up model loading after compilation.
For more information, see the :ref:`NeuronX Distributed Save/Load Developer Guide <neuronx_distributed_save_load_developer_guide>`.

After the model compiles, you will see the following output:

::

    INFO:     Started server process [7]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.

This indicates the server is ready and the model endpoint is available for inference.

Step 3: Test the endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~
You can test the endpoint using curl or any HTTP client:

::

    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen3-32B",
            "prompt": "What is machine learning?",
            "max_tokens": 100,
            "temperature": 0.7
        }'

Step 4: Load the model and measure performance with LLMPerf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Login to the docker container that runs the model (``docker exec -it ...``) and install llmperf:

::

    cd /opt
    git clone https://github.com/ray-project/llmperf.git
    cd llmperf
    pip install -e .

    export OPENAI_API_BASE="http://localhost:8000/v1"
    export OPENAI_API_KEY=dummy

    python token_benchmark_ray.py \
        --model "Qwen/Qwen3-32B" \
        --mean-input-tokens 128 \
        --stddev-input-tokens 0 \
        --mean-output-tokens 512 \
        --stddev-output-tokens 0 \
        --max-num-completed-requests 10 \
        --timeout 1200 \
        --num-concurrent-requests 1 \
        --results-dir /tmp/results \
        --llm-api openai \
        --additional-sampling-params '{}'

We used ``mean-output-tokens 512`` as a baseline example of an output token length to demonstrate SD performance. Shorter values in our case here did not show significant benefits.

Log the results (we kept p99 for brevity):

::

    ttft_s 0.04828366368776187
    end_to_end_latency_s 6.044886132028841
    request_output_throughput_token_per_s 102.27375153804246
    number_input_tokens 128.0
    number_output_tokens 558.0


Scenario 2: Run with Speculative Decoding
------------------------------------------

Step 1: Set environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For speculative decoding, we need to specify both the target model and the draft model:

::

    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    export NEURON_COMPILED_ARTIFACTS="/home/ubuntu/Qwen-32B-BS1-SL6k-TP64-SD"
    export MODEL_ID="Qwen/Qwen3-32B"
    export DRAFT_MODEL_ID="Qwen/Qwen3-0.6B"

Step 2: Start the vLLM server with speculative decoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Invoke the model with speculative decoding enabled:

::

    VLLM_USE_V1=0 vllm serve $MODEL_ID \
        --tensor-parallel-size 64 \
        --max-num-seqs 1 \
        --max-model-len 6400 \
        --override-neuron-config '{"save_sharded_checkpoint": true, "enable_fused_speculation": true}' \
        --speculative-config '{"model": "'"$DRAFT_MODEL_ID"'", "num_speculative_tokens": 7, "max_model_len": 2048, "method": "eagle"}'

The key differences from the baseline configuration are:

- ``--speculative-config``: Specifies the draft model configuration including:
  
  - ``model``: The draft model path (Qwen3-0.6B)
  - ``num_speculative_tokens``: Number of tokens to speculatively generate (7 in this example)
  - ``max_model_len``: Maximum sequence length for the draft model (2048)
  - ``method``: Speculative decoding method (eagle)

- ``enable_fused_speculation``: Enables fused speculation in the Neuron config for improved performance by combining draft model execution with verification

After the model compiles, you will see the same startup messages indicating the server is ready.

Step 3: Test the endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~
Test the endpoint the same way as in Scenario 1:

::

    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen3-32B",
            "prompt": "What is machine learning?",
            "max_tokens": 100,
            "temperature": 0.7
        }'

Step 4: Load the model and measure performance with LLMPerf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Login to the docker container that runs the model (``docker exec -it ...``) and follow Step 4 from the non-SD experiment.
Run the load test with the same configuration.

Log the results (we kept p99 for brevity):

::

    ttft_s 0.04737630250383518
    end_to_end_latency_s 5.6368158639998
    request_output_throughput_token_per_s 137.84216889131872
    number_input_tokens 128.0
    number_output_tokens 565.37


Performance Comparison
----------------------

The table below summarizes the key performance metrics (p99 values) from both configurations:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Metric
     - Without SD
     - With SD
   * - Time to First Token (TTFT)
     - 48.3 ms
     - 47.4 ms
   * - End-to-End Latency
     - 6.04 s
     - 5.64 s
   * - Request Output Throughput
     - 102.3 tokens/s
     - 137.8 tokens/s
   * - Number of Input Tokens
     - 128
     - 128
   * - Number of Output Tokens
     - 558
     - 565

Key observations:

- **Throughput improvement**: Speculative decoding achieves 35% higher throughput (137.8 vs 102.3 tokens/s)
- **Latency reduction**: End-to-end latency is reduced by 7% (5.64s vs 6.04s)
- **TTFT**: Time to first token remains comparable between both configurations

Conclusion
----------
For Qwen3-32B with Qwen3-0.6B as the draft model on Trn2, speculative decoding delivers
35% higher throughput and 7% lower end-to-end latency at 512 output tokens. Performance
gains vary based on model pairing, output length, and workload characteristics. Use this
benchmarking approach to validate the optimal configuration for your use case. 
