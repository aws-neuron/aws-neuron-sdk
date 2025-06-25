.. _nxdi-trn2-llama3.3-70b-dp-tutorial:

Tutorial: Scaling LLM Inference with Data Parallelism on Trn2
=======================================================================================================

Introduction
------------
This tutorial demonstrates how to implement data parallelism (DP) LLM inference with multiple model copies on Neuron. The following sections provides a sequence of steps to stand up multiple Llama 3.3 70B model endpoints on a single `trn2.48xlarge` instance with NxD Inference and vLLM and run data parallel inference. 

.. contents:: Table of contents
   :local:
   :depth: 2

Data Parallel Inference
-----------------------
We can achieve Data Parallelism by using multiple copies of the same model hosted on the instance to process multiple requests simultaneously. Using NxD Inference and vLLM, you can deploy multiple model endpoints by adjusting the tensor parallel degree (Tensor Parallelism (TP) refers to sharding model weight matrices onto multiple NeuronCores within each model copy) and allocating appropriate NeuronCore ranges for each model endpoint. While increasing the batch size with a single copy of the model increases throughput, introducing data parallelism with multiple model endpoints combined with tensor parallelism allows further increase in instance throughput with some impact to latency. Use this technique when you can relax the latency constraint of your application to further maximize the throughput of the instance. 

In this tutorial we use Llama 3.3 70B with DP=2 and TP=32, However, you can follow the same sequence of steps to deploy additional model copies by appropriately changing the tensor parallel degree. You can also use this guide to deploy multiple copies of any other models on Trn1 or Inf2 instances as long as the model fits and the DP x TP degree does not exceed the number of model cores.

Prerequisites:
---------------
Setup and Connect to an Amazon EC2 Trn2 Instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An Amazon EC2 ``trn2.48xlarge`` instance with AWS Neuron SDK version 2.23.0 or later (:ref:`latest-neuron-release`) is required. 

To launch a Trn2 instance using Deep Learning AMI with pre-installed Neuron SDK and NxD Inference dependencies, see :ref:`nxdi-setup`.

Make sure to activate the Neuron virtual environment

::

    source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate

To verify that NxD Inference has installed successfully, check that you can run the `inference_demo` console script.
::

    inference_demo --help

Download  Model Weights
~~~~~~~~~~~~~~~~~~~~~~~~~
To use this tutorial, you must first download a Llama 3.3 70B Instruct model checkpoint from Hugging Face to a local path on the Trn2 instance. For more information, see `Downloading Models <https://huggingface.co/docs/hub/en/models-downloading>`_ in the Hugging Face documentation. You can download and use `meta-llama/Llama-3.3-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`_ for this tutorial.


Install Neuron vLLM Fork
~~~~~~~~~~~~~~~~~~~~~~~~~

NxD Inference supports running models with vLLM. This functionality is
available in the AWS Neuron fork of the vLLM GitHub repository. Install the latest release branch of vLLM from the AWS Neuron fork 
following instructions in the :ref:`vLLM User Guide for NxD Inference<nxdi-vllm-user-guide>`.

.. _install_llmperf:

Install LLMPerf
~~~~~~~~~~~~~~~~

In this tutorial, you will use `LLMPerf <https://github.com/ray-project/llmperf>`_ to measure the performance. 

Install llmperf into the virtual environment.

::

    git clone --branch v2.0 https://github.com/ray-project/llmperf.git
    cd llmperf
    pip install -e .    

Once you have installed LLMPerf, please apply relevant patches as described in :ref:`llm-inference-benchmarking` . Ensure that you apply all the patches described there including the data parallelism support patch. 

Step-by-Step Tutorial Instructions
-----------------------------------

Step 1: Compile the model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Before we launch the model endpoint with vLLM, we'll use the NxD Inference library to compile the model with an appropriate configuration. Refer to :ref:`nxdi-feature-guide` for more information. To compile a model for data parallelism inference, set the ``NUM_CORES``, ``TP_DEGREE``, ``BATCH_SIZE`` to allow for strategic workflow distribution. For DP=2 with BATCH_SIZE>=1, TP_DEGREE should be set to 64/2=32 to maximize NeuronCore utilization across all model copies. Simply create and run a shell script as illustrated below:

`compile_model.sh`

::

    #!/bin/bash
    # Replace with path to your downloaded Hugging Face model checkpoints
    MODEL_PATH="/ubuntu/model_hf/Llama-3.3-70B-Instruct/"

    # This is where the compiled model will be saved. The same path
    # should be used when launching vLLM server for inference.
    export COMPILED_MODEL_PATH="/ubuntu/traced_model/Llama-3.3-70B-Instruct/"

    NUM_CORES=128
    TP_DEGREE=32
    LNC=2
    BATCH_SIZE=4

    export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
    export NEURON_RT_NUM_CORES=$((NUM_CORES/NEURON_RT_VIRTUAL_CORE_SIZE))
    export NEURON_RT_EXEC_TIMEOUT=600
    export XLA_DENSE_GATHER_FACTOR=0
    export NEURON_RT_INSPECT_ENABLE=0

    inference_demo \
        --model-type llama \
        --task-type causal-lm \
            run \
            --model-path $MODEL_PATH \
            --compiled-model-path $COMPILED_MODEL_PATH \
            --torch-dtype bfloat16 \
            --start_rank_id 0 \
            --local_ranks_size $TP_DEGREE \
            --tp-degree $TP_DEGREE \
            --batch-size $BATCH_SIZE \
            --max-context-length 8192 \
            --seq-len 8192 \
            --on-device-sampling \
            --top-k 1 \
            --do-sample \
            --fused-qkv \
            --qkv-kernel-enabled \
            --attn-kernel-enabled \
            --mlp-kernel-enabled \
            --pad-token-id 2 \
            --compile-only \
            --prompt "What is annapurna labs?" 2>&1 | tee log

To compile the model, run this script with command:  ``./compile_model.sh`` 

It's important to specify the path to which the compiled model is saved, as this same path must be used when you later launch the vLLM server for inference, allowing you to use the pre-compiled model without having to compile it again. 

.. note::

    To run this script on trn1, set LNC=1. For more information about LNC, see :ref:`logical-neuroncore-config` .
    Also appropriately change NUM_CORES & TP_DEGREE (eg. 16 for DP=2)

For detailed information about the inference_demo flags, you can consult the :ref:`nxd-inference-api-guide`.


Step 2: Launch model endpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a deployment script (``deploy_vllm_endpoint.sh``) containing below code snippet that configures and launches a model endpoint. The script is parameterized so that you can pass a specific port number, range of neuron cores, tensor parallel degree and batch size. 

Key Parameters Explained:

    * ``MODEL_PATH``: The Hugging Face model identifier or local model_hf path containing Meta-Llama-3.3-70B-Instruct hugging face checkpoints. Eg. /home/ubuntu/model_hf/Llama-3.3-70B-Instruct/
    * ``port``: Network port for the endpoint Eg. 8000. The port number should be unique for each model endpoint. 
    * ``cores``: Range of NeuronCores allocated to this endpoint. This should be a non overlapping range of cores when deploying multiple model endpoints on the same instance. For example, when allocated 32 NeuronCores to a model endpoint specify 0-31 or 32-63. 
    * ``tp_degree``: Degree of tensor parallelism for model sharding. To maximize NeuronCores utilization, reduce tp_degree  while increasing dp_degree.
    * ``bs`` : Batch size specified for model endpoint. 

These parameters should match the values used during compilation step above. 

`deploy_vllm_endpoint.sh`

::

    # Model deployment script with detailed configuration

    # Default values for arguments
    DEFAULT_PORT=8000
    DEFAULT_CORES="0-31"
    DEFAULT_TP_DEGREE=32
    DEFAULT_BS=4

    # Help function
    show_help() {
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  -p port        Port number for vLLM endpoint (default: $DEFAULT_PORT)"
        echo "  -c cores       Range of neuron cores (default: $DEFAULT_CORES)"
        echo "  -t tp_degree   Tensor parallel degree (default: $DEFAULT_TP_DEGREE)"
        echo "  -b bs          Batch size (default: $DEFAULT_BS)"
        echo "  -h             Show this help message"
    }

    # Parse single-letter arguments
    while getopts "p:c:t:b:h" opt; do
        case $opt in
            p) port="$OPTARG" ;;
            c) cores="$OPTARG" ;;
            t) tp_degree="$OPTARG" ;;
            b) bs="$OPTARG" ;;
            h) show_help; exit 0 ;;
            ?) show_help; exit 1 ;;
        esac
    done

    # Set defaults if not provided
    port=${port:-$DEFAULT_PORT}
    cores=${cores:-$DEFAULT_CORES}
    tp_degree=${tp_degree:-$DEFAULT_TP_DEGREE}
    bs=${bs:-$DEFAULT_BS}

    # Environment configurations
    export NEURON_RT_INSPECT_ENABLE=0
    export NEURON_RT_VIRTUAL_CORE_SIZE=2

    # These should be the same paths used when compiling the model.
    MODEL_PATH="/ubuntu/model_hf/Llama-3.3-70B-Instruct/"
    COMPILED_MODEL_PATH="/ubuntu/traced_model/Llama-3.3-70B-Instruct/"

    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH
    export NEURON_RT_VISIBLE_CORES=${cores}

    VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --max-num-seqs ${bs} \
        --max-model-len 12800 \
        --tensor-parallel-size ${tp_degree} \
        --device neuron \
        --use-v2-block-manager \
        --override-neuron-config "{\"on_device_sampling_config\": {\"do_sample\": true, \"global_topk\": 64}}" \
        --port ${port} &
    PID=$!
    echo "vLLM server started with PID $PID"

Run this script to launch 2 vLLM servers. You can run these commands as background processes in the same terminal or run two seperate terminals for each command. We launch two servers, each with a tensor parallel degree of 32 and batch size of 4. Note that the first vLLM server uses neuron cores 0-31 and the second one 32-63.  You can pick any ports that are available.

::

    ./deploy_vllm_endpoint.sh -p 8000 -c 0-31 -t 32 -b 4 &

and

::

    ./deploy_vllm_endpoint.sh -p 8001 -c 32-63 -t 32 -b 4 &


The server start up time can take a few minutes since the model weights are getting loaded. Once the vLLM servers have been launched, you should see the following log output. This implies that the model server has been deployed.

::

    INFO:     Started server process [221607]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)


Step 3: Benchmark the deployed model endpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the above steps, the vLLM server should be running. You can now measure the performance using LLMPerf. Ensure you have made the required changes to use LLMPerf with DP>1 by following :ref:`install_llmperf`

Below is a sample shell script to run LLMPerf. The script allows the user to specify tensor parallelism degree, data parallelism degree, and batch size through command-line arguments, with default values provided. It calculates the concurrency based on batch size and data parallelism, sets up the environment for benchmarking with input tokens N(7936, 30) and output tokens N(256,30), and then runs LlmPerf’s ``token_benchmark_ray.py`` with various parameters to measure the model endpoints’ performance. The benchmark simulates requests with specific input and output token distributions, and collects results for analysis. 

More information about several arguments used in the script can be found in the
`llmperf open source code <https://github.com/ray-project/llmperf/blob/main/token_benchmark_ray.py.>`_

`benchmark_model.sh`

::

    #!/bin/bash

    # Default values for arguments
    DEFAULT_TP_DEGREE=32
    DEFAULT_DP_DEGREE=2
    DEFAULT_BS=1

    # Help function
    show_help() {
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  -t tp_degree          Tensor parallel degree (default: $DEFAULT_TP_DEGREE)"
        echo "  -d dp_degree          Data parallel degree (default: $DEFAULT_DP_DEGREE)"
        echo "  -b bs          Batch size (default: $DEFAULT_BS)"
        echo "  -h             Show this help message"
    }

    # Parse single-letter arguments
    while getopts "t:d:b:h" opt; do
        case $opt in
            t) tp_degree="$OPTARG" ;;
            d) dp_degree="$OPTARG" ;;
            b) bs="$OPTARG" ;;
            h) show_help; exit 0 ;;
            ?) show_help; exit 1 ;;
        esac
    done

    # Set defaults if not provided
    tp_degree=${tp_degree:-$DEFAULT_TP_DEGREE}
    dp_degree=${dp_degree:-$DEFAULT_DP_DEGREE}
    bs=${bs:-$DEFAULT_BS}

    # Calculate total concurrent requests (batch_size * data_parallelism)
    # If result is less than 1, default to batch_size 
    concurrency=$(awk -v batch="$bs" -v dp_degree="$dp_degree" 'BEGIN {
        concurrency = int(batch * dp_degree)
        print (concurrency >= 1 ? concurrency : batch)
    }')
    echo "concurrency: $concurrency"

    MODEL_PATH="/ubuntu/model_hf/Llama-3.3-70B-Instruct/"

    # Modify OpenAI's API key and API base to use vLLM's API server.
    export OPENAI_API_KEY=EMPTY

    #if you have more vLLM servers, append the required number of ports like so:
    #;http://localhost:8001/v1;http://localhost:8002/v1"
    export OPENAI_API_BASE="http://0.0.0.0:8000/v1;http://0.0.0.0:8001/v1"

    python /root/llmperf/token_benchmark_ray.py \
    --model ${MODEL_PATH} \
    --mean-input-tokens 7936 \
    --stddev-input-tokens 30 \
    --mean-output-tokens 256 \
    --stddev-output-tokens 30 \
    --num-concurrent-requests ${concurrency} \
    --results-dir "/ubuntu/results/" \
    --timeout 21600 \
    --max-num-completed-requests 1000 \
    --additional-sampling-params '{"temperature": 0.7, "top_k": 50}' \
    --llm-api "openai"
 
Run this script with ``./benchmark_model.sh -t 32 -d 2 -b 4`` . These args match the args set while launching vLLM servers above.

Once the script starts executing, you will see output like:

::

    INFO worker.py:1852 -- Started a local Ray instance.
      4%|▍         | 39/1000 [01:29<30:14,  1.89s/it]

Once benchmarking is complete, results can be found in the directory specified with the --results-dir flag in the ``benchmark_vllm.sh`` script


Conclusion
-----------

This tutorial demonstrates how 

data parallelism using multiple model copies can help increase the throughput. While standard batching (DP=1, BS>1) processes multiple requests through a single model copy, data parallelism deploys multiple independent model copies that can process different requests simultaneously.
Our experiments with batch sizes 1 & 4 show that as we decrease Tensor Parallelism (TP) from 64 to 16 and increase Data Parallelism (DP) from 1 to 4, we see up to 2x throughput improvement with non optimized configurations. However, this comes with an increase in Time To First Token (TTFT) latency. This illustrates a key consideration: while DP can improve overall system throughput by processing more concurrent requests, it can lead to higher latency

When to choose Data parallel with multiple model copies over using single model copy in an instance:

* Use DP when your workload is collective-bound rather than memory or compute-bound. At high batch sizes, TP64 / TP128 collectives can become slow due to the number of hops and increasing throughput requirements. At high enough batch size, it can be better to pay the cost of duplicated weight loads and use DP with multiple model copies in order to reduce collective latencies.
* Consider DP when you need to handle many concurrent requests and can tolerate moderate latency increases

Implementation requires careful consideration of your total memory budget, as each additional model copy increases memory consumption. You'll need to balance the number of model copies against the resources allocated to each model copy based on your specific throughput and latency requirements.
By understanding these trade-offs and following the implementation guidelines in this tutorial, users can select the most appropriate approach for their specific use case and optimize their inference setup accordingly.
