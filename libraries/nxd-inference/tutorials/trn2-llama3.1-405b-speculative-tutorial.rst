.. _nxdi-trn2-llama3.1-405b-speculative-tutorial:

Tutorial: Using Speculative Decoding and Quantization to improve Llama-3.1-405B inference performance on Trn2 instances
=======================================================================================================================

NeuronX Distributed (NxD) Inference allows you to deploy Llama3.1 405B on
a single Trn2 instance. This tutorial will show you how to optimize inference performance for Llama3.1 405B on a Trn2 instance
with speculative decoding and quantization. We will compile and load the model into a VLLM server and measure performance using LLMPerf.
This tutorial consists of two parts. In the first part, we will collect performance metrics for our base configuration with ``bf16`` model weights. In the second part, we will optimize inference performance with ``fp8`` quantized weights and speculative decoding. 
The performance is then compared with the results from part 1.

.. contents:: Table of contents
   :local:
   :depth: 2

Prerequisites
-----------------------------------------------


Set up and connect to a Trn2.48xlarge instance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a prerequisite, this tutorial requires that you have a Trn2 instance
created from a Deep Learning AMI that has the Neuron SDK pre-installed.

To set up a Trn2 instance using Deep Learning AMI with pre-installed Neuron SDK,
see :ref:`nxdi-setup`.

After setting up an instance, use SSH to connect to the Trn2 instance using the key pair that you
chose when you launched the instance.

After you are connected, activate the Python virtual environment that
includes the Neuron SDK.

::

   source ~/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate

Run ``pip list`` to verify that the Neuron SDK is installed.

::

   pip list | grep neuron

You should see Neuron packages including
``neuronx-distributed-inference`` and ``neuronx-cc``.

Install packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NxD Inference supports running models with vLLM. This functionality is
available in a fork of the vLLM GitHub repository:

- `aws-neuron/upstreaming-to-vllm <https://github.com/aws-neuron/upstreaming-to-vllm/tree/v0.6.x-neuron>`__

To run NxD Inference with vLLM, you need to download and install vLLM from this
fork. Clone the Neuron vLLM fork.

::
   
    git clone -b v0.6.x-neuron https://github.com/aws-neuron/upstreaming-to-vllm.git


Make sure to activate the Neuron virtual environment if using a new terminal instead of the one from connect step above.

::
    
    source ~/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate


Install the Neuron vLLM fork into the virtual environment.

::
    
    cd upstreaming-to-vllm
    pip install -r requirements-neuron.txt
    VLLM_TARGET_DEVICE="neuron" pip install -e .
    cd ..


In this tutorial, you will use `llmperf <https://github.com/ray-project/llmperf>`_ to measure the inference performance of the base Llama-3.1-405b-Instruct configuration and the more
optimized configuration. 
We will use the `load test <https://github.com/ray-project/llmperf?tab=readme-ov-file#load-test>`_ feature of LLMPerf and measure the performance for accepting
10,000 tokens as input and generating 1500 tokens as output.
Install llmperf into the virtual environment.

::

    git clone https://github.com/ray-project/llmperf.git
    cd llmperf
    pip install -e . 


Download models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run inference in the first part of the tutorial, you need to download the Llama-3.1-405b-Instruct model checkpoint with ``bf16`` weights from Hugging Face (`meta-llama/Llama-3.1-405B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct>`__). 
For the second part of the tutorial, you will run a more optimized inference configuration. For this part, you need to download an fp8-quantized Llama3.1-405B-FP8 model checkpoint (`meta-llama/Llama-3.1-405B-Instruct-FP8 <https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct-FP8>`__).
With Speculative Decoding, you will also need to specify a draft model. You can download and use the model checkpoint from `meta-llama/Llama-3.2-1B-Instruct <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`__.
For more information, see
`Downloading models <https://huggingface.co/docs/hub/en/models-downloading>`__
in the Hugging Face documentation. 

Scenario 1: Run Llama-3.1-405b inference with base configuration using ``bf16`` weights
----------------------------------------------------------------------

Step 1: Compile the model and run generate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We will first compile and run generation on a sample prompt using a command
installed by ``neuronx-distributed-inference``. Save the contents of the below script to your favorite 
shell script file, for example, ``compile_model.sh`` and then run it.

Note that we are using the following features as described in
the tutorial for running 405B model :ref:`nxdi-trn2-llama3.1-405b-tutorial`

* Logical NeuronCores (LNC)
* Tensor parallelism (TP) on Trn2
* Optimized Kernels

The script compiles the model and runs generation on the given input prompt. Please refer to :ref:`nxd-inference-api-guide` for more information on these ``inference_demo`` flags.
Note the path we used to save the compiled model. This path should be used
when launching vLLM server for inference so that the compiled model can be loaded without recompilation.

::

    # Replace this with the path where you downloaded and saved the model files.
    MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct/"
    # This is where the compiled model will be saved. The same path
    # should be used when launching vLLM server for inference.
    COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.1-405B-Instruct/"

    NUM_CORES=128
    TP_DEGREE=64
    LNC=2

    export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
    export NEURON_RT_NUM_CORES=$((NUM_CORES/NEURON_RT_VIRTUAL_CORE_SIZE))
    export NEURON_RT_EXEC_TIMEOUT=600 


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
            --batch-size 1 \
            --max-context-length 12288 \
            --seq-len 12800 \
            --on-device-sampling \
            --top-k 1 \
            --do-sample \
            --fused-qkv \
            --sequence-parallel-enabled \
            --qkv-kernel-enabled \
            --attn-kernel-enabled \
            --mlp-kernel-enabled \
            --cc-pipeline-tiling-factor 1 \
            --pad-token-id 2 \
            --logical-neuron-cores $LNC \
            --enable-bucketing \
            --prompt "What is annapurna labs?" 2>&1 | tee log


The above script will compile a Neuron model for this base-case configuration, and also run generate on the example prompt specified with the ``-prompt`` flag. 
You can change this prompt to your prompt of choice. 
The script's output will be written into ``log``, a log file in the working directory. 

In addition, in the subsequent runs of this script, you can add a ``--skip-compile`` flag to skip 
the compiling step since the model is already compiled in the first run of the script. 
This will allow you to test the model with different prompts. 

Step 2: Start the Vllm server with the compiled Neuron model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After compiling the model, you can run the model using vLLM. Save the contents of the below script to another
shell script file, for example, ``start_vllm.sh`` and then run it.

::

    export NEURON_RT_VIRTUAL_CORE_SIZE=2


    MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct"
    COMPILED_MODEL_PATH="/home/ubuntu/traced_models/Llama-3.1-405B-Instruct"


    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH
    VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
        -—model $MODEL_PATH \
        -—max-num-seqs 1 \
        -—max-model-len 12800 \
        -—tensor-parallel-size 64 \
        -—device neuron \
        -—use-v2-block-manager \
        -—override-neuron-config "{}" \
        -—port 8000 & PID=$!
    echo "vLLM server started with PID $PID"

Step 3: Measure performance using LLMPerf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After the above steps, the vllm server should be running. Before we can use the ``llmperf`` package, we need to make a few changes to its code. 
Follow :ref:`benchmarking with LLMPerf guide <llm_perf_patch_changes>` to apply the code changes. 
    
We can now measure the performance using ``llmperf``. Below is a sample shell script to run ``llmperf``. More information about several arguments used in the script can be found in the 
`llmperf open source code <https://github.com/ray-project/llmperf/blob/main/token_benchmark_ray.py>`_ .

::

    # This should be the same path to which the model was downloaded (also used in the above steps).
    MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct"
    # This is the name of directory where the test results will be saved.
    OUTPUT_PATH=llmperf-results-sonnets

    export OPENAI_API_BASE="http://localhost:8000/v1"
    export OPENAI_API_KEY="mock_key"

    python token_benchmark_ray.py \
        --model $MODEL_PATH \
        --mean-input-tokens 10000 \
        --stddev-input-tokens 0 \
        --mean-output-tokens 1500 \
        --stddev-output-tokens 0 \
        --num-concurrent-requests 1\
        --timeout 3600 \
        --max-num-completed-requests 50 \
        --additional-sampling-params '{}' \
        --results-dir $OUTPUT_PATH \
        --llm-api "openai"


The output for this llama-3.1-405B model run for the base case is shown below. Please note that the numbers can slightly vary between runs but should be in the same order of magnitude.
::
    
    Results for token benchmark for /home/ubuntu/models/llama-3.1-405b queried with the openai api.

    inter_token_latency_s
        p25 = 0.03783673520494379
        p50 = 0.037929154633788834
        p75 = 0.03799374728198055
        p90 = 0.03806084386428147
        p95 = 0.03818095359194858
        p99 = 0.03862880035825585
        mean = 0.03790912092492011
        min = 0.03711292916794487
        max = 0.03867580939426865
        stddev = 0.0002364662521116205
    ttft_s
        p25 = 2.437347081664484
        p50 = 2.441959390998818
        p75 = 2.4439403364085592
        p90 = 2.444729209714569
        p95 = 2.445114637189545
        p99 = 79.22927707570342
        mean = 5.451600373298861
        min = 2.427013176959008
        max = 153.00210832804441
        stddev = 21.29264628138615
    end_to_end_latency_s
        p25 = 70.06310007086722
        p50 = 70.09642704750877
        p75 = 70.1557097924524
        p90 = 70.28295350184199
        p95 = 70.56055794338462
        p99 = 148.28325726192182
        mean = 73.19207735829521
        min = 70.00512732309289
        max = 222.50397142698057
        stddev = 21.54750467688136
    request_output_throughput_token_per_s
        p25 = 25.417755028050912
        p50 = 25.463487985775544
        p75 = 25.522234144656743
        p90 = 25.6487981126861
        p95 = 25.729858763245502
        p99 = 25.90146713883131
        mean = 25.13808905954906
        min = 8.080754642125802
        max = 26.021214285642255
        stddev = 2.465472136291901
    number_input_tokens
        p25 = 10000.0
        p50 = 10000.0
        p75 = 10000.0
        p90 = 10000.0
        p95 = 10000.0
        p99 = 10000.0
        mean = 10000.0
        min = 10000
        max = 10000
        stddev = 0.0
    number_output_tokens
        p25 = 1783.0
        p50 = 1785.0
        p75 = 1789.75
        p90 = 1798.1
        p95 = 1803.55
        p99 = 1816.67
        mean = 1787.92
        min = 1779
        max = 1825
        stddev = 8.54720386310933
    Number Of Errored Requests: 0
    Overall Output Throughput: 24.421011092151268
    Number Of Completed Requests: 50
    Completed Requests Per Minute: 0.8195336846889548



Scenario 2: Run Llama-3.1-405b inference with fp8 weights and fused speculation (with draft model)
--------------------------------------------------------------------------------------------------

Step 1: Rescale the model weights to use Neuron FP8 format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Since Neuron device only supports the ``FP8_EXP4 (IEEE-754)`` data type, and the HuggingFace FP8 checkpoint for Llamma-405b is in a different FP8 format (``OCP FP8 E4M3/e4m3fn``) which has a different range, we need to rescale the public model weights. 
Follow this guide to rescale the FP8 model weights from HuggingFace: `link <https://github.com/aws-neuron/neuronx-distributed/blob/main/src/neuronx_distributed/quantization/README_rescaling_fp8_for_neuron.md>`__.

Next we will compile and run the model and record performance metrics.

Step 2: Compile the model and run generate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We will first compile and run generation on a sample prompt using a command
installed by ``neuronx-distributed-inference``. Save the contents of the below script to your favorite 
shell script file, for example, ``compile_model.sh`` and then run it.

Note that we are using the following features as described in
the tutorial for running 405B model :ref:`nxdi-trn2-llama3.1-405b-tutorial`

* Logical NeuronCores (LNC)
* Tensor parallelism (TP) on Trn2
* Optimized Kernels

The compiling script is similar to the one in part 1. 
Note that we have added the path for the draft model.

::
    
    # Replace this with the path where you downloaded and saved the model files.
    MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct-FP8-rescaled/"
    # Replace this with the path where you downloaded and saved the draft model files.
    DRAFT_MODEL_PATH="/home/ubuntu/models/Llama-3.2-1b-instruct/"    
    # This is where the compiled model (.pt file) and sharded checkpoints will be saved. The same path
    # should be used when launching vLLM server for inference.
    COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.1-405B-Instruct/"

    NUM_CORES=128
    TP_DEGREE=64
    LNC=2


    export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
    export NEURON_RT_NUM_CORES=$((NUM_CORES/NEURON_RT_VIRTUAL_CORE_SIZE))
    export NEURON_RT_EXEC_TIMEOUT=600 
    export XLA_HANDLE_SPECIAL_SCALAR=1
    export UNSAFE_FP8FNCAST=1

    inference_demo \
        -—model-type llama \
        -—task-type causal-lm \
        run \
            -—model-path $MODEL_PATH \
            -—compiled-model-path $COMPILED_MODEL_PATH \
            -—torch-dtype bfloat16 \
            -—start_rank_id 0 \
            -—local_ranks_size $TP_DEGREE \
            -—tp-degree $TP_DEGREE \
            -—batch-size 1 \
            -—max-context-length 12288 \
            -—seq-len 12800 \
            -—on-device-sampling \
            -—top-k 1 \
            -—fused-qkv \
            -—sequence-parallel-enabled \
            -—qkv-kernel-enabled \
            -—attn-kernel-enabled \
            -—mlp-kernel-enabled \
            -—cc-pipeline-tiling-factor 1 \
            -—draft-model-path $DRAFT_MODEL_PATH \
            -—enable-fused-speculation \
            -—speculation-length 7 \
            -—no-trace-tokengen-model \
            -—pad-token-id 2 \
            -—logical-neuron-cores $LNC \
            -—quantized-mlp-kernel-enabled \
            -—quantization-type per_channel_symmetric \
            -—rmsnorm-quantize-kernel-enabled \
            -—enable-bucketing \
            -—prompt "What is annapurna labs?" \
            -—context-encoding-buckets 1024 2048 4096 10240 12288 \
            -—token-generation-buckets 12800 2>&1 | tee compile_and_generate_log


The above script will compile a Neuron model with fused speculation, and also run generate on the example prompt specified with the ``-prompt`` flag. Please refer to :ref:`nxd-inference-api-guide` for more information on these ``inference_demo`` flags.

You can change this prompt to your prompt of choice. 
The script's output will be written into ``compile_and_generate_log``, a log file in the working directory. 

In this script, we also turn on some additional environment variables: ``XLA_HANDLE_SPECIAL_SCALAR`` and ``UNSAFE_FP8FNCAST`` to enable Neuron compiler to treat rescaled ``FP8FN`` weights as
``FP8_EXP4`` weights.

In addition, in the subsequent runs of this script, you can add a ``--skip-compile`` flag to skip 
the compiling step since the model is already compiled in the first run of the script. 
This will allow you to test the model with different prompts. 



Step 3: Start the Vllm server with the compiled Neuron model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After compiling the model, you can run the model using vLLM. Save the contents of the below script to another
shell script file, for example, ``start_vllm.sh`` and then run it.

::

    export NEURON_RT_INSPECT_ENABLE=0
    export NEURON_RT_VIRTUAL_CORE_SIZE=2
    export XLA_HANDLE_SPECIAL_SCALAR=1
    export UNSAFE_FP8FNCAST=1


    MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct-FP8-rescaled"
    DRAFT_MODEL_PATH="/home/ubuntu/models/Llama-3.2-1b-instruct"
    COMPILED_MODEL_PATH="/home/ubuntu/traced_models/Llama-3.1-405B-Instruct_fp8"


    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH
    VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
        -—model $MODEL_PATH \
        -—max-num-seqs 1 \
        -—max-model-len 12800 \
        -—tensor-parallel-size 64 \
        -—device neuron \
        -—speculative-max-model-len 12800 \
        -—speculative-model $DRAFT_MODEL_PATH \
        -—num-speculative-tokens 7 \
        -—use-v2-block-manager \
        -—override-neuron-config "{\"enable_fused_speculation\":true, \"quantized-mlp-kernel-enabled\":true, \"quantization-type\":\"per_channel_symmetric\"}" \
        -—port 8000 & PID=$!
    echo "vLLM server started with PID $PID"

Step 4: Measure performance using LLMPerf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After the above steps, the vllm server should be running. Before we can use the ``llmperf`` package, we need to make a few changes to its code. 
Follow :ref:`benchmarking with LLMPerf guide <llm_perf_patch_changes>` to apply the code changes.
    
We can now measure the performance using ``llmperf``. Run the following script with the modified ``llmperf`` package.

::

    # This should be the same path to which the model was downloaded (also used in the above steps).
    MODEL_PATH="/home/ubuntu/models/Llama-3.1-405B-Instruct-FP8-rescaled"
    # This is the name of directory where the test results will be saved.
    OUTPUT_PATH=llmperf-results-sonnets

    export OPENAI_API_BASE="http://localhost:8000/v1"
    export OPENAI_API_KEY="mock_key"

    python token_benchmark_ray.py \
        --model $MODEL_PATH \
        --mean-input-tokens 10000 \
        --stddev-input-tokens 0 \
        --mean-output-tokens 1500 \
        --stddev-output-tokens 0 \
        --num-concurrent-requests 1\
        --timeout 3600 \
        --max-num-completed-requests 50 \
        --additional-sampling-params '{}' \
        --results-dir $OUTPUT_PATH \
        --llm-api "openai"


The output for this llama-3.1-405B model run with fused speculation with fused spec is shown below. Please note that the numbers can slightly vary between runs but should be in the same order of magnitude. 

::

    Results for token benchmark for /home/ubuntu/models/Llama-3.1-405B-Instruct-FP8-rescaled queried with the openai api.

    inter_token_latency_s
        p25 = 0.008220573497974934
        p50 = 0.008265312568750231
        p75 = 0.008438719224417583
        p90 = 0.00848199803312309
        p95 = 0.008495625438929224
        p99 = 0.011143428944987235
        mean = 0.008419798457414533
        min = 0.008173695931987216
        max = 0.01364151847269386
        stddev = 0.0007612118573477839
    ttft_s
        p25 = 2.2543624382815324
        p50 = 2.254961202503182
        p75 = 2.2576071268413216
        p90 = 2.2596270388457924
        p95 = 2.260639927221928
        p99 = 2.2628143909573555
        mean = 2.256157155628316
        min = 2.2534945809748024
        max = 2.2629711360204965
        stddev = 0.0023667267664955545
    end_to_end_latency_s
        p25 = 14.586015026085079
        p50 = 14.65608573507052
        p75 = 14.91364526405232
        p90 = 14.977840351965279
        p95 = 15.000083449739032
        p99 = 18.969864878777866
        mean = 14.886235136194154
        min = 14.520539953839034
        max = 22.716861865017563
        stddev = 1.1415236552464672
    request_output_throughput_token_per_s
        p25 = 100.64608830743339
        p50 = 102.4148205461138
        p75 = 102.90679421801005
        p90 = 103.02201242683091
        p95 = 103.26614794565539
        p99 = 103.36118277211666
        mean = 101.22055373532301
        min = 66.0742671641385
        max = 103.37081160698546
        stddev = 5.19249551094185
    number_input_tokens
        p25 = 10000.0
        p50 = 10000.0
        p75 = 10000.0
        p90 = 10000.0
        p95 = 10000.0
        p99 = 10000.0
        mean = 10000.0
        min = 10000
        max = 10000
        stddev = 0.0
    number_output_tokens
        p25 = 1501.0
        p50 = 1501.0
        p75 = 1501.0
        p90 = 1501.0
        p95 = 1501.0
        p99 = 1501.0
        mean = 1501.0
        min = 1501
        max = 1501
        stddev = 0.0
    Number Of Errored Requests: 0
    Overall Output Throughput: 100.69986490153724
    Number Of Completed Requests: 50
    Completed Requests Per Minute: 4.025311055357918




Conclusion
-----------------------------------------------------------
As seen from the table below, draft model based fused speculative decoding and quantization significantly improved inference performance: TPOT reduced by 4x and output token throughput increased by 4x, while TTFT decreased from 2442 ms to 2255 ms compared to baseline without speculative decoding.
Please note that batch size of 1 is used in this tutorial for computing the below metrics.

.. csv-table::
   :file: llama405b_perf_comparison.csv
   :header-rows: 1