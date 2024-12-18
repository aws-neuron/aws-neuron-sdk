.. _nxdi-trn2-llama3.3-70b-tutorial:

Tutorial: Using Speculative Decoding to improve Llama-3.3-70B inference performance on Trn2 instances
=======================================================================================================

NeuronX Distributed (NxD) Inference allows you to deploy Llama3.3 70B on
a single Trn2 or Trn1 instance. This tutorial provides a step-by-step
guide to deploy Llama3.3 70B on a Trn2 instance using two different configurations, one without
speculative decoding and the other with vanilla speculative decoding enabled
(with Llama-3.2 1B as the draft model).
We will also measure performance by running a load test using LLMPerf
and compare key metrics between the two configurations.
We will use a batch size of 1 throughout the tutorial.

.. contents:: Table of contents
   :local:
   :depth: 2

Prerequisites:
---------------
Set up and connect to a Trn2.48xlarge instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~
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


In this tutorial, you will use `llmperf <https://github.com/ray-project/llmperf>`_ to measure the performance.
We will use the `load test <https://github.com/ray-project/llmperf?tab=readme-ov-file#load-test>`_ feature of LLMPerf and measure the performance for accepting
10,000 tokens as input and generating 1500 tokens as output.
Install llmperf into the virtual environment.

::

    git clone https://github.com/ray-project/llmperf.git
    cd llmperf
    pip install -e . 


Download Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To use this sample, you must first download a 70B model checkpoint from Hugging Face
to a local path on the Trn2 instance. For more information, see
`Downloading models <https://huggingface.co/docs/hub/en/models-downloading>`__
in the Hugging Face documentation. You can download and use `meta-llama/Llama-3.3-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`__
for this tutorial.

Since we will be using Speculative Decoding in the second configuration, 
you will also need a draft model checkpoint. You can download and use `meta-llama/Llama-3.2-1B-Instruct <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`__.


Scenario 1: Run Llama3.3 70B on Trn2
-------------------------------------
In this scenario, you will run Llama3.3 70B on Trn2 without Speculative Decoding
using bfloat16 precision.

Step 1: Compile the model
~~~~~~~~~~~~~~~~~~~~~~~~~~
We will first compile and run generation on a sample prompt using a command
installed by ``neuronx-distributed-inference``. Save the contents of the below script to your favorite 
shell script file, for example, ``compile_model.sh`` and then run it.

Note that we are using the following features as described in
the tutorial for running 405B model :ref:`nxdi-trn2-llama3.1-405b-tutorial`

* Logical NeuronCores (LNC)
* Tensor parallelism (TP) on Trn2
* Optimized Kernels

The script compiles the model and runs generation on the given input prompt.
Note the path we used to save the compiled model. This path should be used
when launching vLLM server for inference so that the compiled model can be loaded without recompilation.

::

    # Replace this with the path where you downloaded and saved the model files.
    MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
    # This is where the compiled model will be saved. The same path
    # should be used when launching vLLM server for inference.
    COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.3-70B-Instruct/"

    NUM_CORES=128
    TP_DEGREE=64
    LNC=2

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



Step 2: Run the model using vLLM 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After compiling the model, you can run the model using vLLM. Save the contents of the below script to another
shell script file, for example, ``start_vllm.sh`` and then run it.

::

    export NEURON_RT_INSPECT_ENABLE=0 
    export NEURON_RT_VIRTUAL_CORE_SIZE=2

    # These should be the same paths used when compiling the model.
    MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
    COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.3-70B-Instruct/"

    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH
    VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --max-num-seqs 1 \
        --max-model-len 12800 \
        --tensor-parallel-size 64 \
        --device neuron \
        --use-v2-block-manager \
        --override-neuron-config "{\"on_device_sampling_config\": {\"do_sample\": true}}" \
        --port 8000 &
    PID=$!
    echo "vLLM server started with PID $PID"

Step 3: Measure performance using LLMPerf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the above steps, the vllm server should be running. 
You can now measure the performance using LLMPerf. Below is a sample shell script to run LLMPerf.

To provide the model with 10000 tokens as input and generate 1500 tokens as output on average,
we use the following parameters from LLMPerf:

::

    --mean-input-tokens 10000 \
    --mean-output-tokens 1500 \


More information about several arguments used in the script can be found in the 
`llmperf open source code <https://github.com/ray-project/llmperf/blob/main/token_benchmark_ray.py>`_.

::

    # This should be the same path to which the model was downloaded (also used in the above steps).
    MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
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
        --tokenizer $MODEL_PATH \
        --additional-sampling-params '{}' \
        --results-dir $OUTPUT_PATH \
        --llm-api "openai"

A sample output from the above script is shown below:

::

    Results for token benchmark for /home/ubuntu/models/Llama-3.3-70B-Instruct/ queried with the openai api.

    inter_token_latency_s
        p25 = 0.019814822451599563
        p50 = 0.019832020386432607
        p75 = 0.01984963524178602
        p90 = 0.01985819646107654
        p95 = 0.019871625061845408
        p99 = 0.02061684579865075
        mean = 0.019860720100291072
        min = 0.019783137260004878
        max = 0.02133107245961825
        stddev = 0.00021329793592557677
    ttft_s
        p25 = 0.5723962930496782
        p50 = 0.5756837059743702
        p75 = 0.5782957510091364
        p90 = 0.5809791539330036
        p95 = 0.5902622325113043
        p99 = 25.081049750000144
        mean = 1.536737611917779
        min = 0.5699969907291234
        max = 48.603518176823854
        stddev = 6.79209192602991
    end_to_end_latency_s
        p25 = 30.299682187382132
        p50 = 30.3268030770123
        p75 = 30.348097508074716
        p90 = 30.367999098449946
        p95 = 30.383213692484425
        p99 = 56.00018264657342
        mean = 31.32914203199558
        min = 30.249366438016295
        max = 80.60140019096434
        stddev = 7.110435685337879
    request_output_throughput_token_per_s
        p25 = 49.45944310946372
        p50 = 49.494171785795885
        p75 = 49.538473422552784
        p90 = 49.56724071383475
        p95 = 49.58726816215459
        p99 = 49.61382242393379
        mean = 48.88194397874459
        min = 18.62250527216358
        max = 49.62087398014387
        stddev = 4.367006858791291
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
    Overall Output Throughput: 47.479805693322504
    Number Of Completed Requests: 50
    Completed Requests Per Minute: 1.897926943104164


Scenario 2: Run Llama3.3 70B on Trn2 with Speculative Decoding
--------------------------------------------------------------
In this scenario, you will run Llama3.3 70B on Trn2 with Speculative Decoding.
Specifically, we will use the below variations from the supported variants as described in
:ref:`nxd-speculative-decoding`

* Vanilla Speculative Decoding with Llama-3.2-1B as the draft model :ref:`nxd-vanilla-speculative-decoding`
* Fused Speculation for improved performance :ref:`nxd-fused-speculative-decoding`

Step 1: Compile the model
~~~~~~~~~~~~~~~~~~~~~~~~~~
When compiling the model to use speculative decoding, you need to provide 
a draft model checkpoint and a few additional parameters to the ``inference_demo`` command.

For a quick review, here are the additional arguments provided:

::

            --draft-model-path $DRAFT_MODEL_PATH \
            --enable-fused-speculation \
            --speculation-length 7 \
            --no-trace-tokengen-model \

The complete script to compile the model for this configuration is shown below:

::

    # This is the same path as in the previous scenario.
    MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
    # This is the path where the draft model is downaloded and saved.
    DRAFT_MODEL_PATH="/home/ubuntu/models/Llama-3.2-1B-Instruct/"
    # As in the previous scenario, this is where the compiled model will be saved.
    COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.3-70B-Instruct/"

    NUM_CORES=128
    TP_DEGREE=64
    LNC=2

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
            --batch-size 1 \
            --max-context-length 12288 \
            --seq-len 12800 \
            --on-device-sampling \
            --top-k 1 \
            --fused-qkv \
            --sequence-parallel-enabled \
            --qkv-kernel-enabled \
            --attn-kernel-enabled \
            --mlp-kernel-enabled \
            --cc-pipeline-tiling-factor 1 \
            --draft-model-path $DRAFT_MODEL_PATH \
            --enable-fused-speculation \
            --speculation-length 7 \
            --no-trace-tokengen-model \
            --pad-token-id 2 \
            --logical-neuron-cores $LNC \
            --enable-bucketing \
            --prompt "What is annapurna labs?" 2>&1 | tee log

Step 2: Run the model using vLLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Similar to compiling the model, we need to specify parameters specific to 
speculative decoding when running the model using vLLM.

For a quick glance, these are the parameters that are different for 
running vLLM server with model compiled using speculative decoding:

::

            --speculative-max-model-len 12800 \
            --speculative-model $DRAFT_MODEL_PATH \
            --num-speculative-tokens 7 \
            --override-neuron-config "{\"enable_fused_speculation\":true}" \
            
Here is the complete script to run the model using vLLM with speculative decoding:

::

    export NEURON_RT_INSPECT_ENABLE=0 
    export NEURON_RT_VIRTUAL_CORE_SIZE=2

    # These should be the same paths used when compiling the model.
    MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
    DRAFT_MODEL_PATH="/home/ubuntu/models/Llama-3.2-1B-Instruct/"
    COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.3-70B-Instruct/"

    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH
    VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --max-num-seqs 1 \
        --max-model-len 12800 \
        --tensor-parallel-size 64 \
        --device neuron \
        --speculative-max-model-len 12800 \
        --speculative-model $DRAFT_MODEL_PATH \
        --num-speculative-tokens 7 \
        --use-v2-block-manager \
        --override-neuron-config "{\"enable_fused_speculation\":true}" \
        --port 8000 &
    PID=$!
    echo PID=$PID
    echo "vLLM server started with PID $PID"

Step 3: Measure performance using LLMPerf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The script to measure the performance using LLMPerf is same as the one used in the first scenario.

For convenience, here's the script once again:

::

    # This should be the same path to which the model was downloaded (also used in the above steps).
    MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
    # This is the name of directory where the test results will be saved. Use a different name for this scenario.
    OUTPUT_PATH=llmperf-results-sonnets-speculative

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
        --tokenizer $MODEL_PATH \
        --additional-sampling-params '{}' \
        --results-dir $OUTPUT_PATH \
        --llm-api "openai"

A sample output from the above script is shown below:

::

    Results for token benchmark for /home/ubuntu/models/Llama-3.3-70B-Instruct/ queried with the openai api.

    inter_token_latency_s
        p25 = 0.005602470060500006
        p50 = 0.005631429936426381
        p75 = 0.005658711920414741
        p90 = 0.005684578440866123
        p95 = 0.005713548544825365
        p99 = 0.006577651818428922
        mean = 0.00566579679981318
        min = 0.005526937821879983
        max = 0.007301821645038824
        stddev = 0.00024078117608748752
    ttft_s
        p25 = 0.6115399743430316
        p50 = 0.6124009389895946
        p75 = 0.6139737505000085
        p90 = 0.6173051572870463
        p95 = 0.6198122691828758
        p99 = 27.979491891143738
        mean = 1.6862781716044992
        min = 0.6109044901095331
        max = 54.25300705060363
        stddev = 7.585774430901768
    end_to_end_latency_s
        p25 = 9.01934456313029
        p50 = 9.060323748504743
        p75 = 9.101599234272726
        p90 = 9.139903657091782
        p95 = 9.185261113895104
        p99 = 37.84166024243913
        mean = 10.185618488714098
        min = 8.901652369182557
        max = 65.20616552000865
        stddev = 7.940203139754607
    request_output_throughput_token_per_s
        p25 = 164.91607258958172
        p50 = 165.66736932540198
        p75 = 166.4200873246242
        p90 = 167.1482207218583
        p95 = 167.262433556426
        p99 = 167.99197421975765
        mean = 162.7853177282462
        min = 23.019295614605873
        max = 168.6203794248862
        stddev = 20.210056581750347
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
        p99 = 1502.02
        mean = 1501.04
        min = 1501
        max = 1503
        stddev = 0.282842712474619
    Number Of Errored Requests: 0
    Overall Output Throughput: 143.3838447738045
    Number Of Completed Requests: 50
    Completed Requests Per Minute: 5.7313800341285175

Conclusion
-----------
As seen in the table below, when speculative decoding with
draft model (combined with fused speculative decoding) is used,
TPOT improves by about 72% and there is a 3x improvement in output
token throughput compared to when no speculative decoding is used.

.. csv-table::
   :file: llama70b_perf_comparison.csv
   :header-rows: 1

