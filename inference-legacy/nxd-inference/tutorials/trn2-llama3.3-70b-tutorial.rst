.. _nxdi-trn2-llama3.3-70b-tutorial:

Tutorial: Using Speculative Decoding to improve Llama-3.3-70B inference performance on Trn2 instances
=======================================================================================================

NeuronX Distributed (NxD) Inference allows you to deploy Llama3.3 70B on
a single Trn2 or Trn1 instance. This tutorial provides a step-by-step
guide to deploy Llama3.3 70B on a Trn2 instance using two different configurations, one without
speculative decoding and the other with draft model based speculative decoding enabled
(with Llama-3.2 1B as the draft model).
We will also measure performance by running a load test using LLMPerf
and compare key metrics between the two configurations.
While this tutorial uses batch size 1 for demonstration purposes, the model configuration provides support for batch sizes up to 4.

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
available in the AWS Neuron fork of the vLLM GitHub repository. Install the latest release branch of vLLM from the AWS Neuron fork 
following instructions in the :ref:`vLLM User Guide for NxD Inference<nxdi-vllm-user-guide-v1>`.

In this tutorial, you will use `llmperf <https://github.com/ray-project/llmperf>`_ to measure the performance.
We will use the `load test <https://github.com/ray-project/llmperf?tab=readme-ov-file#load-test>`_ feature of LLMPerf and measure the performance for accepting
10,000 tokens as input and generating 1500 tokens as output.
Install llmperf into the virtual environment.

::

    git clone https://github.com/ray-project/llmperf.git
    cd llmperf
    pip install -e . 


Download models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To use this sample, you must first download a 70B model checkpoint from Hugging Face
to a local path on the Trn2 instance. For more information, see
`Downloading models <https://huggingface.co/docs/hub/en/models-downloading>`__
in the Hugging Face documentation. You can download and use `meta-llama/Llama-3.3-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`__
for this tutorial.

Since we will be using Speculative Decoding in the second configuration, 
you will also need a draft model checkpoint. You can download and use `meta-llama/Llama-3.2-1B-Instruct <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>`__.

.. note::

    NxD Inference supports batch sizes up to 4 for this model configuration. To determine the optimal batch size for your specific use case, we recommend incrementally testing batch sizes from 1 to 4 while monitoring your application's performance metrics.

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

* Logical NeuronCore Configuration (LNC)
* Tensor parallelism (TP) on Trn2
* Optimized Kernels

The script compiles the model and runs generation on the given input prompt.
Note the path we used to save the compiled model. This path should be used
when launching vLLM server for inference so that the compiled model can be loaded without recompilation.
Please refer to :ref:`nxd-inference-api-guide` for more information on these ``inference_demo`` flags.


.. note::

    Known issue: Using kernels with bucket length of 1024 or less may lead to ``Numerical Error`` in inference.

    ::

        RuntimeError: Failed to execute the model status=1003 message=Numerical Error


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
            --enable-bucketing \
            --context-encoding-buckets 2048 4096 8192 12288 \
	        --token-generation-buckets 2048 4096 8192 12800 \
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
    VLLM_RPC_TIMEOUT=100000 vllm serve \
        --model $MODEL_PATH \
        --max-num-seqs 1 \
        --max-model-len 12800 \
        --tensor-parallel-size 64 \
        --device neuron \
        --use-v2-block-manager \
        --override-neuron-config "{\"on_device_sampling_config\": {\"do_sample\": true}, \"skip_warmup\": true}" \
        --port 8000 &
    PID=$!
    echo "vLLM server started with PID $PID"

Step 3: Measure performance using LLMPerf
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the above steps, the vllm server should be running. 
You can now measure the performance using LLMPerf. Before we can use the ``llmperf`` package, we need to make a few changes to its code. 
Follow :ref:`benchmarking with LLMPerf guide <llm_perf_patch_changes>` to apply the code changes.


Below is a sample shell script to run LLMPerf. To provide the model with 10000 tokens as input and generate 1500 tokens as output on average,
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
        p25 = 0.01964743386193489
        p50 = 0.01965969146322459
        p75 = 0.019672998415771872
        p90 = 0.01969826815724373
        p95 = 0.019810569172135244
        p99 = 0.020350346909947692
        mean = 0.01969182239660784
        min = 0.0196275211258056
        max = 0.020702997242410977
        stddev = 0.00015700734112322808
    ttft_s
        p25 = 0.8109508841298521
        p50 = 0.8142827898263931
        p75 = 30.46490489714779
        p90 = 30.513100237119943
        p95 = 30.521608413150535
        p99 = 48.876512633068415
        mean = 11.503728219866753
        min = 0.8080519903451204
        max = 66.4881955627352
        stddev = 15.692731777293613
    end_to_end_latency_s
        p25 = 30.296781020238996
        p50 = 30.326033774763346
        p75 = 59.9560666854959
        p90 = 60.001504834741354
        p95 = 60.028880204679446
        p99 = 79.1842334462329
        mean = 41.04328096391633
        min = 30.265212223865092
        max = 97.54387667682022
        stddev = 15.796048923358924
    request_output_throughput_token_per_s
        p25 = 25.044969421803977
        p50 = 49.49542857484997
        p75 = 49.543217224244
        p90 = 49.583184869985566
        p95 = 49.58588728343319
        p99 = 49.592597790896676
        mean = 40.91042833304163
        min = 15.387946954098137
        max = 49.59489426003143
        stddev = 11.825984480587056
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
    Overall Output Throughput: 36.55567822866449
    Number Of Completed Requests: 50
    Completed Requests Per Minute: 1.4612140207588533


Scenario 2: Run Llama3.3 70B on Trn2 with Speculative Decoding
--------------------------------------------------------------
In this scenario, you will run Llama3.3 70B on Trn2 with Speculative Decoding.
Specifically, we will use the below variations from the supported variants as described in
:ref:`nxd-speculative-decoding`

* Speculative Decoding with Llama-3.2-1B as the draft model :ref:`nxd-vanilla-speculative-decoding`
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

Please refer to :ref:`nxd-inference-api-guide` for more information on these ``inference_demo`` flags.
The complete script to compile the model for this configuration is shown below:


.. note::

    Known issue: Using kernels with bucket length of 1024 or less may lead to ``Numerical Error`` in inference.

    ::

        RuntimeError: Failed to execute the model status=1003 message=Numerical Error


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
            --pad-token-id 2 \
            --enable-bucketing \
            --context-encoding-buckets 2048 4096 8192 12288 \
	        --token-generation-buckets 2048 4096 8192 12800 \
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
    VLLM_RPC_TIMEOUT=100000 vllm serve \
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
The script to measure the performance using LLMPerf is same as the one used in the first scenario. Before we can use the ``llmperf`` package, we need to make a few changes to its code. 
Follow :ref:`benchmarking with LLMPerf guide <llm_perf_patch_changes>` to apply the code changes.

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
        p25 = 0.0053349758717231455
        p50 = 0.005386366705410183
        p75 = 0.005441084293027719
        p90 = 0.005499971026182175
        p95 = 0.005520176071580499
        p99 = 0.005911254031351169
        mean = 0.00540780140378178
        min = 0.005264532127728065
        max = 0.006265544256816307
        stddev = 0.00013951778334019935
    ttft_s
        p25 = 0.8693495176266879
        p50 = 0.870149074587971
        p75 = 0.8710820493288338
        p90 = 0.8725412225350737
        p95 = 0.8742059985175729
        p99 = 36.83790613239617
        mean = 2.280795605443418
        min = 0.8676468348130584
        max = 71.38881027325988
        stddev = 9.97280539681726
    end_to_end_latency_s
        p25 = 8.873123338911682
        p50 = 8.950916013680398
        p75 = 9.030085149221122
        p90 = 9.120021602977067
        p95 = 9.150626054406166
        p99 = 45.70815015356973
        mean = 10.393093119114637
        min = 8.766328778117895
        max = 80.78758085798472
        stddev = 10.158917239418473
    request_output_throughput_token_per_s
        p25 = 166.22213179149702
        p50 = 167.69243252025473
        p75 = 169.16253286110174
        p90 = 169.52692450439133
        p95 = 169.81518762962915
        p99 = 170.85438941846397
        mean = 164.631719334475
        min = 18.579588397857652
        max = 171.2233293995004
        stddev = 21.152953887186314
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
    Overall Output Throughput: 144.17136914316023
    Number Of Completed Requests: 50
    Completed Requests Per Minute: 5.76285918335928

Conclusion
-----------
As seen in the table below, TPOT reduced by 3.6x and output token throughput increased by 4x when using speculative decoding with draft model combined with fused speculative decoding,
compared to baseline without speculative decoding. Please note that batch size of 1 is used in this tutorial for computing the below metrics.


.. csv-table::
   :file: llama70b_perf_comparison.csv
   :header-rows: 1

