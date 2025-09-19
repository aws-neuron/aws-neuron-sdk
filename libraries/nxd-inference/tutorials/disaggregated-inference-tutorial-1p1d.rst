.. _nxdi-disaggregated-inference-1p1d-tutorial:

Tutorial: Static 1P1D Disaggregated Inference on Trn2 [BETA]
============================================================

Overview
~~~~~~~~

This tutorial will mainly cover how to run Disaggregated Inference (DI) 1P1D (1 prefill, 1 Decode) 
either on a single Trn2 instance (1P and 1D both are on same instance) or on 2 instances 
(1P and 1D are on separate instances). It will provide scripts that can setup both
single and multi instance workflows. Next, the tutorial will demonstrate how to benchmark DI. Finally,
we show how to benchmark non Disaggregated Inference (non-DI) continuous batching to compare results between DI vs. non-DI.

Read the :ref:`DI Developer Guide<nxdi-disaggregated-inference>` for more detailed information.

.. note::

   This tutorial was tested on trn2.48xlarge but its concepts are also be applicable to trn1.32xlarge.

Set up and connect trn2.48xlarge instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a prerequisite, this tutorial requires that you have one or two Trn2 instances
with Neuron SDK, Neuron vLLM and Elastic Fabric Adapter (EFA) enabled and installed. The Neuron Deep Learning AMI
comes with Neuron dependencies and EFA enabled and installed so it is the recommended
way to run this tutorial.

To set up a Trn2 instance using Deep Learning AMI with pre-installed Neuron SDK,
see :ref:`nxdi-setup`.

.. note::

   Disaggregated Inference is only supported on Neuron instances with EFA enabled (trn1.32xlarge or trn2.48xlarge).
   EFA is still required even when running single instance as the KV cache transfer happens through EFA.

If you choose to manually install NxD Inference follow the 
`EFA setup guide <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html>`_ to install and enable EFA.


If running multi-instance it is recommended to have shared storage between the two instances to avoid having
to download, compile and save scripts twice. For more details, see documentation on mounting 
`EFS <https://docs.aws.amazon.com/efs/latest/ug/mount-multiple-ec2-instances.html>`_ or 
`FSX <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/storage_fsx.html>`_ filesystems.

After setting up an instance, use SSH to connect to the Neuron instance(s) using the key pair that you
chose when you launched the instance.

After you are connected, activate the Python virtual environment that includes the Neuron SDK.

::

   source ~/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate

Install the latest release branch of vLLM from the AWS Neuron fork 
following the instructions in the :ref:`vLLM User Guide for NxD Inference<nxdi-vllm-user-guide>`.


Run ``pip list`` to verify that the Neuron SDK is installed.

::

   pip list | grep neuron

You should see Neuron packages including
``neuronx-distributed-inference`` and ``neuronx-cc`` and ``vllm``.

Download Dependencies
~~~~~~~~~~~~~~~~~~~~~

To use this sample, you must first download a `Llama-3.3-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`_ model checkpoint from Hugging Face
to a local path on the Trn2 instance. 
Note that you may need access from Meta for model download.
For more information, see
`Downloading models <https://huggingface.co/docs/hub/en/models-downloading>`_
in the Hugging Face documentation.


Compile the model
~~~~~~~~~~~~~~~~~

Compile the model for Neuron by using the following ``compile.sh`` script.

::

   #!/bin/bash
   # copy and paste me into a file called compile.sh
   # then run chmod +x compile.sh

   # Parse command line arguments
   while [[ $# -gt 0 ]]; do
      case $1 in
         --tp-degree)
               TP_DEGREE="$2"
               shift 2
               ;;
         --batch-size)
               BATCH_SIZE="$2"
               shift 2
               ;;
         --model-path)
               MODEL_PATH="$2"
               shift 2
               ;;
         *)
               echo "Unknown parameter: $1"
               echo "Usage: $0 --tp-degree <value> --batch-size <value> --model-path <path>"
               exit 1
               ;;
      esac
   done

   export COMPILED_MODEL_PATH="di_traced_model_tp${TP_DEGREE}_b${BATCH_SIZE}/"

   inference_demo \
      --model-type llama \
      --task-type causal-lm \
      run \
      --model-path $MODEL_PATH \
      --compiled-model-path $COMPILED_MODEL_PATH \
      --torch-dtype bfloat16 \
      --tp-degree $TP_DEGREE \
      --batch-size $BATCH_SIZE \
      --ctx-batch-size 1 \
      --tkg-batch-size $BATCH_SIZE \
      --is-continuous-batching \
      --max-context-length 8192 \
      --seq-len 8192 \
      --on-device-sampling \
      --fused-qkv \
      --global-topk 256 --dynamic \
      --top-k 50 --top-p 0.9 --temperature 0.7 \
      --do-sample \
      --sequence-parallel-enabled \
      --qkv-kernel-enabled \
      --attn-kernel-enabled \
      --mlp-kernel-enabled \
      --cc-pipeline-tiling-factor 1 \
      --pad-token-id 2 \
      --logical-neuron-cores 2 \
      --context-encoding-buckets 256 512 1024 2048 4096 8192 \
      --token-generation-buckets 512 1024 2048 4096 8192 \
      --apply-seq-ids-mask \
      --enable-bucketing \
      --prompt "test prompt" \
      --save-sharded-checkpoint \
      --attn-block-tkg-nki-kernel-enabled \
      --attn-block-tkg-nki-kernel-cache-update \
      --k-cache-transposed \
      --async-mode \
      --compile-only

The ``--apply-seq-ids-mask`` flag is required for DI because it
tells Neuron to only update the KV cache of the current sequence ID to ensure 
KV cache integrity, and ultimately, accuracy.

Multi-Instance
---------------
For multi-instance run: 

::

   ./compile.sh --tp-degree 64 --batch-size 4 --model-path path/to/your/downloaded/model

Single-Instance
---------------
For single-instance run: 

::

   ./compile.sh --tp-degree 32 --batch-size 4 --model-path path/to/your/downloaded/model

We compile for ``tp-degree=32`` because 1 prefill server will take up half 
of the Neuron Cores cores while the decode server will take up the other half.


Launch the Prefill and Decode Servers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a script called ``server.sh``, which you can use to launch prefill and
decode servers.

``NEURON_RT_ASYNC_SENDRECV_EXPERIMENTAL_ENABLED=1`` is currently required as DI is still in beta.
``NEURON_RT_ASYNC_SENDRECV_BOOTSTRAP_PORT=45645`` is required to tell the Neuron Runtime which port to use for KV Cache transfer communications.
``NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=2`` enables :ref:`Asynchronous Runtime Support<nxdi_async_mode_feature_guide>`

The ``KVTransferConfig`` provided to both servers on startup have key information.
``kv_connector=NeuronConnector`` lets vLLM know to use the Neuron implementation for KV cache transfer.
``kv_role=producer`` lets vLLM know that this server's job is to do prefill.
``kv_role=consumer`` lets vLLM know that this server's job is to do decode.
``neuron_core_offset=n`` lets vLLM know that the model is hosted starting on the nth Neuron Core.


::

   #!/bin/bash
   # copy and paste me into a file called server.sh
   # then run chmod +x server.sh

   #!/bin/bash

   # Parse command line arguments
   while [[ $# -gt 0 ]]; do
      case $1 in
         --tp-degree)
               TP_DEGREE="$2"
               shift 2
               ;;
         --batch-size)
               BATCH_SIZE="$2"
               shift 2
               ;;
         --model-path)
               MODEL_PATH="$2"
               shift 2
               ;;
         --compiled-model-path)
               COMPILED_MODEL_PATH="$2"
               shift 2
               ;;
         --neuron-send-ip)
               SEND_IP="$2"
               shift 2
               ;;
         --neuron-recv-ip)
               RECV_IP="$2"
               shift 2
               ;;
         *)
               echo "Unknown parameter: $1"
               echo "Usage: $0 --tp-degree <value> --batch-size <value> --model-path <path> \
                              --compiled-model-path <path> --send-ip <ip> --recv-ip <ip>"
               exit 1
               ;;
      esac
   done

   export NEURON_RT_ASYNC_SENDRECV_BOOTSTRAP_PORT=45645
   export NEURON_RT_ASYNC_SENDRECV_EXPERIMENTAL_ENABLED=1
   export NEURON_COMPILED_ARTIFACTS="$COMPILED_MODEL_PATH"
   export NEURON_SEND_IP="$SEND_IP"
   export NEURON_RECV_IP="$RECV_IP"
   export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=2

   if [ "$SEND" = "1" ]; then
      PORT=8100
      if [ "$SINGLE_INSTANCE" = "1" ]; then
         export NEURON_RT_VISIBLE_CORES=0-31
      fi
      TRANSFER_CONFIG='{
               "kv_connector":"NeuronConnector",
               "kv_buffer_device":"cpu",
               "kv_role":"kv_producer",
               "kv_rank":0,
               "kv_parallel_size":2,
               "kv_buffer_size":2e11,
               "kv_ip":"'"$NEURON_SEND_IP"'",
               "neuron_core_offset": 0
         }'
      
   else
      PORT=8200
      if [ "$SINGLE_INSTANCE" = "1" ]; then
         NC_OFFSET=32
         export NEURON_RT_VISIBLE_CORES=32-63
      else   
         NC_OFFSET=0
      fi
      TRANSFER_CONFIG='{
               "kv_connector":"NeuronConnector",
               "kv_buffer_device":"cpu",
               "kv_role":"kv_consumer",
               "kv_rank":1,
               "kv_parallel_size":2,
               "kv_buffer_size":2e11,
               "kv_ip":"'"$NEURON_SEND_IP"'",
               "neuron_core_offset": "'"$NC_OFFSET"'"
         }'
   fi

   python3 -m vllm.entrypoints.openai.api_server \
         --model "$MODEL_PATH" \
         --max-num-seqs "$BATCH_SIZE" \
         --max-model-len 8192 \
         --tensor-parallel-size "$TP_DEGREE" \
         --device neuron \
         --use-v2-block-manager \
         --override-neuron-config "{}" \
         --kv-transfer-config "$TRANSFER_CONFIG" \
         --port "$PORT"


You may need multiple terminals to run the following commands.

For multi-instance choose one instance to be your prefill instance and
one instance to be your decode instance. Get the IP addresses of them by running
``hostname -i`` and use them in the commands below. Single instance can use ``127.0.0.1``
as the IP address since prefill and decode always run on the same instance.

Multi-Instance
---------------

To launch a prefill server for multi-instance run: 

::

   SEND=1 ./server.sh --tp-degree 64 --batch-size 4 \
                      --model-path path/to/your/downloaded/model \
                      --compiled-model-path di_traced_model_tp64_b4/ \
                      --neuron-send-ip prefill_ip --neuron-recv-ip decode_ip

To launch a decode server open up a new tab and run: 

::

   ./server.sh --tp-degree 64 --batch-size 4 \
               --model-path path/to/your/downloaded/model \
               --compiled-model-path di_traced_model_tp64_b4/  \
               --neuron-send-ip prefill_ip --neuron-recv-ip decode_ip


Single-Instance
---------------
To launch a prefill server for single-instance run: 

::

   SEND=1 SINGLE_INSTANCE=1 ./server.sh --tp-degree 32 --batch-size 4 \
                                        --model-path path/to/your/downloaded/model \
                                        --compiled-model-path di_traced_model_tp32_b4/ \
                                        --neuron-send-ip 127.0.0.1 --neuron-recv-ip 127.0.0.1


To launch a decode server open up a new tab and run: 

::

   SINGLE_INSTANCE=1 ./server.sh --tp-degree 32 --batch-size 4 \
                                 --model-path path/to/your/downloaded/model \
                                 --compiled-model-path di_traced_model_tp32_b4/ \
                                 --neuron-send-ip 127.0.0.1 --neuron-recv-ip 127.0.0.1



When you see the line ``INFO:     Uvicorn running on http://0.0.0.0:8100 (Press CTRL+C to quit)``
on your prefill and decode server tabs your servers are ready.

Launch a Router (Proxy Server)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both servers need to receive a request to run inference. The component that does this job is called the 
router as mentioned in :ref:`DI Developer Guide<nxdi-disaggregated-inference>`.
We offer an implementation of a router called the ``neuron-proxy-server``.
The ``neuron-proxy-server`` is an entrypoint in our fork of vLLM which launches a proxy server that
will take a request and forward it to both the prefill and decode servers. It will 
then capture their responses and format them back to the user. 

The implementation of the neuron-proxy-server can be found 
`here <https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.24-vllm-v0.7.2/vllm/neuron_immediate_first_token_proxy_server.py>`_.


For multi-instance run the router as another process on your prefill instance. 
For single-instance run the router as another process on your Trn2.

A router can run on any instance that has a connection to both the prefill and decode nodes.
For multi-instance 1P1D, it makes the most sense to have the router on the prefill node to reduce network latency.

Launch the proxy server by running:

::

   pip install quart # only install one time
   neuron-proxy-server --prefill-ip your_prefill_ip --decode-ip your_decode_ip --prefill-port 8100 --decode-port 8200

The proxy server is ready when you see the line ``INFO:hypercorn.error:Running on http://127.0.0.1:8000 (CTRL + C to quit)``

Test the DI Setup
~~~~~~~~~~~~~~~~~

Run a sanity check to see if you DI setup is working by sending a curl request to the ``neuron-proxy-server``:

::

   curl -s http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
      "model": "path/to/your/downloaded/model",
      "prompt": ["a tornado is a"],
      "max_tokens": 10,
      "temperature": 0
      }'

A successful response looks like:
``{"id": ... :[{"index":0,"text":" rotating column of air that forms during severe thunderstorms" ... }``

The ``neuron-proxy-server`` also supports the streaming of responses. It can be tested by:

::

   curl -s http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
      "model": "path/to/your/downloaded/model",
      "prompt": ["a tornado is a"],
      "max_tokens": 10,
      "temperature": 0,
      "stream": true
      }'


Benchmark the DI Setup
~~~~~~~~~~~~~~~~~~~~~~

Install LLMPerf
---------------

We will use `LLMPerf <https://github.com/ray-project/llmperf>`_ to measure the performance.

LLMPerf will send requests to the ``neuron-proxy-server`` and capture data including Time To First Token,
Inter Token Latency and throughput.

Install llmperf into the ``aws_neuronx_venv_pytorch_2_7_nxd_inference`` virtual environment.

For multi-instance LLMperf is only required to be installed on the prefill instance where you will run benchmarking.

::

    git clone https://github.com/ray-project/llmperf.git
    cd llmperf
    pip install -e .    

Once you have installed LLMPerf, apply the ``neuron_perf.patch`` as described in :ref:`llm-inference-benchmarking`. 

Next use the ``llmperf.sh`` script to run benchmarks.

::

   #!/bin/bash
   # copy and paste me into a file called llmperf.sh
   # then run chmod +x llmperf.sh

   # Set environment variables
   export OPENAI_API_BASE="http://localhost:8000/v1"
   export OPENAI_API_KEY="mock_key"

   python llmperf/token_benchmark_ray.py \
      --model=$MODEL_PATH \
      --tokenizer=$MODEL_PATH \
      --mean-input-tokens=1024 \
      --stddev-input-tokens=0\
      --mean-output-tokens=100 \
      --stddev-output-tokens=10 \
      --max-num-completed-requests=200 \
      --timeout=1720000 \
      --num-concurrent-requests=4 \
      --results-dir=llmperf_results \
      --llm-api=openai \
      --additional-sampling-params "{\"top_k\": 50, \"top_p\": 0.9, \"temperature\": 0.7}"

Since the ``llmperf.sh`` script sends requests to localhost, it should be run on the same instance
the router is running on.

In multi-instance that means as a separate process on your prefill instance.
For single instance that means a separate process on your Trn2.

::

   MODEL_PATH=path/to/your/downloaded/model ./llmperf.sh 

This will run a total of 200 requests and your final output should have the line:
``Completed Requests Per Minute: xx.xxxxxxx``. Scroll up to see metrics such as
Inter Token Latency and Time To First Token.


Benchmark a Non-DI Continuous Batching Setup for Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compare Disaggregated Inference against non-DI continuous batching 
we will run benchmarks without Disaggregated Inference.

First kill all DI servers. Then kill the ``neuron-proxy-server``.

We will run the same compiled model as a singular server for non-DI benchmarks.
For single instance non-DI benchmarking we will start one TP=32 server. For multi-instance non-DI 
benchmarking we will start one TP=64 server. This means you do not need your second (decode) instance for this step.
Latency can be compared directly in DI vs non-DI benchmarks. You might need to adjust the throughput related 
metrics based on number of instances to compare apples-to-apples between DI and non-D1. 
In this case, Non-DI throughput should be doubled before comparing with DI as the non-DI benchmark uses half the amount of hardware.

Use the ``baseline_server.sh`` to launch a vLLM server without DI.

::

   #!/bin/bash
   # copy and paste me into a file called baseline_server.sh
   # then run chmod +x baseline_server.sh

   #!/bin/bash

   # Parse command line arguments
   while [[ $# -gt 0 ]]; do
      case $1 in
         --tp-degree)
               TP_DEGREE="$2"
               shift 2
               ;;
         --batch-size)
               BATCH_SIZE="$2"
               shift 2
               ;;
         --model-path)
               MODEL_PATH="$2"
               shift 2
               ;;
         --compiled-model-path)
               COMPILED_MODEL_PATH="$2"
               shift 2
               ;;
         *)  
               echo "Unknown parameter: $1"
               echo "Usage: $0 --tp-degree <value> --batch-size <value> --model-path <path> \
                              --compiled-model-path <path>"
               exit 1
               ;;
      esac
   done

   export NEURON_COMPILED_ARTIFACTS="$COMPILED_MODEL_PATH"
   export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=2

   if [ "$SINGLE_INSTANCE" = "1" ]; then
      NEURON_RT_VISIBLE_CORES=0-31
   fi

   python3 -m vllm.entrypoints.openai.api_server \
         --model "$MODEL_PATH" \
         --max-num-seqs "$BATCH_SIZE" \
         --max-model-len 8192 \
         --tensor-parallel-size "$TP_DEGREE" \
         --device neuron \
         --use-v2-block-manager \
         --override-neuron-config "{}" \
         --port 8000


Multi-Instance
---------------
Launch for multi-instance with:

::
   
   ./baseline_server.sh --tp-degree 64 --batch-size 4 \
                        --model-path path/to/your/downloaded/model \
                        --compiled-model-path di_traced_model_tp64_b4/


Single-Instance
---------------
Launch for single-instance with:

::
   
   SINGLE_INSTANCE=1 ./baseline_server.sh --tp-degree 32 --batch-size 4 \
                                          --model-path path/to/your/downloaded/model \
                                          --compiled-model-path di_traced_model_tp32_b4/

Now we have a server launched with the same underlying model but with DI turned off.

Then on the same instance run llmperf which will now directly send requests to the server
instead of going through a proxy:

::

   MODEL_PATH=path/to/your/downloaded_model ./llmperf.sh 

This will run a total of 200 requests and your final output should have the line:
``Completed Requests Per Minute: xx.xxxxxxx``. Scroll up to see metrics such as
Inter Token Latency and Time To First Token.


Known Issues
~~~~~~~~~~~~

``ENC:kv_store_acquire_file_lock   Failed to open kv store server lock file Permission denied`` 
usually means that another user on the system ran a DI workload and left behind a lock file
that the current user does not have access to. The solution is to delete ``/tmp/nrt_kv_store_server.lock`` file.