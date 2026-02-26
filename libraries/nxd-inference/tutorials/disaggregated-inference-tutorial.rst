.. _nxdi-disaggregated-inference-tutorial:

Tutorial: Disaggregated Inference [BETA]
================================================

Overview
~~~~~~~~

This tutorial shows how to run Disaggregated Inference (DI) using prefill and decode vLLM workers. You'll learn how to set up both worker types and scale from a basic 1P1D setup to larger configurations. The guide includes benchmarks that show how DI improves performance compared to standard inference, especially for long input sequences.

DI splits work between prefill workers and decode workers. Each worker needs:

* A Trn1 or Trn2 instance 
* Neuron SDK
* A supported vLLM version
* Elastic Fabric Adapter (EFA)

DI also needs a proxy server to manage traffic between workers and an etcd service for worker registration. You can run these on a basic EC2 instance like an M-series.

For more details, see the :ref:`DI Developer Guide<nxdi-disaggregated-inference>`.

.. note::

  This tutorial works with trn2.48xlarge and trn1.32xlarge instances.

Before You Begin  
~~~~~~~~~~~~~~~~

You need:

* A Trn1 or Trn2 instance with Neuron SDK, Neuron vLLM, and EFA enabled (see :ref:`nxdi-setup`)
* An m5.xlarge instance with Ubuntu or Amazon Linux

.. note::
   DI only works on Neuron instances with EFA (trn1.32xlarge or trn2.48xlarge). You need EFA even for single-instance setups.

.. tip::
   Use the AWS Neuron Deep Learning Container (DLC) to avoid manual setup. We'll use the vllm-inference-neuronx DLC in this guide.

Select and Compile Your Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DI works best with large models that have billions of parameters. We'll use ``meta-llama/Llama-3.3-70B-Instruct`` as an example. First, compile your model following the :ref:`nxdi-trn2-llama3.3-70b-dp-tutorial` guide. Make sure to set the correct input shapes and tensor parallelism.

Set Up the etcd Server
~~~~~~~~~~~~~~~~~~~~~~~

1. Connect to your EC2 proxy instance using SSH or Session Manager
2. Run these commands:

.. code-block:: bash

   sudo su - ubuntu
   HOST_IP=$(hostname -i | awk '{print $1}')

   # Remove old containers
   docker rm -f etcd proxy 2>/dev/null || true

   # Start etcd
   docker run -d \
     --name etcd \
     --shm-size=10g \
     --privileged \
     -p 8989:8989 \
     -e ETCD_IP=$HOST_IP \
     ubuntu:22.04 \
     bash -c "apt-get update && apt-get install -y etcd && \
              exec etcd \
                --data-dir=/etcd-data \
                --listen-client-urls=http://0.0.0.0:8989 \
                --advertise-client-urls=http://\$ETCD_IP:8989 \
                --listen-peer-urls=http://127.0.0.1:21323 \
                --initial-advertise-peer-urls=http://127.0.0.1:21323 \
                --initial-cluster=default=http://127.0.0.1:21323 \
                --name=default"

   # Start proxy
   docker run -d \
     --name proxy \
     --shm-size=10g \
     --privileged \
     -p 8000:8000 \
     -e ETCD_IP=$HOST_IP \
     -e ETCD_PORT=8989 \
     public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py310-sdk2.25.1-ubuntu22.04 \
     bash -c "exec neuron-proxy-server --etcd \$ETCD_IP:\$ETCD_PORT"

Verify both services are running:

.. code-block:: bash

   docker ps

Start the Prefill Server
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run these commands:

.. code-block:: bash

   sudo su - ubuntu
   export MODEL="meta-llama/Llama-3.3-70B-Instruct"
   export VLLM_BATCH=8
   export MAX_LEN=8192
   export ETCD="${HOST_IP}:8989"
   export PORT=8000

   # Remove old container
   docker rm -f prefill-vllm-server1 2>/dev/null || true

   # Start prefill server
   docker run -d \
     --name prefill-vllm-server1 \
     --privileged \
     --device /dev/infiniband/uverbs0 \
     --shm-size=10g \
     -p ${PORT}:${PORT} \
     -e MODEL \
     -e VLLM_BATCH \
     -e MAX_LEN \
     -e ETCD \
     -e PORT \
     public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py310-sdk2.25.1-ubuntu22.04 \
     bash -c "exec python3 -m vllm.entrypoints.openai.api_server \
       --model \$MODEL \
       --max-num-seqs \$VLLM_BATCH \
       --max-model-len \$MAX_LEN \
       --tensor-parallel-size 64 \
       --device neuron \
       --speculative-max-model-len \$MAX_LEN \
       --override-neuron-config '{}' \
       --kv-transfer-config '{\"kv_connector\":\"NeuronConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":2e11,\"etcd\":\"\$ETCD\"}' \
       --port \$PORT"

Note: The prefill server uses ``kv_role:kv_producer`` in its configuration.

Start the Decode Server
~~~~~~~~~~~~~~~~~~~~~~~~~

Run similar commands for the decode server:

.. code-block:: bash

   sudo su - ubuntu
   export MODEL="meta-llama/Llama-3.3-70B-Instruct"
   export VLLM_BATCH=8
   export MAX_LEN=8192
   export ETCD="${HOST_IP}:8989"
   export PORT=8000

   # Remove old container
   docker rm -f decode-vllm-server1 2>/dev/null || true

   # Start decode server
   docker run -d \
     --name decode-vllm-server1 \
     --privileged \
     --device /dev/infiniband/uverbs0 \
     --shm-size=10g \
     -p ${PORT}:${PORT} \
     -e MODEL \
     -e VLLM_BATCH \
     -e MAX_LEN \
     -e ETCD \
     -e PORT \
     public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.9.1-neuronx-py310-sdk2.25.1-ubuntu22.04 \
     bash -c "exec python3 -m vllm.entrypoints.openai.api_server \
       --model \$MODEL \
       --max-num-seqs \$VLLM_BATCH \
       --max-model-len \$MAX_LEN \
       --tensor-parallel-size 64 \
       --device neuron \
       --speculative-max-model-len \$MAX_LEN \
       --override-neuron-config '{}' \
       --kv-transfer-config '{\"kv_connector\":\"NeuronConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":2e11,\"etcd\":\"\$ETCD\"}' \
       --port \$PORT"

Note: The decode server uses ``kv_role:kv_consumer`` in its configuration.

Test Your Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test your DI setup with this simple request:

.. code-block:: bash

   curl -s http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
      "model": "meta-llama/Llama-3.3-70B-Instruct",
      "prompt": ["a tornado is a"],
      "max_tokens": 10,
      "temperature": 0
      }'

Scale Your Setup
~~~~~~~~~~~~~~~~~~~~~

To add more capacity:

1. Launch additional prefill workers when you need more compute power
2. Launch additional decode workers when you need more memory
3. Workers can run on the same instance or different ones
4. New workers automatically register with etcd
5. The proxy automatically routes traffic to all workers

Benchmark Your Setup
~~~~~~~~~~~~~~~~~~~~~~

Install LLMPerf
---------------

1. Get LLMPerf:

.. code-block:: bash

   git clone https://github.com/ray-project/llmperf.git
   cd llmperf
   pip install -e .    

2. Apply the ``neuron_perf.patch`` as shown in :ref:`llm-inference-benchmarking`

3. Create this benchmark script (``llmperf.sh``):

.. code-block:: bash

   #!/bin/bash
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

4. Run the benchmark:

.. code-block:: bash

   MODEL_PATH=path/to/your/downloaded/model ./llmperf.sh 

Compare with Standard Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To benchmark without DI:

1. Stop all DI servers and the proxy
2. Create this script (``baseline_server.sh``):

.. code-block:: bash

   #!/bin/bash
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

3. Run for multi-instance:

.. code-block:: bash
   
   ./baseline_server.sh --tp-degree 64 --batch-size 4 \
                        --model-path path/to/your/downloaded/model \
                        --compiled-model-path di_traced_model_tp64_b4/

Or for single-instance:

.. code-block:: bash
   
   SINGLE_INSTANCE=1 ./baseline_server.sh --tp-degree 32 --batch-size 4 \
                                          --model-path path/to/your/downloaded/model \
                                          --compiled-model-path di_traced_model_tp32_b4/

4. Run the benchmark:

.. code-block:: bash

   MODEL_PATH=path/to/your/downloaded_model ./llmperf.sh 

Known Issues
~~~~~~~~~~~~

If you see ``ENC:kv_store_acquire_file_lock Failed to open kv store server lock file Permission denied``, delete the lock file:

.. code-block:: bash

   sudo rm /tmp/nrt_kv_store_server.lock