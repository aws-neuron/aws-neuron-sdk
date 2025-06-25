.. _nxdi-trn2-llama3.3-70b-apc-tutorial:

Tutorial: Using Prefix Caching with Llama-3.3-70B on Trn2 instances
===================================================================

This tutorial provides a step-by-step guide to deploy Llama3.3 70B using 
NeuronX Distributed (NxD) Inference on a single Trn2.48xl instance using two
different configurations, one with prefix caching enabled and the other
without prefix caching. We will also measure average response time
for both the configurations with prompts containing a common prefix.

.. contents:: Table of contents
   :local:
   :depth: 2


Background, Concepts, and Optimizations
---------------------------------------

Block KV Cache Layout
~~~~~~~~~~~~~~~~~~~~~

To support prefix caching, NxDI now uses block kv cache layout. Enable block layout of
the cache by setting ``is_block_kv_layout=True`` in NeuronConfig. The first two
dimensions of the KV cache are set to the number of blocks and block size, respectively.
These configurations are specified using ``pa_num_blocks`` and ``pa_block_size`` in NeuronConfig.

For optimal performance with Neuron, it's recommended to set ``pa_block_size=32``.
The minimum required ``pa_num_blocks`` can be calculated using the formula
``(batch_size * max_seq_len) / block_size`` where batch_size is the compiled batch size
and max_seq_len is the maximum sequence length of the compiled model on Neuron.
While using the minimum block calculation will produce accurate results, it's recommended
to initialize as many blocks as possible without exceeding HBM space limitations. This
ensures that Neuron has sufficient blocks to save as much prefix data as possible. More cache
blocks implies higher prefix caching hit rate and hence better context encoding performance.

Kernels
~~~~~~~

NxD Inference supports kernels that optimize parts of the modeling code
for best performance when prefix caching is enabled.

- Token generation attention kernel with block kv cache read and update capabilities.
  This kernel reads the cache blocks using the active block table, converts the required
  blocks into flat layout, performs attention and scatters back the computed key and value
  to the correct slot in the block cache layout. To enable this kernel, set
  ``attn_block_tkg_nki_kernel_enabled=True`` and ``attn_block_tkg_nki_kernel_cache_update=True``
  in NeuronConfig.


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

   source ~/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate

Run ``pip list`` to verify that the Neuron SDK is installed.

::

   pip list | grep neuron

You should see Neuron packages including
``neuronx-distributed-inference`` and ``neuronx-cc``.

Install packages
~~~~~~~~~~~~~~~~~
NxD Inference supports running models with vLLM. This functionality is
available in a fork of the vLLM GitHub repository:

- `aws-neuron/upstreaming-to-vllm <https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.24-vllm-v0.7.2>`__

To run NxD Inference with vLLM, you need to download and install vLLM from this
fork. Clone the Neuron vLLM fork.

::
   
    git clone -b neuron-2.24-vllm-v0.7.2 https://github.com/aws-neuron/upstreaming-to-vllm.git


Make sure to activate the Neuron virtual environment if using a new terminal instead of the one from connect step above.

::
    
    source ~/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate


Install the Neuron vLLM fork into the virtual environment.

::
    
    cd upstreaming-to-vllm
    pip install -r requirements-neuron.txt
    VLLM_TARGET_DEVICE="neuron" pip install -e .
    cd ..


Download models
^^^^^^^^^^^^^^^
To use this sample, you must first download a 70B model checkpoint from Hugging Face
to a local path on the Trn2 instance. For more information, see
`Downloading models <https://huggingface.co/docs/hub/en/models-downloading>`__
in the Hugging Face documentation. You can download and use `meta-llama/Llama-3.3-70B-Instruct <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`__
for this tutorial.

Scenario 1: Run Llama3.3 70B on Trn2 without Prefix Caching
-----------------------------------------------------------

Step 1: Compile the model
~~~~~~~~~~~~~~~~~~~~~~~~~

We will first compile using a command installed by ``neuronx-distributed-inference``.
Save the contents of the below script to your favorite 
shell script file, for example, ``compile_model.sh`` and then run it.

Note that we are also using the following features as described in
the tutorial for running 405B model :ref:`nxdi-trn2-llama3.1-405b-tutorial`

* Logical NeuronCore Configuration (LNC)
* Tensor parallelism (TP) on Trn2
* Optimized Kernels

Note the path we used to save the compiled model. This path should be used
when launching vLLM server for inference so that the compiled model can be loaded without recompilation.
Refer to :ref:`nxd-inference-api-guide` for more information on these ``inference_demo`` flags.

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
            --batch-size 4 \
            --is-continuous-batching \
            --ctx-batch-size 1 \
            --tkg-batch-size 4 \
            --max-context-length 8192 \
            --seq-len 8192 \
            --on-device-sampling \
            --top-k 1 \
            --do-sample \
            --fused-qkv \
            --sequence-parallel-enabled \
            --qkv-kernel-enabled \
            --attn-kernel-enabled \
            --mlp-kernel-enabled \
            --attn-block-tkg-nki-kernel-enabled \
            --attn-block-tkg-nki-kernel-cache-update \
            --k-cache-transposed \
            --cc-pipeline-tiling-factor 1 \
            --pad-token-id 2 \
            --enable-bucketing \
            --context-encoding-buckets 512 1024 2048 4096 8192 \
            --token-generation-buckets 512 1024 2048 4096 8192 \
            --compile-only \
            --prompt "What is annapurna labs?" 2>&1 | tee log.txt


Step 2: Serve the model using vLLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        --max-num-seqs 4 \
        --max-model-len 8192 \
        --tensor-parallel-size 64 \
        --device neuron \
        --use-v2-block-manager \
        --block-size 32 \
        --port 8000 &
    PID=$!
    echo "vLLM server started with PID $PID"

If you see the below logs, that means your server is up and running:
::

    INFO: Started server process [284309]
    INFO: Waiting for application startup.
    INFO: Application startup complete.
    INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

Step 3: Analyze Request response from server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An example script has been added to demonstrate how a common lookup table is used to
answer 10 different questions while measuring the total response time. The lookup table
serves as a shared prefix that's consistently applied across all 10 input prompts.
The script will calculate and display the average time required to answer all questions.

Open a new terminal as the client where you can send requests to the
server. Save the contents of the example below to another
shell script file, for example, ``send_request.sh`` and then run it.

::

    MODEL_PATH="/home/ubuntu/models/Llama-3.3-70B-Instruct/"
    LONG_PROMPT=$(cat << 'EOL'
    You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as follows.
    # Table
    | ID  | Name          | Age | Occupation    | Country       | Email                  | Phone Number   | Address                       |
    |-----|---------------|-----|---------------|---------------|------------------------|----------------|------------------------------|
    | 1   | John Doe      | 29  | Engineer      | USA           | john.doe@example.com   | 555-1234       | 123 Elm St, Springfield, IL  |
    | 2   | Jane Smith    | 34  | Doctor        | Canada        | jane.smith@example.com | 555-5678       | 456 Oak St, Toronto, ON      |
    | 3   | Alice Johnson | 27  | Teacher       | UK            | alice.j@example.com    | 555-8765       | 789 Pine St, London, UK      |
    | 4   | Bob Brown     | 45  | Artist        | Australia     | bob.b@example.com      | 555-4321       | 321 Maple St, Sydney, NSW    |
    | 5   | Carol White   | 31  | Scientist     | New Zealand   | carol.w@example.com    | 555-6789       | 654 Birch St, Wellington, NZ |
    | 6   | Dave Green    | 28  | Lawyer        | Ireland       | dave.g@example.com     | 555-3456       | 987 Cedar St, Dublin, IE     |
    | 7   | Emma Black    | 40  | Musician      | USA           | emma.b@example.com     | 555-1111       | 246 Ash St, New York, NY     |
    | 8   | Frank Blue    | 37  | Chef          | Canada        | frank.b@example.com    | 555-2222       | 135 Spruce St, Vancouver, BC |
    | 9   | Grace Yellow  | 50  | Engineer      | UK            | grace.y@example.com    | 555-3333       | 864 Fir St, Manchester, UK   |
    | 10  | Henry Violet  | 32  | Artist        | Australia     | henry.v@example.com    | 555-4444       | 753 Willow St, Melbourne, VIC|
    | 11  | Irene Orange  | 26  | Scientist     | New Zealand   | irene.o@example.com    | 555-5555       | 912 Poplar St, Auckland, NZ  |
    | 12  | Jack Indigo   | 38  | Teacher       | Ireland       | jack.i@example.com     | 555-6666       | 159 Elm St, Cork, IE         |
    | 13  | Karen Red     | 41  | Lawyer        | USA           | karen.r@example.com    | 555-7777       | 357 Cedar St, Boston, MA     |
    | 14  | Leo Brown     | 30  | Chef          | Canada        | leo.b@example.com      | 555-8888       | 246 Oak St, Calgary, AB      |
    | 15  | Mia Green     | 33  | Musician      | UK            | mia.g@example.com      | 555-9999       | 975 Pine St, Edinburgh, UK   |
    | 16  | Noah Yellow   | 29  | Doctor        | Australia     | noah.y@example.com     | 555-0000       | 864 Birch St, Brisbane, QLD  |
    | 17  | Olivia Blue   | 35  | Engineer      | New Zealand   | olivia.b@example.com   | 555-1212       | 753 Maple St, Hamilton, NZ   |
    | 18  | Peter Black   | 42  | Artist        | Ireland       | peter.b@example.com    | 555-3434       | 912 Fir St, Limerick, IE     |
    | 19  | Quinn White   | 28  | Scientist     | USA           | quinn.w@example.com    | 555-5656       | 159 Willow St, Seattle, WA   |
    | 20  | Rachel Red    | 31  | Teacher       | Canada        | rachel.r@example.com   | 555-7878       | 357 Poplar St, Ottawa, ON    |
    | 21  | Steve Green   | 44  | Lawyer        | UK            | steve.g@example.com    | 555-9090       | 753 Elm St, Birmingham, UK   |
    | 22  | Tina Blue     | 36  | Musician      | Australia     | tina.b@example.com     | 555-1213       | 864 Cedar St, Perth, WA      |
    | 23  | Umar Black    | 39  | Chef          | New Zealand   | umar.b@example.com     | 555-3435       | 975 Spruce St, Christchurch, NZ|
    | 24  | Victor Yellow | 43  | Engineer      | Ireland       | victor.y@example.com   | 555-5657       | 246 Willow St, Galway, IE    |
    | 25  | Wendy Orange  | 27  | Artist        | USA           | wendy.o@example.com    | 555-7879       | 135 Elm St, Denver, CO       |
    | 26  | Xavier Green  | 34  | Scientist     | Canada        | xavier.g@example.com   | 555-9091       | 357 Oak St, Montreal, QC     |
    | 27  | Yara Red      | 41  | Teacher       | UK            | yara.r@example.com     | 555-1214       | 975 Pine St, Leeds, UK       |
    | 28  | Zack Blue     | 30  | Lawyer        | Australia     | zack.b@example.com     | 555-3436       | 135 Birch St, Adelaide, SA   |
    | 29  | Amy White     | 33  | Musician      | New Zealand   | amy.w@example.com      | 555-5658       | 159 Maple St, Wellington, NZ |
    | 30  | Ben Black     | 38  | Chef          | Ireland       | ben.b@example.com      | 555-7870       | 246 Fir St, Waterford, IE    |
    EOL
    )

    questions=(
        "Question: what is the age of John Doe? Your answer: The age of John Doe is "
        "Question: what is the age of Zack Blue? Your answer: The age of Zack Blue is "
        "Question: Which country is Ben Black from? Your answer: The country of Ben Black is "
        "Question: Who has rachel.r@example.com as their email domain? Your answer: The email domain rachel.r@example.com belongs to "
        "Question: What is the phone number for contacting Karen Red? Your answer: The phone number for contacting Karen Red is "
        "Question: What is the occupation of Tina Blue? Your answer: The occupation of Tina Blue is "
        "Question: What is the name of the person with id as 29? Your answer: The name of the person with id as 29 is "
        "Question: What is the address of Alice Johnson? Your answer: The address of Alice Johnson is "
        "Question: What is the id of Irene Orange? Your answer: The id of Irene Orange is "
        "Question: What is the age of Leo Brown? Your answer: The age of Leo Brown is "
    )


    # Function to make a single request
    make_request() {
        local question=$1
        local prompt_with_suffix="${LONG_PROMPT}

    Based on the table above, please answer this question:
    ${question}"
        
        local escaped_prompt=$(echo "$prompt_with_suffix" | jq -Rs .)
        
        # Make the curl request and capture both response and time
        local response_file=$(mktemp)
        time_output=$(TIMEFORMAT='%R'; { time curl -s http://localhost:8000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL_PATH\",
                \"messages\": [
                    {
                        \"role\": \"user\",
                        \"content\": ${escaped_prompt}
                    }
                ]
            }" > "$response_file"; } 2>&1)
        
        # Extract the response content
        local response_content=$(cat "$response_file" | jq -r '.choices[0].message.content')
        rm "$response_file"
        
        # Return both time and response
        echo "TIME:$time_output"
        echo "RESPONSE:$response_content"
    }

    # Make first request (warm-up) with a random question
    random_index=$((RANDOM % ${#questions[@]}))
    echo "Warm-up request with question: ${questions[$random_index]}"
    IFS=$'\n' read -r -d '' time_str response_str < <(make_request "${questions[$random_index]}" && echo '')
    echo "Response: $response_str"
    echo "Time taken: ${time_str#TIME:} seconds"
    echo "Warm-up complete"
    echo "-------------------"

    # Make 10 timed requests with random questions
    total_time=0
    for i in {0..9}; do
        random_index=$i
        #random_index=$((RANDOM % ${#questions[@]}))
        question="${questions[$random_index]}"
        echo "Request $i with question: $question"
        
        IFS=$'\n' read -r -d '' time_str response_str < <(make_request "$question" && echo '')
        time_taken=${time_str#TIME:}
        response=${response_str#RESPONSE:}
        
        total_time=$(echo "$total_time + $time_taken" | bc -l)
        echo "Response: $response"
        echo "Time taken: ${time_taken} seconds"
        echo "-------------------"
    done

    # Calculate and display average time
    average_time=$(echo "scale=3; $total_time / 10" | bc -l)
    echo "Average time across 10 requests: ${average_time} seconds"

Output from the script would include all the answers to the questions along with the
average time to process all the requests at the very end as shown below.

::

    Average time across 10 requests: .388 seconds


Scenario 2: Run Llama3.3 70B on Trn2 with Prefix Caching
--------------------------------------------------------

Step 1: Compile the model
~~~~~~~~~~~~~~~~~~~~~~~~~

The compilation script with prefix caching adds extra flags specific to prefix caching
to enable and configure Block KV cache layout along with enabling the kernels used with
prefix caching. Please refer to :ref:`nxdi_prefix_caching` for more information on the
prefix caching flags used below.

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
            --batch-size 4 \
            --is-continuous-batching \
            --ctx-batch-size 1 \
            --tkg-batch-size 4 \
            --max-context-length 8192 \
            --seq-len 8192 \
            --on-device-sampling \
            --top-k 1 \
            --do-sample \
            --fused-qkv \
            --sequence-parallel-enabled \
            --qkv-kernel-enabled \
            --attn-kernel-enabled \
            --mlp-kernel-enabled \
            --attn-block-tkg-nki-kernel-enabled \
            --attn-block-tkg-nki-kernel-cache-update \
            --cc-pipeline-tiling-factor 1 \
            --pad-token-id 2 \
            --enable-bucketing \
            --context-encoding-buckets 512 1024 2048 4096 8192 \
            --token-generation-buckets 512 1024 2048 4096 8192 \
            --prefix-buckets 512 1024 2048 \
            --enable-block-kv-layout \
            --pa-num-blocks 2048 \
            --pa-block-size 32 \
            --enable-prefix-caching \
            --compile-only \
            --prompt "What is annapurna labs?" 2>&1 | tee log.txt


Step 2: Serve the model using vLLM with prefix caching enabled
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After compiling the model, you can serve the model using vLLM with prefix caching enabled.
Save the contents of the below script to another
shell script file, for example, ``start_vllm_apc.sh`` and then run it.

Note that we use ``--enable-prefix-caching`` in vLLM to enable prefix caching, along
with ``--block-size 32`` and ``--num-gpu-blocks-override 2048`` which are consistent
with ``--pa-block-size 32`` and ``--pa-num-blocks 2048`` flags specified during model
compilation.

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
        --max-num-seqs 4 \
        --max-model-len 8192 \
        --tensor-parallel-size 64 \
        --device neuron \
        --use-v2-block-manager \
        --num-gpu-blocks-override 2048 \
        --enable-prefix-caching \
        --block-size 32 \
        --override-neuron-config "{\"is_block_kv_layout\": true, \"is_prefix_caching\": true}" \
        --port 8000 &
    PID=$!
    echo "vLLM server started with PID $PID"

Wait for the server to be up and running before proceeding further.

Step 3: Analyze Request response from server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the same ``send_request.sh`` script file from scenario 1,
to send identical request to the server with prefix caching enabled.
The average time to respond to all the requests will be printed in the terminal.

::

    Average time across 10 requests: .288 seconds

As seen from the two scenarios, average time with prefix caching enabled is lesser than the time
it takes to serve the same requests with prefix caching disabled. This is attributed to the lesser
time to compute the first token by reusing the common prefix across all the prompts.

We also ran the same model configurations with public datasets with varying cache hit rates for 
benchmarking prefix caching on neuron and here are the results that we achieved

.. csv-table::
   :file: llama70b_apc_perf_comparison.csv
   :header-rows: 1

Conclusion
-----------

In general, with a higher ratio of prefix(shared prompt) to prefill tokens that results in higher cache-hit rate, 
prefix caching achieves a TTFT speedup of up to 3x compared to when prefix caching is disabled. When the dataset has
low prefix cache hit rate, prefix caching TTFT performance can degrade slightly due to the overhead of supporting
block KV cache layout, as seen in the HumanEval dataset.