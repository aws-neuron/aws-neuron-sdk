.. _nxdi-trn2-llama3.1-8b-multi-lora-tutorial:

Tutorial: Multi-LoRA serving for Llama-3.1-8B on Trn2 instances
=======================================================================================================

NeuronX Distributed (NxD) Inference supports multi-LoRA serving. This tutorial provides a step-by-step
guide for multi-LoRA serving with Llama-3.1-8B as the base model on a Trn2 instance.
It describes two different ways of running multi-LoRA serving with NxDI directly and through vLLM (with NxDI)
We will use LoRA adapters downloaded from HuggingFace as examples for serving.

.. contents:: Table of contents
   :local:
   :depth: 2

Prerequisites:
--------------
Set up and connect to a Trn2.48xlarge instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a prerequisite, this tutorial requires that you have a Trn2 instance
created from a Deep Learning AMI that has the Neuron SDK pre-installed.

To set up a Trn2 instance using Deep Learning AMI with pre-installed Neuron SDK,
see :ref:`nxdi-setup`.

After setting up an instance, use SSH to connect to the Trn2 instance using the key pair that you
chose when you launched the instance.

After you are connected, activate the Python virtual environment that includes the Neuron SDK.

::

   source ~/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate

Run ``pip list`` to verify that the Neuron SDK is installed.

::

   pip list | grep neuron

You should see Neuron packages including
``neuronx-distributed-inference`` and ``neuronx-cc``.

Install packages
~~~~~~~~~~~~~~~~
NxD Inference supports running models with vLLM. This functionality is
available in a fork of the vLLM GitHub repository:

- `aws-neuron/upstreaming-to-vllm <https://github.com/aws-neuron/upstreaming-to-vllm/tree/releases/v2.23.0-v0>`__

To run NxD Inference with vLLM, you need to download and install vLLM from this
fork. Clone the Neuron vLLM fork.

::
   
    git clone -b releases/v2.23.0-v0 https://github.com/aws-neuron/upstreaming-to-vllm.git
    source ~/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate


Install the Neuron vLLM fork into the virtual environment.

::
    
    cd upstreaming-to-vllm
    pip install -r requirements-neuron.txt
    VLLM_TARGET_DEVICE="neuron" pip install -e .
    cd ..


Download base model and LoRA adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To use this sample, you must first download a `Llama-3.1-8B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`__ model checkpoint from Hugging Face
to a local path on the Trn2 instance. 
Note that you may need access from Meta for model download.
For more information, see
`Downloading models <https://huggingface.co/docs/hub/en/models-downloading>`__
in the Hugging Face documentation.

You also need to download LoRA adapters from Hugging Face for multi-LoRA serving.
As examples, you can download `nvidia/llama-3.1-nemoguard-8b-topic-control <https://huggingface.co/nvidia/llama-3.1-nemoguard-8b-topic-control>`__ 
and `reissbaker/llama-3.1-8b-abliterated-lora <https://huggingface.co/reissbaker/llama-3.1-8b-abliterated-lora>`__.


Run multi-LoRA serving on Trn2 from NxD Inference
-------------------------------------------------
We will run multi-LoRA serving from NxD inference with ``inference_demo`` on Trn2 using Llama-3.1-8B and two LoRA adapters. The data type is bfloat16 precision.

You should specifically set the following configurations when enabling multi-LoRA serving with ``inference_demo``.

- ``enable_lora`` - The flag to enable multi-LoRA serving in NxD Inference. Defaults to `False`.

- ``max_loras`` - The maximum number of concurrent LoRA adapters in device memory. Defaults to ``1``.

- ``max_lora_rank`` - The highest LoRA rank that needs to be supported. Defaults to ``16``. If it is not specified, the maximum LoRA rank of the LoRA adapter checkpoints will be used.

- ``lora_ckpt_path`` - The checkpoint path for LoRA adapter in the format of ``adapter_id : path``. Please set this flag multiple times if multiple LoRA adapters are needs.

- ``adapter_id`` - The adapter ID for prompt. Each prompt comes with an adapter ID.


Save the contents of the below script to your favorite 
shell script file, for example, ``multi_lora_model.sh`` and then run it.
The script compiles the model and runs generation on the given input prompt.

::

    # Replace this with the path where you downloaded and saved the model files.
    MODEL_PATH="/home/ubuntu/models/Llama-3.1-8B-Instruct/"
    # Replace the following with the paths where you downloaded and saved the LoRA adapters.
    LORA_PATH_1="/home/ubuntu/models/loras/llama-3.1-nemoguard-8b-topic-control"
    LORA_PATH_2="/home/ubuntu/models/loras/llama-3.1-8b-abliterated-lora"
    # This is where the compiled model will be saved.
    COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.1-8B-Lora/"

    NUM_CORES=128
    TP_DEGREE=32
    LNC=2

    export NEURON_RT_VIRTUAL_CORE_SIZE=$LNC
    export NEURON_RT_NUM_CORES=$TP_DEGREE
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
            --batch-size 2 \
            --max-context-length 12288 \
            --seq-len 64 \
            --on-device-sampling \
            --top-k 1 \
            --do-sample \
            --pad-token-id 2 \
            --enable-bucketing \
            --enable-lora \
            --max-loras 2 \
            --lora-ckpt-path "lora_id_1 : ${LORA_PATH_1}" \
            --lora-ckpt-path "lora_id_2 : ${LORA_PATH_2}" \
            --prompt "I believe the meaning of life is" \
            --adapter-id lora_id_1 \
            --prompt "I believe the meaning of life is" \
            --adapter-id lora_id_2 \
            | tee log

NxDI expects the same number of prompts and adapter IDs in the script.
A prompt is mapped to the adapter ID with the same order.
For example, the first prompt in the script assoicates with ``lora_id_1`` and the second one assoicates with ``lora_id_2``.
Although the two prompts are the same, NxD Inference will generate different outputs due to different adapter IDs.


Using vLLM for multi-LoRA serving on Trn2
-----------------------------------------

We can run multi-LoRA serving on Trn2 with vLLM for Llama models. Please refer to :ref:`nxdi-vllm-user-guide` for more details on how to run model inference on TRN2 with vLLM.


Multi-LoRA Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~

You should specifically set the following configurations when enabling multi-LoRA serving with vLLM.

- ``enable_lora`` - The flag to enable multi-LoRA serving in NxD Inference. Defaults to `False`.

- ``max_loras`` - The maximum number of concurrent LoRA adapters in device memory. Defaults to ``1``.

- ``max_lora_rank`` - The highest LoRA rank that needs to be supported. Defaults to ``16``. If it is not specified, the maximum LoRA rank of the LoRA adapter checkpoints will be used.

- ``lora_modules`` - Set the LoRA checkpoint paths and their adapter IDs in the format of ``adapter_id_1=path1 adapter_id_2=path2 ...``.


Offline inference example
~~~~~~~~~~~~~~~~~~~~~~~~~

You can also run multi-LoRA serving offline on TRN2 with vLLM.

.. code:: ipython3

    import os
    os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.openai.serving_models import LoRAModulePath
    from vllm.lora.request import LoRARequest

    MODEL_PATH="/home/ubuntu/models/Llama-3.1-8B-Instruct/"
    # LoRA checkpoint paths.
    LORA_PATH_1="/home/ubuntu/models/loras/llama-3.1-nemoguard-8b-topic-control"
    LORA_PATH_2="/home/ubuntu/models/loras/llama-3.1-8b-abliterated-lora"

    # Sample prompts.
    prompts = [
        "The president of the United States is",
        "The capital of France is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(top_k=1)

    # Create an LLM with multi-LoRA serving.
    llm = LLM(
        model=MODEL_PATH,
        max_num_seqs=2,
        max_model_len=64,
        tensor_parallel_size=32,
        device="neuron",
        override_neuron_config={
            "sequence_parallel_enabled": False,
        },
        lora_modules=[
            LoRAModulePath(name="lora_id_1", path=LORA_PATH_1),
	        LoRAModulePath(name="lora_id_2", path=LORA_PATH_2),
        ],
        enable_lora=True,
        max_loras=2,
    )
    """ 
    The format of multi-lora requests using NxDI as the backend is different from the default format in vLLM: https://docs.vllm.ai/en/v0.9.0/features/lora.html because NxDI currently doesn't support dynamic loading of LoRA adapters.
    Only the lora_name needs to be specified.  
    The lora_id and lora_path are supplied at the LLM class/server initialization, after which the paths are
    handled by NxDI.
    """
    lora_req_1 = LoRARequest("lora_id_1", 0, " ")
	lora_req_2 = LoRARequest("lora_id_2", 1, " ")
    outputs = llm.generate(prompts, sampling_params, lora_request=[lora_req_1, lora_req_2])

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")



Online server example
~~~~~~~~~~~~~~~~~~~~~

Save the contents of the below script to another shell script file, for example, ``start_vllm.sh`` and then run it.

::

    export NEURON_RT_INSPECT_ENABLE=0 
    export NEURON_RT_VIRTUAL_CORE_SIZE=2

    # These should be the same paths used when compiling the model.
    MODEL_PATH="/home/ubuntu/models/Llama-3.1-8B-Instruct/"
    # Replace the following with the paths where you downloaded and saved the LoRA adapters.
    LORA_PATH_1="/home/ubuntu/models/loras/llama-3.1-nemoguard-8b-topic-control"
    LORA_PATH_2="/home/ubuntu/models/loras/llama-3.1-8b-abliterated-lora"
    # This is where the compiled model will be saved.
    COMPILED_MODEL_PATH="/home/ubuntu/traced_model/Llama-3.1-8B-Lora/"

    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
    export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH
    VLLM_RPC_TIMEOUT=100000 python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --max-num-seqs 2 \
        --max-model-len 64 \
        --tensor-parallel-size 32 \
        --device neuron \
        --use-v2-block-manager \
        --enable-lora \ 
        --max-loras 2 \
        --override-neuron-config "{\"sequence_parallel_enabled\": false}" \
        --lora-modules lora_id_1=${LORA_PATH_1} lora_id_2=${LORA_PATH_2} \
        --port 8000 &
    PID=$!
    echo "vLLM server started with PID $PID"



After the vLLM server is launched, we can send requests to the server for serving. A sample request is:

::

    curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "lora_id_1",
        "messages": [
            {
                "role": "user", 
                "content": "The president of the United States is"
            }
        ] 
    }'