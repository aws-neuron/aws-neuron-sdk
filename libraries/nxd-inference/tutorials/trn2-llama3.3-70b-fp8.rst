.. _nxdi-vllm-llama-fp8-inference-tutorial:

Tutorial: Deploy fp8 quantized Llama3.3-70B on Trn2 instances
============================================================================================

Quantization can significantly reduce the model size and inference time. This tutorial provides a step-by-step guide to deploy a fp8 quantized Llama3.3-70B on Trainium2 instances. We utilize the custom quantization feature to quantize specific layers from the original model checkpoint. 

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


For this tutorial, we use a pre-installed virtual environment in the DLAMI at ``/opt/aws_neuronx_venv_pytorch_inference_vllm``. If you prefer to use a container, start a built-in vLLM Neuron Deep Learning Container (DLC). For more information about available containers,
see the `AWS Neuron Deep Learning Containers repository <https://github.com/aws-neuron/deep-learning-containers#vllm-inference-neuronx>`_.



Step 1: Quantize the Llama3.3 70B b16 checkpoint to fp8 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We first quantize the `original Llama3.3 70B model <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>`_ checkpoint using `modules from Neuronx Distributed <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/custom-quantization.html#quantize-using-nxd>`_.
In the below script, ``modules_to_not_convert`` contains the layers that are not being quantized to fp8. In this instance, we quantize only the mlp layers except the first and the last layer. If you have a similar FP8 checkpoint, you can skip this step and use that.
Use the below code snippet to create a script for quantization and execute the script. This will create a fp8 checkpoint in the `output_path`.
::

    import json
    import torch
    from typing import Optional, List
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
    from neuronx_distributed_inference.modules.checkpoint import prune_state_dict,save_state_dict_safetensors
    from neuronx_distributed.quantization.quantization_utils import quantize_pytorch_model_per_channel_symmetric, convert_qint8_to_int8_state_dict

    model_path = "<path to the bf16 checkpoint>"
    output_path = "<path to save the quantized checkpoint>"

    modules_to_not_convert = [
        "lm_head",
        "layers.0.mlp",
        "layers.79.mlp",
        "layers.0.self_attn",
        "layers.1.self_attn",
        "layers.2.self_attn",
        "layers.3.self_attn",
        "layers.4.self_attn",
        "layers.5.self_attn",
        "layers.6.self_attn",
        "layers.7.self_attn",
        "layers.8.self_attn",
        "layers.9.self_attn",
        "layers.10.self_attn",
        "layers.11.self_attn",
        "layers.12.self_attn",
        "layers.13.self_attn",
        "layers.14.self_attn",
        "layers.15.self_attn",
        "layers.16.self_attn",
        "layers.17.self_attn",
        "layers.18.self_attn",
        "layers.19.self_attn",
        "layers.20.self_attn",
        "layers.21.self_attn",
        "layers.22.self_attn",
        "layers.23.self_attn",
        "layers.24.self_attn",
        "layers.25.self_attn",
        "layers.26.self_attn",
        "layers.27.self_attn",
        "layers.28.self_attn",
        "layers.29.self_attn",
        "layers.30.self_attn",
        "layers.31.self_attn",
        "layers.32.self_attn",
        "layers.33.self_attn",
        "layers.34.self_attn",
        "layers.35.self_attn",
        "layers.36.self_attn",
        "layers.37.self_attn",
        "layers.38.self_attn",
        "layers.39.self_attn",
        "layers.40.self_attn",
        "layers.41.self_attn",
        "layers.42.self_attn",
        "layers.43.self_attn",
        "layers.44.self_attn",
        "layers.45.self_attn",
        "layers.46.self_attn",
        "layers.47.self_attn",
        "layers.48.self_attn",
        "layers.49.self_attn",
        "layers.50.self_attn",
        "layers.51.self_attn",
        "layers.52.self_attn",
        "layers.53.self_attn",
        "layers.54.self_attn",
        "layers.55.self_attn",
        "layers.56.self_attn",
        "layers.57.self_attn",
        "layers.58.self_attn",
        "layers.59.self_attn",
        "layers.60.self_attn",
        "layers.61.self_attn",
        "layers.62.self_attn",
        "layers.63.self_attn",
        "layers.64.self_attn",
        "layers.65.self_attn",
        "layers.66.self_attn",
        "layers.67.self_attn",
        "layers.68.self_attn",
        "layers.69.self_attn",
        "layers.70.self_attn",
        "layers.71.self_attn",
        "layers.72.self_attn",
        "layers.73.self_attn",
        "layers.74.self_attn",
        "layers.75.self_attn",
        "layers.76.self_attn",
        "layers.77.self_attn",
        "layers.78.self_attn",
        "layers.79.self_attn"
    ]

    def quantize(model: torch.nn.Module, dtype=torch.qint8, modules_to_not_convert: Optional[List[str]] = None) -> torch.nn.Module:
        quant_model = quantize_pytorch_model_per_channel_symmetric(model,dtype=dtype, modules_to_not_convert=modules_to_not_convert)
        model_quant_sd = quant_model.state_dict()
        convert_qint8_to_int8_state_dict(model_quant_sd)
        quantized_state_dict = prune_state_dict(model_quant_sd)
        return quantized_state_dict

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generation_config = GenerationConfig.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)

    state_dict = quantize(model,torch.float8_e4m3fn,modules_to_not_convert)

    save_state_dict_safetensors(state_dict=state_dict,state_dict_dir=output_path)

    #save tokenizer, config in new checkpoint folder
    tokenizer.save_pretrained(output_path)
    config.save_pretrained(output_path)
    generation_config.save_pretrained(output_path)

    modules_to_not_convert_json = {
        "model": {
            "modules_to_not_convert": modules_to_not_convert
        }
    }

    with open(f"{output_path}/modules_to_not_convert.json", "w") as f:
        json.dump(modules_to_not_convert_json, f, indent=2)




Step 2: Compile the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this step, we use the quantized fp8 checkpoint to compile the model using a utility from `neuronx-distributed-inference <https://github.com/aws-neuron/neuronx-distributed-inference>`_.
Note that we are using multiple optimization features like tensor parallelism, sequence parallelism and optimized kernels for attention, mlp and QKV computation.
You can modify some of the below parameters based on your use case:

* ``tp-degree``: set this to the number of neuron cores for partitioning the model. Typically ``local_ranks_size`` needs to be set to the same value.
* ``batch-size``: set this to the desired number of requests to process simultaneously. Along with this, ``tkg-batch-size`` and ``max-batch-size`` should be set to the same value.
* ``seq-len``: set this to the maximum sequence length during inference. i.e. sum of input and output sequence lengths.

::

    export NEURON_RT_INSPECT_ENABLE=0
    export NEURON_RT_EXEC_TIMEOUT=600
    export NEURON_RT_VIRTUAL_CORE_SIZE=2
    export NEURON_RT_NUM_CORES=64
    export XLA_DENSE_GATHER_FACTOR=0
    export XLA_IR_DEBUG=1
    export XLA_HLO_DEBUG=1
    export XLA_HANDLE_SPECIAL_SCALAR=1
    export UNSAFE_FP8FNCAST=1
    export DISABLE_NUMERIC_CC_TOKEN=1
    MODEL_PATH="<path to the fp8 model checkpoint"
    COMPILED_MODEL_PATH="<folder to save compiled artifacts>"
    export BASE_COMPILE_WORK_DIR="<folder to save compiled artifacts>"
    inference_demo \
        --model-type llama \
        --task-type causal-lm \
        run \
        --model-path $MODEL_PATH \
        --compiled-model-path $COMPILED_MODEL_PATH \
        --torch-dtype bfloat16 \
        --batch-size 4 \
        --enable-bucketing \
        --local_ranks_size 64 \
        --tp-degree 64 \
        --start_rank_id 0 \
        --pad-token-id 0 \
        --cc-pipeline-tiling-factor 1 \
        --on-device-sampling \
        --global-topk 256 \
        --dynamic \
        --top-k 50 \
        --top-p 0.9 \
        --temperature 0.7 \
        --do-sample \
        --sequence-parallel-enabled \
        --fused-qkv \
        --qkv-kernel-enabled \
        --attn-kernel-enabled \
        --mlp-kernel-enabled \
        --logical-neuron-cores 2 \
        --prompt "What is annapurna labs?" \
        --ctx-batch-size 1 \
        --tkg-batch-size 4 \
        --max-batch-size 4 \
        --is-continuous-batching \
        --compile-only \
        --quantized-mlp-kernel-enabled \
        --quantization-type per_channel_symmetric \
        --rmsnorm-quantize-kernel-enabled \
        --modules-to-not-convert-file $MODEL_PATH/modules_to_not_convert.json \
        --async-mode \
        --attn-block-tkg-nki-kernel-enabled \
        --attn-block-tkg-nki-kernel-cache-update \
        --k-cache-transposed \
        --save-sharded-checkpoint \
        --max-context-length 4096 \
        --seq-len 5120 \
        --context-encoding-buckets  2048 4096 \
        --token-generation-buckets  5120   2>&1 | tee compile.log

.. note::

    There is a known limitation for compiling the fp8 model directly through vllm. This will be fixed in a future release.


Step 3: Serve the model using vLLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this step, we use the pre-compiled model from the previous step and serve it using vllm.

* ``tensor-parallel-size``: set this to the ``tp-degree`` used during compilation.
* ``max-num-seqs``: set this to the ``batch-size`` used during compilation.
* ``max-model-len``: set this to ``seq-len`` from the above step.

Note that we set an environment variable (``NEURON_COMPILED_ARTIFACTS``) to the path that has the compiled model from the previous step. The vllm command skips compilation and loads the model using the pre-compiled artifacts.
::

    export NEURON_RT_INSPECT_ENABLE=0
    export NEURON_RT_EXEC_TIMEOUT=600
    export NEURON_RT_VIRTUAL_CORE_SIZE=2
    export NEURON_RT_NUM_CORES=64
    export NEURON_RT_VISIBLE_CORES='0-63'
    export XLA_DENSE_GATHER_FACTOR=0
    export XLA_IR_DEBUG=1
    export XLA_HLO_DEBUG=1
    export XLA_HANDLE_SPECIAL_SCALAR=1
    export UNSAFE_FP8FNCAST=1
    export DISABLE_NUMERIC_CC_TOKEN=1
    export VLLM_RPC_TIMEOUT=100000
    export VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference
    
    MODEL_PATH="<path to Llama3.3 70B fp8 checkpoint>"
    COMPILED_MODEL_PATH="<path to a folder that has the pre-compiled model artifacts from the previous step>"
    export NEURON_COMPILED_ARTIFACTS=$COMPILED_MODEL_PATH

    vllm serve \
        $MODEL_PATH \
        --tensor-parallel-size 64 \
        --max-num-seqs 4 \
        --max-model-len 5120 \
        --port 8000 \
        --disable-log-requests \
        --block_size 128 \
        --num-gpu-blocks-override 4 \
        --no-enable-prefix-caching \
        --additional-config '{
            "override_neuron_config": {
                "max_prompt_length": 4096
               }
        }' 2>&1 | tee vllm.log 


Once the model is loaded, you will see the following output:

::

    INFO:     Started server process [7]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.

This indicates the server is ready and the model endpoint is available for inference.

Step 4: Test the endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~
You can test the endpoint using curl or any HTTP client:

::

    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "<model name>",
            "prompt": "What is machine learning?",
            "max_tokens": 100,
            "temperature": 0.7
        }'


Conclusion
----------
You have successully quantized a Llama3.3 70B model to fp8 and deployed the model on Trainium 2 for inference. To evaluate the accuracy of the quantized model, run accuracy evaluation tests using :ref:`accuracy-eval-with-datasets`.