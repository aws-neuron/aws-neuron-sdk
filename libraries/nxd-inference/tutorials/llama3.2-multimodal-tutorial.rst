.. _nxdi-llama3.2-multimodal-tutorial:

Tutorial: Deploying Llama3.2 Multimodal Models
========================================

NeuronX Distributed Inference (NxDI) enables you to deploy ``Llama-3.2-90B-Vision-Instruct`` models on 
Neuron Trainium and Inferentia instances.

You can run Llama3.2 Multimodal with default configuration options. NxD Inference also provides several 
features and configuration options that you can use to optimize and tune the performance for inference. 
This guide walks through how to run Llama3.2 Multimodal with vLLM, and how to enable optimizations for 
inference performance on Trn1/Inf2 instances. It takes about 20-60 minutes to complete. Please note that
``Llama-3.2-11B-Vision-Instruct`` is currently not supported on PyTorch 2.5.

.. contents:: Table of contents
   :local:
   :depth: 2

Step 1: Set up Development Environment
--------------------------------------
1. Launch a ``trn1.32xlarge`` or ``inf2.48xlarge`` instance on Ubuntu 22 with Neuron Multi-Framework DLAMI.
   Please refer to the :ref:`setup guide <setup-ubuntu22-multi-framework-dlami>`
   if you don’t have one yet. If you are looking to install NxD Inference library without using pre-existing 
   DLAMI, please refer to the :ref:`nxdi-setup`.

2. Use default virtual environment pre-installed with the Neuron SDK. We currently only support Llama3.2
   Multimodal 90B model with PyTorch 2.5.
   
::

    source /opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate


3. Install the fork of vLLM (v0.6.x-neuron) that supports NxD Inference following :ref:`nxdi-vllm-user-guide`.
   
4. You should now have the Neuron SDK and other necessary packages installed,
   including ``neuronx-distributed-inference``, ``neuronx-cc``, ``torch``, ``torchvision``, and ``vllm-neuronx``.


Step 2: Download and Convert Checkpoints
----------------------------------------
Download Llama3.2 Multimodal models from either 
`Meta's official website <https://www.llama.com/llama-downloads/>`__ 
or `HuggingFace(HF) <https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct>`__. 

NxDI supports HF checkpoint out-of-the-box. To use the Meta checkpoint, 
you need to run the following script to convert the downloaded Meta checkpoint into NxDI supported format.

::

    python -m neuronx_distributed_inference.models.mllama.convert_mllama_weights_to_neuron \
        --input-dir <path_to_meta_pytorch_checkpoint> \
        --output-dir <path_to_neuron_checkpoint> \
        --instruct \
        --num-shards 8 #(for 90B)

After the script is finished running, you should have the following configuration 
and checkpoint files. Verify by ``ls <path_to_neuron_checkpoint>``:

::

    chat_template.json      model.safetensors          tokenizer_config.json
    config.json             special_tokens_map.json
    generation_config.json  tokenizer.json


.. note::

    The following code examples use 90B model HF checkpoint, as it is supported by default, 
    no script needs to be run.


Step 3: Deploy with vLLM Inference
------------------------------------------
We provide two examples to deploy Llama3.2 Multimodal with vLLM: 

1. Offline inference: you can provide prompts in a python script and execute it.

2. Online inference: you will serve the model in an online server and send requests.

If you already have a compiled model artifact in ``MODEL_PATH`` 
with the same specified configuration, or if you have set an environment variable 
``NEURON_COMPILED_ARTIFACTS`` , the vLLM engine will load the compiled model and run 
inference directly. Otherwise, it will automatically compile and save a new model artifact. 
See :ref:`nxdi-vllm-user-guide` for more. We provide example configurations here, continue 
reading on how to tune them per your use case.


Configurations
~~~~~~~~~~~~~~

You should specifically tune these configurations when optimizing performance for 
Llama3.2 Multimodal models. Please refer to :ref:`nxdi-feature-guide` for detailed 
explanation of each configuration.

- ``MODEL_PATH`` - The directory containing NxDI-supported configs, checkpoints, 
  and neuron compiled artifacts.

- ``BATCH_SIZE`` - Batch size and sequence length together are bounded by device 
  memory. For sequence shorter than 16k, we support up to batch size 4. For longer 
  sequence, we support batch size 1.

- ``SEQ_LEN`` - The entire sequence length combining input and output sequence. We 
  support sequence length up to 16k for 90B model.

- ``TENSOR_PARALLEL_SIZE`` - For best performance, choose the maximum supported 
  value by your instance, that is divisible by the model’s hidden sizes and number 
  of attention heads: 32 for ``trn1.32xlarge`` and 16 for ``inf2.48xlarge``.

- ``CONTEXT_ENCODING_BUCKETS`` - Set based on your distribution of input/context 
  length. For example, suppose 90% of the input traffic is shorter than 1k sequence, 
  and all are less than 2k, then we should set the context encoding buckets to be 
  ``[1024, 2048]``.

- ``TOKEN_GENERATION_BUCKETS`` - Set based on your distribution of entire sequence 
  length. Use similar principle as above.

.. note::

    Longer sequence takes up more memory, so we should use less buckets. For example, 
    to compile the 90B model on ``trn1.32xlarge`` with ``SEQ_LEN=16384, BATCH_SIZE=4``, 
    we can use buckets ``[1024, 2048, 16384]`` to cover the longest possible sequence as 
    well as shorter sequence where the majority of traffic comes from. We also set an 
    environment variable by ``export NEURON_SCRATCHPAD_PAGE_SIZE=1024`` to increase the 
    scratchpad size in our direct memory access engine to fit the large tensors.

- ``SEQUENCE_PARALLEL_ENABLED`` - Set to ``True`` to enable sequence parallel. 
  In principle, sequence parallel helps scaling to long sequence length by splitting 
  tensors along the sequence dimension. However, for short sequence length less than 
  2k, it is not worth to pay for the collectives overhead when compute workload is 
  manageable. So in this example, as we configured sequence length to be no more than 2k,
  we disabled the sequence parallel.

- ``IS_CONTINUOUS_BATCHING`` - Set based on your input traffic. For example, suppose 
  end-to-end latency to generate an entire output sequence (batch size 1) is 1 second 
  in average. However, you receive a request every 0.5 second. Then it is beneficial 
  to enable continuous batching so that new request can get generation started before 
  prior request is finished.

- ``ON_DEVICE_SAMPLING_CONFIG`` - We enable on-device sampling to perform sampling 
  logic on the Neuron device (rather than on the CPU) to achieve better performance.


Model Inputs
~~~~~~~~~~~~
- ``PROMPTS: List[str]`` - Batch of text prompts.
- ``IMAGES: List[Union[PIL.Image.Image, torch.Tensor]]`` - Batch of image 
  prompts. We currently support one image per prompt as recommended by 
  `Meta <https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/vision_prompt_format.md#notes-1>`__. 
  If the prompt has no image, use an empty tensor.
- ``SAMPLING_PARAMS: List[Dict]``  - Batch of sampling parameters. With dynamic sampling, 
  you can pass different ``top_k``, ``top_p``, and ``temperature`` values for each 
  input in a batch.
  

Offline Example
~~~~~~~~~~~~~~~

::

    import torch
    import requests
    from PIL import Image

    from vllm import LLM, SamplingParams
    from vllm import TextPrompt

    from neuronx_distributed_inference.models.mllama.utils import add_instruct

    def get_image(image_url):
        image = Image.open(requests.get(image_url, stream=True).raw)
        return image


    # Configurations
    MODEL_PATH = "/home/ubuntu/model_hf/Llama-3.2-90B-Vision-Instruct-hf"
    BATCH_SIZE = 4
    SEQ_LEN = 2048
    TENSOR_PARALLEL_SIZE = 32
    CONTEXT_ENCODING_BUCKETS = [1024, 2048]
    TOKEN_GENERATION_BUCKETS = [1024, 2048]
    SEQUENCE_PARALLEL_ENABLED = False
    IS_CONTINUOUS_BATCHING = True
    ON_DEVICE_SAMPLING_CONFIG = {"global_topk":64, "dynamic": True, "deterministic": False}

    # Model Inputs
    PROMPTS = ["What is in this image? Tell me a story",
                "What is the recipe of mayonnaise in two sentences?" ,
                "Describe this image",
                "What is the capital of Italy famous for?",
                ]
    IMAGES = [get_image("https://github.com/meta-llama/llama-models/blob/main/models/scripts/resources/dog.jpg?raw=true"),
              torch.empty((0,0)),
              get_image("https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/nxd-inference-block-diagram.jpg"),
              torch.empty((0,0)),
              ]
    SAMPLING_PARAMS = [dict(top_k=1, temperature=1.0, top_p=1.0, max_tokens=256),
                       dict(top_k=1, temperature=0.9, top_p=1.0, max_tokens=256),
                       dict(top_k=10, temperature=0.9, top_p=0.5, max_tokens=512),
                       dict(top_k=10, temperature=0.75, top_p=0.5, max_tokens=1024),
                       ]


    def get_VLLM_mllama_model_inputs(prompt, single_image, sampling_params):
        # Prepare all inputs for mllama generation, including:
        # 1. put text prompt into instruct chat template
        # 2. compose single text and single image prompt into Vllm's prompt class
        # 3. prepare sampling parameters
        input_image = single_image
        has_image = torch.tensor([1])
        if isinstance(single_image, torch.Tensor) and single_image.numel() == 0:
            has_image = torch.tensor([0])

        instruct_prompt = add_instruct(prompt, has_image)
        inputs = TextPrompt(prompt=instruct_prompt)
        inputs["multi_modal_data"] = {"image": input_image}
        # Create a sampling params object.
        sampling_params = SamplingParams(**sampling_params)
        return inputs, sampling_params

    def print_outputs(outputs):
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


    if __name__ == '__main__':
        assert len(PROMPTS) == len(IMAGES) == len(SAMPLING_PARAMS), \
            f"""Text, image prompts and sampling parameters should have the same batch size, 
                got {len(PROMPTS)}, {len(IMAGES)}, and {len(SAMPLING_PARAMS)}"""

        # Create an LLM.
        llm = LLM(
            model=MODEL_PATH,
            max_num_seqs=BATCH_SIZE,
            max_model_len=SEQ_LEN,
            block_size=SEQ_LEN,
            device="neuron",
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            override_neuron_config={
                "context_encoding_buckets": CONTEXT_ENCODING_BUCKETS,
                "token_generation_buckets": TOKEN_GENERATION_BUCKETS,
                "sequence_parallel_enabled": SEQUENCE_PARALLEL_ENABLED,
                "is_continuous_batching": IS_CONTINUOUS_BATCHING,
                "on_device_sampling_config": ON_DEVICE_SAMPLING_CONFIG,
            }
        )

        batched_inputs = []
        batched_sample_params = []
        for pmpt, img, params in zip(PROMPTS, IMAGES, SAMPLING_PARAMS):
            inputs, sampling_params = get_VLLM_mllama_model_inputs(pmpt, img, params)
            # test batch-size = 1
            outputs = llm.generate(inputs, sampling_params)
            print_outputs(outputs)
            batched_inputs.append(inputs)
            batched_sample_params.append(sampling_params)

        # test batch-size = 4
        outputs = llm.generate(batched_inputs, batched_sample_params)
        print_outputs(outputs)


This script will print the outputs. Below is an example output from image-text prompt:

::

    Prompt: '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What is 
    in this image? Tell me a story<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', 
    Generated text: 'The image shows a dog riding a skateboard. The dog is standing on the 
    skateboard, which is in the middle of the road. The dog is looking at the camera with its 
    mouth open, as if it is smiling. The dog has floppy ears and a long tail. It is wearing a 
    collar around its neck. The skateboard is black with red wheels. The background is blurry, 
    but it appears to be a city street with buildings and cars in the distance.'


Online Example
~~~~~~~~~~~~~~
First, open a terminal and spin up a server of the model. If you specify a
new set of configurations, a new neuron model artifact will be compiled now.

::

    MODEL_PATH="/home/ubuntu/model_hf/Llama-3.2-90B-Vision-Instruct-hf"
    python3 -m vllm.entrypoints.openai.api_server \
        --model $MODEL_PATH \
        --tensor-parallel-size 32 \
        --max-model-len 2048 \
        --max-num-seqs 4 \
        --device neuron \
        --override-neuron-config '{
            "context_encoding_buckets": [1024, 2048], 
            "token_generation_buckets": [1024, 2048], 
            "sequence_parallel_enabled": false, 
            "is_continuous_batching": true, 
            "on_device_sampling_config": {
                "global_topk": 64, 
                "dynamic": true, 
                "deterministic": false
            }
        }'

If you see the below logs, that means your server is up and running:
::

    INFO: Started server process [284309]
    INFO: Waiting for application startup.
    INFO: Application startup complete.
    INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

Then open a new terminal as the client where you can send requests to the
server. We’ve enabled continuous batching by default, so you can open up to
``--max-num-seqs`` client terminals to send requests. To send a text-only request:
::
    MODEL_PATH="/home/ubuntu/model_hf/Llama-3.2-90B-Vision-Instruct-hf"
    curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{ 
            "model": "'"$MODEL_PATH"'",
            "messages": [ 
                    {
                    "role": "user", 
                    "content": "What is the capital of Italy?" 
                    } 
            ] 
            }'

You should receive outputs shown in the client terminal shortly:

::
    
    {"id":"chat-2df3e876738b470ab27b090e0a09736e","object":"chat.completion",
    "created":1734401826,"model":"/home/ubuntu/model_hf/Llama-3.2-90B-Vision-Instruct-hf/",
    "choices":[{"index":0,"message":{"role":"assistant","content":"The capital of Italy is 
    Rome.","tool_calls":[]},"logprobs":null,"finish_reason":"stop","stop_reason":null}],
    "usage":{"prompt_tokens":42,"total_tokens":50,"completion_tokens":8},"prompt_logprobs":null}



If the request fails, try setting ``export VLLM_RPC_TIMEOUT=180000`` environment variable. The timeout value depends on the
model and deployment configuration used.

To send a request with both text and image prompts:

::

    curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Describe this image"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_images/nxd-inference-block-diagram.jpg"
                }
                }
            ]
            }
        ]
        }'

You can expect results appear in the client terminal shortly:

::
    
    {"id":"chat-fd1319865bd44d6aa60a4739cce61c9d","object":"chat.completion",
    "created":1734401984,"model":"/home/ubuntu/model_hf/Llama-3.2-90B-Vision-Instruct-hf/",
    "choices":[{"index":0,"message":{"role":"assistant","content":"The image presents a 
    diagram illustrating the components of NxD Inference, with a focus on inference modules 
    and additional modules. The diagram is divided into two main sections: \"Inference 
    Modules\" and \"Additional Modules.\" \n\n**Inference Modules:**\n\n*   Attention 
    Techniques\n*   KV Caching\n*   Continuous Batching\n\n**Additional Modules:**\n\n*   
    Speculative Decoding (Draft model and Draft heads (Medusa / Eagle))\n\nThe diagram also 
    includes a section titled \"NxD Core (Distributed Strategies, Distributed Model Tracing)\" 
    and a logo for PyTorch at the bottom.","tool_calls":[]},"logprobs":null,
    "finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":14,"total_tokens":137,
    "completion_tokens":123},"prompt_logprobs":null}
