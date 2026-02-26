.. meta::
   :description: Launch the vLLM OpenAI-compatible server on AWS Neuron for interactive inference.
   :date_updated: 2025-01-15

.. _quickstart-online-serving:

Quickstart: Serve models online with vLLM on Neuron
===================================================

This quickstart shows you how to launch the vLLM OpenAI-compatible API server on AWS Neuron. You install the ``vllm-neuron`` plugin, start the server, validate it with ``curl``, and call it from the OpenAI Python SDK.

**This quickstart is for**: Developers who need an interactive, low-latency serving endpoint on Neuron  
**Time to complete**: ~20 minutes

Prerequisites
-------------

Before you begin, make sure you have:

* An EC2 instance with Neuron cores and network access to Hugging Face Hub.
* The Neuron SDK installed (see :ref:`Setup Instructions<nxdi-setup>`).
* Python 3.10 or later with ``pip``.
* Basic familiarity with running Python scripts in a virtual environment.

.. note::
   For the fastest setup, consider the vLLM Neuron Deep Learning Container (DLC), which bundles the SDK, vLLM, and dependencies. See :ref:`quickstart_vllm_dlc_deploy`.

Step 1: Install the ``vllm-neuron`` plugin
-------------------------------------------

In this step, you install the Neuron-enabled vLLM plugin inside your Python environment.

.. code-block:: bash

   # Activate your Neuron virtual environment
   source ~/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate

   # Clone the vLLM Neuron plugin repository
   git clone https://github.com/vllm-project/vllm-neuron.git
   cd vllm-neuron

   # Install with the Neuron package repository
   pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .

.. important::
   The ``--extra-index-url`` flag pulls the Neuron-compatible wheels from the AWS repository.

To confirm the install succeeded, run ``python -c "import vllm"`` and verify no errors display.

Step 2: Launch the API server
-----------------------------

In this step, you start an OpenAI-compatible endpoint with a LLaMA model.

.. code-block:: bash

    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"

    vllm serve \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --tensor-parallel-size 8 \
      --max-model-len 128 \
      --max-num-seqs 4 \
      --no-enable-prefix-caching \
      --port 8000 \
      --additional-config '{
        "override_neuron_config": {
          "enable_bucketing": false
        }
      }'

Key arguments:

* ``--tensor-parallel-size``: Matches the number of Neuron cores you want to use.
* ``--max-model-len`` and ``--max-num-seqs``: Duplicate limits from your offline workflow.
* ``--additional-config``: Wrap Neuron overrides under ``override_neuron_config`` (``enable_bucketing`` here).
* ``--port``: Choose the listening port for the API server.

Step 3: Verify the endpoint with ``curl``
------------------------------------------

In this step, you confirm the server is responding by sending a chat completion request.

.. code-block:: bash

    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [
              {"role": "system", "content": "You are a concise assistant."},
              {"role": "user", "content": "List three Neuron optimization tips."}
            ],
            "temperature": 0.2
          }'

If successful, the server returns a JSON payload containing the generated answer.

Step 4: Call the API with the OpenAI SDK
-----------------------------------------

Now that the server is live, call it using the OpenAI Python SDK.

.. code-block:: python

    from openai import OpenAI

    # Client setup
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )

    models = client.models.list()
    model_name = models.data[0].id

    max_tokens = 50
    temperature = 1.0
    top_p = 1.0
    top_k = 50
    stream = False

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello, my name is Llama"}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        extra_body={"top_k": top_k},
    )

    generated_text = ""
    if stream:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                generated_text += chunk.choices[0].delta.content
    else:
        generated_text = response.choices[0].message.content

    print(generated_text)

Step 5: Explore advanced configuration (optional)
-------------------------------------------------

The commands below show optional tuning features to adapt the server for different workloads.

**Reuse compiled models to avoid recompilation**:

.. code-block:: bash

    # Create directory for compiled artifacts if it doesn't exist
    mkdir -p ./neuron_compiled_models/llama3-8b
    
    # Set the environment variable before launching the server
    export NEURON_COMPILED_ARTIFACTS="./neuron_compiled_models/llama3-8b"

**Enable prefix caching when prompts share a long context**:

.. code-block:: bash

    vllm serve \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --tensor-parallel-size 32 \
      --max-model-len 1024 \
      --max-num-seqs 8 \
      --enable-prefix-caching \
      --block-size 32 \
      --num-gpu-blocks-override 256 \
      --additional-config '{
        "override_neuron_config": {
          "is_prefix_caching": true,
          "is_block_kv_layout": true,
          "pa_num_blocks": 256,
          "pa_block_size": 32
        }
      }' \
      --port 8000

**Use Eagle speculative decoding when you have an EAGLE checkpoint available**:

Below is an example of how to run vLLM inference with an EAGLE V1 checkpoint

.. note::
   Eagle draft checkpoints must be converted for NxD Inference compatibility and include the target model's LM head. Follow the guidance at `EAGLE Checkpoint Compatibility <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html#eagle-checkpoint-compatibility>`_.

.. code-block:: bash

    vllm serve \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --tensor-parallel-size 32 \
      --max-model-len 2048 \
      --max-num-seqs 4 \
      --speculative-config '{"model": "./eagle_draft_converted", "method": "eagle", "num_speculative_tokens": 5, "max_model_len": 2048}' \
      --port 8000

**Use MultiLoRA**:

.. note::
   For multi-LoRA serving, you can optionally create a JSON configuration file that maps LoRA adapter IDs to their checkpoint paths. This enables dynamic adapter loading and swapping between HBM and host memory.

.. code-block:: bash

    # Example JSON configuration (save as lora_config.json):
    # {
    #   "lora-ckpt-dir": "/opt/lora/tinyllama/",
    #   "lora-ckpt-paths": {
    #     "tarot_adapter": "tarot",
    #     "support_adapter": "mental-health"
    #   },
    #   "lora-ckpt-paths-cpu": {
    #     "tarot_adapter": "tarot",
    #     "support_adapter": "mental-health"
    #   }
    # }

    vllm serve \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --tensor-parallel-size 32 \
      --max-model-len 1024 \
      --max-num-seqs 2 \
      --enable-lora \
      --max-loras 2 \
      --max-cpu-loras 4 \
      --lora-modules \
        tarot_adapter=/opt/lora/tinyllama/tarot \
        support_adapter=/opt/lora/tinyllama/mental-health \
      --additional-config '{"override_neuron_config": {"lora_ckpt_json": "/path/to/lora_config.json"}}' \
      --port 8000

Clients can select an adapter per request by setting the ``model`` field to the adapter ID in the request. The ``max-loras`` parameter controls concurrent adapters in HBM, while ``max-cpu-loras`` controls adapters in host memory with dynamic swapping support.

**Tune context and token buckets for long prompts**:

.. code-block:: bash

    export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"

    vllm serve \
      --model "meta-llama/Llama-3.1-8B-Instruct" \
      --tensor-parallel-size 32 \
      --max-num-seqs 1 \
      --max-model-len 1024 \
      --port 8000 \
      --no-enable-prefix-caching \
      --additional-config '{
        "override_neuron_config": {
          "enable_bucketing": true,
          "context_encoding_buckets": [256, 512, 1024],
          "token_generation_buckets": [32, 64, 128, 256, 512, 768],
          "max_context_length": 1024,
          "seq_len": 1024,
          "batch_size": 1,
          "ctx_batch_size": 1,
          "tkg_batch_size": 1,
          "is_continuous_batching": true
        }
      }'

Set ``NEURON_COMPILED_ARTIFACTS`` before launching if you want to reuse artifacts across runs.

Confirmation
------------

Resend the ``curl`` request from Step 3 or rerun the OpenAI SDK snippet from Step 4. Successful responses confirm that the server is up and reachable. You can also open ``http://localhost:8000/health`` to check the health probe.

Common issues
-------------

- **Server exits immediately**: Confirm ``--tensor-parallel-size`` matches the number of available Neuron cores.
- **Requests return 5xx errors**: Lower ``--max-num-seqs`` or ``--max-model-len`` if the model runs out of memory.
- **Initial requests take too long**: Set ``NEURON_COMPILED_ARTIFACTS`` so subsequent launches reuse compiled artifacts.

Clean up
--------

Stop the API server (Ctrl+C). Deactivate your Python environment with ``deactivate``. Remove the cloned ``vllm-neuron`` repository if you no longer need it, and clear any cached artifacts if disk space is a concern.

Next steps
----------

* Explore prefix caching, Eagle speculative decoding, and other options in :ref:`nxdi-feature-guide`.
* Review supported model architectures in :ref:`nxdi-supported-model-architectures`.
* Try the offline batch quickstart (:ref:`quickstart-offline-serving`) if you need non-interactive inference.

Further reading
---------------

- :ref:`nxdi-vllm-user-guide-v1` – Complete integration reference.
- :ref:`nxdi-tutorials-index` – In-depth tutorials and workflow guides.
- `OpenAI Python SDK reference <https://github.com/openai/openai-python>`_ – API documentation for the client used in Step 4.
