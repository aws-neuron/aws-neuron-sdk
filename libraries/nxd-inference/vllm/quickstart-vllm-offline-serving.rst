.. meta::
   :description: Learn how to run your first offline vLLM batch inference job on AWS Neuron.
   :date_updated: 2025-12-02

.. _quickstart-offline-serving:

Quickstart: Run offline inference with vLLM on Neuron
======================================================

This quickstart walks you through running vLLM in offline (batch) inference mode on AWS Neuron. You install the ``vllm-neuron`` plugin, generate text for a batch of prompts, and cache the compiled artifacts so reruns stay fast.

**This quickstart is for**: Developers who want to run offline/batch inference on Neuron without an API server  
**Time to complete**: ~20 minutes

Prerequisites
-------------

Before you begin, make sure you have:

* An EC2 instance with Neuron cores and network access to Hugging Face Hub.
* The Neuron SDK installed (see :ref:`Setup Instructions<nxdi-setup>`).
* Python 3.10 or later with ``pip``.
* Basic familiarity with running Python scripts in a virtual environment.

.. note::
   For the fastest setup, consider the vLLM Neuron Deep Learning Container (DLC) which bundles the SDK, vLLM, and dependencies. See :ref:`quickstart_vllm_dlc_deploy`.

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
   The ``--extra-index-url`` flag ensures Neuron-compatible wheels are pulled from the AWS repository.

To confirm the install succeeded, run ``python -c "import vllm"`` and verify no errors display.

Step 2: Run a batch inference job
---------------------------------

In this step, you run a short Python script that generates completions for three prompts using the Llama 3.1 8B Instruct model.

.. tip::
   **Before your first run**, set the ``NEURON_COMPILED_ARTIFACTS`` environment variable to enable caching. This lets subsequent runs skip the Neuron compilation phase and load instantly:

   .. code-block:: bash

      export NEURON_COMPILED_ARTIFACTS="./compiled_models"

   After the first run completes, the ``compiled_models`` directory will contain the cached artifacts.

.. code-block:: python

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=32,
        max_num_seqs=1,
        max_model_len=128,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        additional_config={
            "override_neuron_config": {
                "skip_warmup": True,
            },
        },
    )

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm.generate(prompts, SamplingParams(top_k=10))
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")

If the script succeeds, you will see each prompt followed by generated text in the console.

Step 3: Optimize model loading with sharded checkpoints
-------------------------------------------------------

In this step, you configure vLLM to save sharded checkpoints, which significantly speeds up model loading on subsequent runs.

By default, vLLM shards the model weights during every load, which can take considerable time. Setting ``save_sharded_checkpoint: true`` saves the sharded weights to disk after the first run, eliminating this overhead.

.. code-block:: python

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=32,
        max_num_seqs=1,
        max_model_len=128,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        additional_config={
            "override_neuron_config": {
                "skip_warmup": True,
                "save_sharded_checkpoint": true,
            },
        },
    )

After the first run, the sharded checkpoint is saved alongside your model files. Subsequent runs will load the pre-sharded weights directly, reducing initialization time.

Step 4: Try advanced configuration options (optional)
-----------------------------------------------------

In this step, you explore optional tuning features that can improve throughput for specific workloads.

**Enable prefix caching when prompts share a long system prefix**:

.. note::
   To understand how to configure prefix caching parameters like ``num_gpu_blocks_override``, ``block_size``, ``pa_num_blocks``, and ``pa_block_size``, 
   see the `Llama 3.3 70B prefix caching tutorial <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/trn2-llama3.3-70b-apc-tutorial.html#Scenario-1:-Run-Llama3.3-70B-on-Trn2-without-Prefix-Caching>`_.

.. code-block:: python

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=32,
        max_num_seqs=4,
        max_model_len=2048,
        num_gpu_blocks_override=4096,
        block_size=32,
        enable_prefix_caching=True,
        additional_config={
            "override_neuron_config": {
                "is_prefix_caching": True,
                "is_block_kv_layout": True,
                "pa_num_blocks": 4096,
                "pa_block_size": 32,
                "skip_warmup": True,
            },
        },
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
    ]

    outputs = llm.generate(prompts, SamplingParams(temperature=0.0))

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")

**Use Eagle speculative decoding when you have an EAGLE checkpoint available**:

Below is an example of how to run vLLM inference with an EAGLE V1 checkpoint

.. note::
   Eagle draft checkpoints must be converted for NxD Inference compatibility and include the target model's LM head. Follow the guidance at `EAGLE Checkpoint Compatibility <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html#eagle-checkpoint-compatibility>`_.

.. code-block:: python

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tensor_parallel_size=32,
        max_num_seqs=4,
        max_model_len=256,
        speculative_config={
            "model": "./eagle_draft_converted",
            "num_speculative_tokens": 5,
            "max_model_len": 256,
            "method": "eagle",
        },
    )

    prompts = [
        "The key benefits of cloud computing are",
        "Python is a popular programming language because",
        "Machine learning models can be improved by",
    ]

    outputs = llm.generate(prompts, SamplingParams(top_k=50, max_tokens=100))

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")

Confirmation
------------

Re-run the script from Step 2. You should see completions printed again, and the log will indicate:

* Compiled artifacts were loaded from cache (if ``NEURON_COMPILED_ARTIFACTS`` is set)
* Sharded checkpoint was loaded directly (if ``save_sharded_checkpoint: true`` was used)

If you enable Neuron debug logging, look for ``Loaded Neuron compiled artifacts`` messages.

Common issues
-------------

- **Initial run takes too long**: Set ``NEURON_COMPILED_ARTIFACTS`` before running so the second run reuses the cache.
- **Model loading is slow on every run**: Enable ``save_sharded_checkpoint: true`` in ``override_neuron_config`` to avoid re-sharding the model weights each time.
- **Warmup adds latency**: Keep ``skip_warmup=True`` in ``override_neuron_config`` if your workload does not require the warmup pass.

Clean up
--------

Deactivate your Python environment with ``deactivate``. Delete the ``compiled_models`` directory if you no longer need the cached artifacts. Remove any sharded checkpoint directories created by ``save_sharded_checkpoint``. Remove the cloned ``vllm-neuron`` repository if finished testing.

Next steps
----------

* Explore prefix caching, Eagle speculative decoding, and other options in :ref:`nxdi-feature-guide`.
* Review supported model architectures in :ref:`nxdi-supported-model-architectures`.
* Switch to the online serving quickstart (:ref:`quickstart-online-serving`) when you need an API endpoint.

Further reading
---------------

- :ref:`nxdi-vllm-user-guide-v1`: Complete integration reference.
- :ref:`nxdi-tutorials-index`: In-depth tutorials and workflow guides.
- `Downloading models from Hugging Face <https://huggingface.co/docs/hub/en/models-downloading>`_: Instructions for obtaining model checkpoints.
