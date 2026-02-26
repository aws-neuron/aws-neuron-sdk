.. -*- mode: rst -*-

.. meta::
   :description: Learn how to get started with the {{ data.model.display_name }} model with Neuron, using recommended online and offline serving configurations.

.. _nxdi-models-{{ data.model.name | lower | replace(".", "-") | replace("/", "-") }}:

{{ data.model.display_name }}
=====================================

.. toctree::
   :hidden:

Learn how to get started with the {{ data.model.display_name }} model with Neuron, using recommended online and offline serving configurations. 

About {{ data.model.display_name }}
-------------------------------------------------------------------

{{ data.model.description }}

For detailed model specifications, capabilities, and checkpoints, see the official `{{ data.model.checkpoint }} <https://huggingface.co/{{ data.model.checkpoint }}>`_ model card on Hugging Face.

.. _nxdi-models-{{ data.model.name | lower | replace(".", "-") | replace("/", "-") }}-quickstart:

Quickstart
-----------------

The following examples show how to use {{ data.model.display_name }} with NeuronX Distributed Inference (NxDI) framework and vLLM for both online and offline use cases on Neuron devices.

.. admonition:: Before you start...
   :class: note

   Before running the sample code below, review how to set up your environment by following the :ref:`NxDI Setup Guide <nxdi-setup>`. Additionally, download the model checkpoint to a local directory of your choice (such as ``~/models/{{ data.model.name }}/``).

{%- macro render_nxdi_code(config) %}

.. code-block:: python
   :linenos:
   :emphasize-lines: 9,10,11,12{% for key, value in config.neuron.items() %}{% if key not in ["extra"] %},{{ loop.index + 12 }}{% endif %}{% endfor %}

   import torch
   from transformers import AutoTokenizer, GenerationConfig

   from neuronx_distributed_inference.models.config import NeuronConfig
   from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
   from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
   from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

   MODEL_PATH = "~/models/{{ data.model.name }}/"
   TRACED_MODEL_PATH = "~/traced_models/{{ data.model.name }}/"
   SEED = 0
   NEURON_CONFIG = NeuronConfig({% for key, value in config.neuron.items() %}{% if key not in ["extra"] %}
      {{ key }}={% if value is string %}"{{ value }}"{% else %}{{ value }}{% endif %},
      {%- endif %}{%- endfor %}
   )

   # Set random seed for reproducibility
   torch.manual_seed(SEED)

   # Initialize configs and tokenizer.
   generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
   eos = generation_config.eos_token_id
   generation_config_kwargs = {
      "do_sample": True,
      "top_k": 1,
      "pad_token_id": eos[0] if isinstance(eos, list) else eos,
   }
   generation_config.update(**generation_config_kwargs)
   config = LlamaInferenceConfig(NEURON_CONFIG, load_config=load_pretrained_config(MODEL_PATH))

   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
   tokenizer.pad_token = tokenizer.eos_token

   # Compile and save model.
   print("Compiling and saving model...")
   model = NeuronLlamaForCausalLM(MODEL_PATH, config)
   model.compile(TRACED_MODEL_PATH)
   tokenizer.save_pretrained(TRACED_MODEL_PATH)

   # Load from compiled checkpoint.
   print("Loading model from compiled checkpoint...")
   model = NeuronLlamaForCausalLM(TRACED_MODEL_PATH)
   model.load(TRACED_MODEL_PATH)

   # Generate outputs.
   print("Generating outputs...")
   prompts = ["I believe the meaning of life is", "The color of the sky is"]
   sampling_params = prepare_sampling_params(
      batch_size=NEURON_CONFIG.batch_size,
      top_k=[10, 5],
      top_p=[0.5, 0.9],
      temperature=[0.9, 0.5],
   )
   print(f"Prompts: {prompts}")

   inputs = tokenizer(prompts, padding=True, return_tensors="pt")
   generation_model = HuggingFaceGenerationAdapter(model)
   outputs = generation_model.generate(
      inputs.input_ids,
      generation_config=generation_config,
      attention_mask=inputs.attention_mask,
      max_length=model.config.neuron_config.max_length,
      sampling_params=sampling_params,
   )

   output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
   print("Generated outputs:")
   for i, output_token in enumerate(output_tokens):
      print(f"Output {i}: {output_token}")

{%- endmacro %}

{%- macro render_offline_code(config) %}

.. code-block:: python
   :linenos:
   :emphasize-lines: 9{% for key, value in config.vllm.items() %}{% if key != "extra" %},{{ loop.index + 9 }}{% endif %}{% endfor %}

   import os

   os.environ["VLLM_NEURON_FRAMEWORK"] = "neuronx-distributed-inference"

   from vllm import LLM, SamplingParams

   # Create an LLM.
   llm = LLM(
      model="~/models/{{ data.model.name }}/",{% for key, value in config.vllm.items() %}{% if key != "extra" %}
      {{ key }}={% if value is string %}"{{ value }}"{% else %}{{ value }}{% endif %},
      {%- endif %}{%- endfor %}
   )

   # Sample prompts.
   prompts = [
      "The president of the United States is",
      "The capital of France is",
      "The future of AI is",
   ]
   outputs = llm.generate(prompts, SamplingParams(top_k=1))

   for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

{%- endmacro %}

{%- macro render_online_code(config) %}

.. code-block:: bash
   :linenos:
   :emphasize-lines: 2,3{% for key, value in config.vllm.items() %}{% if key != "extra" %},{{ loop.index + 3 }}{% endif %}{% endfor %}

   vllm serve \
      --model="~/models/{{ data.model.name }}/"{% for key, value in config.vllm.items() %}{% if key != "extra" %} \
      --{{ key | replace("_", "-") }}{% if value is sameas true %}{% elif value is mapping %}='{{ value | tojson | replace("True", "true") | replace("False", "false") }}'{% elif value is string %}="{{ value }}"{% else %}={{ value }}{% endif %}{% endif %}{% endfor %} \
      --port=8080 

Once the vLLM server is online, submit requests using the example below:

.. literalinclude:: ../../examples/vllm_client.py
   :linenos:
   :language: python

{%- endmacro %}

.. tab-set::

   .. tab-item:: NxDI
      :selected:

      Select the instance type and make sure to update the highlighted code below to match your chosen path before you execute it.

      .. tab-set::
      {% for hardware_type, default_config in data.defaults.items() %}
         {%- set config = data.configurations[default_config.config] %}

         .. tab-item:: {{ hardware_type }}
            {% if loop.first %}:selected:{% endif %}

{{ render_nxdi_code(config) | indent(12, true) }}

      {% endfor %}

   .. tab-item:: Offline serving

      Select the instance type and make sure to update the highlighted code below to match your chosen path before you execute it.

      .. tab-set::
      {% for hardware_type, default_config in data.defaults.items() %}
         {%- set config = data.configurations[default_config.config] %}

         .. tab-item:: {{ hardware_type }}
            {% if loop.first %}:selected:{% endif %}

{{ render_offline_code(config) | indent(12, true) }}

      {% endfor %}

   .. tab-item:: Online serving

      Select the instance type and make sure to update the highlighted code below to match your chosen path before you execute it.

      .. tab-set::
      {% for hardware_type, default_config in data.defaults.items() %}
         {%- set config = data.configurations[default_config.config] %}

         .. tab-item:: {{ hardware_type }}
            {% if loop.first %}:selected:{% endif %}

{{ render_online_code(config) | indent(12, true) }}

      {% endfor %}

.. _nxdi-models-{{ data.model.name | lower | replace(".", "-") | replace("/", "-") }}-benchmarks:

{% if data.benchmarks %}
Benchmarks
------------------------

Select a metric to view performance benchmarks for various **batch sizes** and **input|output** sequence length combinations.

.. tab-set::

   .. tab-item:: Latency
      :sync: Latency

      Measured in: seconds (s)

      .. df-table::
         :header-rows: 1

         latency_data = {{ data.benchmarks.Latency | tojson }}
         df_raw = pd.DataFrame(latency_data)

         cols = [c for c in df_raw.columns if c not in ('neuron_config', 'batch_size')]

         df_grouped = df_raw.groupby('batch_size')[cols].min().round(3)
         df = df_grouped.reset_index()
         df.rename(columns={'batch_size': 'Batch Size'}, inplace=True)

   .. tab-item:: Throughput
      :sync: Throughput

      Measured in: tokens per second (tok/s)

      .. df-table::
         :header-rows: 1

         throughput_data = {{ data.benchmarks.Throughput | tojson }}
         df_raw = pd.DataFrame(throughput_data)

         cols = [c for c in df_raw.columns if c not in ('neuron_config', 'batch_size')]

         df_grouped = df_raw.groupby('batch_size')[cols].max().round(2)
         df = df_grouped.reset_index()
         df.rename(columns={'batch_size': 'Batch Size'}, inplace=True)

   .. tab-item:: TTFT
      :sync: TTFT

      Measured in: seconds (s)

      .. df-table::
         :header-rows: 1

         ttft_data = {{ data.benchmarks.TTFT | tojson }}
         df_raw = pd.DataFrame(ttft_data)

         cols = [c for c in df_raw.columns if c not in ('neuron_config', 'batch_size')]

         df_grouped = df_raw.groupby('batch_size')[cols].min().round(3)
         df = df_grouped.reset_index()
         df.rename(columns={'batch_size': 'Batch Size'}, inplace=True)

   .. tab-item:: ITL
      :sync: ITL

      Measured in: seconds (s)

      .. df-table::
         :header-rows: 1

         itl_data = {{ data.benchmarks.ITL | tojson }}
         df_raw = pd.DataFrame(itl_data)

         cols = [c for c in df_raw.columns if c not in ('neuron_config', 'batch_size')]

         df_grouped = df_raw.groupby('batch_size')[cols].min().round(5)
         df = df_grouped.reset_index()
         df.rename(columns={'batch_size': 'Batch Size'}, inplace=True)

.. admonition:: Tip
   :class: tip

   Further improvements and optimizations are possible through the :ref:`Neuron Kernel Interface (NKI) <neuron-nki>`.


{% endif %}

.. _nxdi-models-{{ data.model.name | lower | replace(".", "-") | replace("/", "-") }}-neuron-config:

Recommended configuration
--------------------------

{% if data.recommendations %}
Select a use case to view the recommended Neuron configuration. For the definitions of the flags listed below, see the :ref:`NxDI API reference guide <nxd-inference-api-guide>`.

{%- set throughput_config = data.configurations[data.recommendations.Throughput.config] %}
{%- set latency_config = data.configurations[data.recommendations.Latency.config] %}

.. tab-set::

   .. tab-item:: Offline serving
      :sync: Throughput

      For most use cases, the configuration below can be used to optimize **throughput** on Neuron devices. You can also increase the ``batch_size`` or use quantization to improve throughput even further. 

      {% if throughput_config.dp_degree != 1 %}
      For this specific configuration, we recommend using **Data Parallelism (DP) of {{ throughput_config.dp_degree }}**. For more details on how to implement data parallelism, refer to the :ref:`Data Parallelism on Trn2 <nxdi-trn2-llama3.3-70b-dp-tutorial>` tutorial.
      {% endif %}

      :bdg-info:`{{ throughput_config.instance_type }}`

      .. code-block:: python
         :linenos:

         NeuronConfig({% for key, value in throughput_config.neuron.items() %}{% if key not in ["extra"] %}
            {{ key }}={% if value is string %}"{{ value }}"{% else %}{{ value }}{% endif %},
            {%- endif %}{%- endfor %}
         )

   .. tab-item:: Online serving
      :sync: Latency

      For most use cases, the configuration below can be used to optimize **latency** on Neuron devices.

      {% if latency_config.dp_degree != 1 %}
      For this specific configuration, we recommend using **Data Parallelism (DP) of {{ latency_config.dp_degree }}**. For more details on how to implement data parallelism, refer to the :ref:`Data Parallelism on Trn2 <nxdi-trn2-llama3.3-70b-dp-tutorial>` tutorial.
      {% endif %}

      :bdg-info:`{{ latency_config.instance_type }}`

      .. code-block:: python
         :linenos:

         NeuronConfig({% for key, value in latency_config.neuron.items() %}{% if key not in ["extra"] %}
            {{ key }}={% if value is string %}"{{ value }}"{% else %}{{ value }}{% endif %},
            {%- endif %}{%- endfor %}
         )

{% else %}
.. note::

   The recommended configuration for the {{ data.model.display_name }} model is coming soon...

{% endif %}
