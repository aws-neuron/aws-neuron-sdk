.. _nxdi-trn2-llama3.1-405b-tutorial:

Tutorial: Deploying Llama3.1 405B (Trn2)
========================================

NeuronX Distributed (NxD) Inference enables you to deploy Llama3.1 405B on
a single Trn2 instance.

You can run Llama3.1 405B with default configuration options. NxD
Inference also provides several features and configuration options that
you can use to optimize and tune the performance of Llama3.1 405B on
Trn2. This guide walks through how to run Llama3.1 405B on Trn2 with
vLLM, and how to enable these optimizations for optimal performance. In addition, we also have a separate tutorial for running Llama3.1 405B with vanilla fused speculative decoding :ref:`nxdi-trn2-llama3.1-405b-speculative-tutorial`. 

.. contents:: Table of contents
   :local:
   :depth: 2

Background, Concepts, and Optimizations
---------------------------------------

Logical NeuronCore Configuration (LNC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On Trn2, the Neuron SDK supports *Logical NeuronCore Configuration
(LNC)*, which determines the number of NeuronCores visible to the Neuron SDK.
When running on Trn2, the Neuron SDK is optimized for LNC=2, which means
each NeuronCore visible to the Neuron SDK is two physical NeuronCores.
The LNC configuration also affects what TP degree options you can use.

NxD Inference automatically chooses the correct LNC configuration
based on the target platform.

For more information about LNC, see :ref:`logical-neuroncore-config`.

Tensor parallelism (TP) on Trn2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each Trn2 instance has 128 Neuron cores. With LNC=2, you can set a TP
degree up to 64. We recommend that you use LNC=2 for all models on Trn2.

For more information about tensor parallelism in NxD Inference, see
:ref:`nxdi-tensor-parallelism`.

Optimizing Performance
~~~~~~~~~~~~~~~~~~~~~~

EAGLE Speculative Decoding
^^^^^^^^^^^^^^^^^^^^^^^^^^

Speculative decoding is a performance optimization technique where a
smaller *draft* LLM model predicts the next tokens, and the larger *target*
LLM model verifies those predictions.

NxD Inference supports EAGLE v1 speculative decoding with a
flat draft structure. To use EAGLE v1, you must use an EAGLE checkpoint for a draft model 
that is not tree-based and is specifically fine-tuned for EAGLE speculation. For more
information about EAGLE, see the official implementation on GitHub: `SafeAILab/EAGLE <https://github.com/SafeAILab/EAGLE>`__.

To optimize performance for EAGLE speculative decoding, NxD Inference uses
a feature called *fused speculation*, where the
draft model and target model are fused into a single compiled model artifact
to improve performance. Fused speculation uses a different config called
FusedSpecNeuronConfig, which specifies the model class. draft config,
and draft model path to fuse with the target model.

For more information about speculative decoding in NxD Inference, including
other types of speculative decoding supported, see :ref:`nxd-speculative-decoding`.

FP8 Quantization
^^^^^^^^^^^^^^^^

NxD Inference supports FP8 quantization, where model weights and data
are converted to a smaller data type to reduce memory bandwidth usage.
FP8 quantization enables optimal usage of memory bandwidth to improve
model performance. For more information, see :ref:`nxdi-weight-quantization`.

NxD Inference also supports KV cache quantization, where the KV cache is
quantized to FP8. For more information, see :ref:`nxdi-kv-cache-quantization`.

Optimized Kernels
^^^^^^^^^^^^^^^^^

NxD Inference supports kernels that optimize parts of the modeling code
for best performance.

- Flash attention. This kernel uses a sharded flash attention
  implementation to improve performance during the context encoding
  pass. This kernel is enabled automatically at supported sequence
  lengths. For LNC2, NxD Inference automatically enables flash attention for sequence lengths of
  256 and larger that are divisible by 256. For LNC1, NxD Inference automatically enables flash attention
  for sequence lengths of 4096 and larger. You can also enable it with ``attn_kernel_enabled=True`` in
  NeuronConfig. NxD Inference automatically enables the flash attention kernel
  at supported sequence lengths even if ``attn_kernel_enabled`` is ``false``.
- QKV. This kernel fuses the QKV layers to improve performance during
  the attention forward pass. To enable this kernel, set
  ``qkv_kernel_enabled=True`` in NeuronConfig.
- MLP. This kernel implements the MLP module used in decoder layers. To
  enable this kernel, set ``mlp_kernel_enabled=True`` in NeuronConfig.
- Quantized MLP. This kernel implements a quantized version of the MLP
  kernel. This kernel uses FP8 compute to improve performance. To enable
  this kernel, set ``quantized_mlp_kernel_enabled=True``. This kernel requires
  ``mlp_kernel_enabled=True``.

.. note::
   To use the QKV and MLP kernels, you must set ``torch_dtype`` to ``torch.bfloat16``
   in NeuronConfig.

.. _nxdi-trn2-llama3.1-405b-running:

Tutorial: Run Llama3.1 405B on Trn2
-----------------------------------

As a prerequisite, this tutorial requires that you have a Trn2 instance
created from a Deep Learning AMI that has the Neuron SDK pre-installed.

To set up a Trn2 instance using Deep Learning AMI with pre-installed Neuron SDK,
see :ref:`nxdi-setup`.

Step 1: Connect to the Trn2 instance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use SSH to connect to the Trn2 instance using the key pair that you
chose when you launched the instance.

After you are connected, activate the Python virtual environment that
includes the Neuron SDK.

::

   source ~/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate

Run ``pip list`` to verify that the Neuron SDK is installed.

::

   python -m pip list

You should see Neuron packages including
``neuronx-distributed-inference`` and ``neuronx-cc``.

Step 2: Install the vLLM version that supports NxD Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NxD Inference supports running models with vLLM. This functionality is
available in a fork of the vLLM GitHub repository:

- `aws-neuron/upstreaming-to-vllm <https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.22-vllm-v0.7.2>`__

To run NxD Inference with vLLM, you download and install vLLM from this
fork. Clone the Neuron vLLM fork.

::
   
    git clone -b neuron-2.22-vllm-v0.7.2 https://github.com/aws-neuron/upstreaming-to-vllm.git


Activate the Neuron virtual environment.

::
    
    source ~/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate


Install the Neuron vLLM fork into the virtual environment.

::
    
    cd upstreaming-to-vllm
    pip install -r requirements-neuron.txt
    VLLM_TARGET_DEVICE="neuron" pip install -e .


Step 3: Deploy Llama 3.1 405B sample code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose one of the following examples to run on the Trn2 instance:

1. Deploy Llama3.1 405B with vLLM offline inference. This example demonstrates
   how to deploy on Trn2 with vLLM and topK sampling.

2. Deploy Llama3.1 405B with EAGLE speculative decoding. This example
   demonstrates how to use EAGLE to optimize Llama3.1 405B on Trn2.

Example 1: Deploy Llama3.1 405B on Trn2 with vLLM offline inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates how to deploy Llama3.1 405B on Trn2 with vLLM
offline inference and the following configuration options:

- Sequence length: 2048 tokens
- Max context length: 1024 tokens
- Speculation length: 6 tokens
- Flash attention, QKV, and MLP kernels
- On-device sampling with topK sampling

To use this sample, you must first download a 405B model checkpoint from Hugging Face
to a local path on the Trn2 instance. For more information, see
`Downloading models <https://huggingface.co/docs/hub/en/models-downloading>`__
in the Hugging Face documentation. You can download and use `meta-llama/Llama-3.1-405B-Instruct <https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct>`__
for this tutorial.

::

   import os
   import torch
   
   from vllm import LLM, SamplingParams
   
   # Force vLLM framework to use neuronx-distributed-inference
   os.environ['VLLM_NEURON_FRAMEWORK'] = "neuronx-distributed-inference"
   
   model_path = "/home/ubuntu/models/Llama-3.1-405B-Instruct/"
   
   
   def run_llama_generate():
       # Initialize vLLM.
       llm = LLM(
           model=model_path,
           tensor_parallel_size=64,
           max_num_seqs=1,
           max_model_len=2048,
           block_size=2048,
           dtype=torch.bfloat16,
           # Configure NeuronConfig.
           override_neuron_config={
               "max_context_length": 1024,
               "skip_warmup": True,
           },
           device="neuron"
       )
   
       # Run vLLM to generate outputs.
       prompts = ["I believe the meaning of life is"]
       sampling_params = SamplingParams(top_k=50)
       outputs = llm.generate(prompts, sampling_params)
       for output in outputs:
           prompt = output.prompt
           generated_text = output.outputs[0].text
           print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
   
   
   if __name__ == "__main__":
       run_llama_generate()

Example 2: Deploy Llama3.1 405B on Trn2 with EAGLE speculative decoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates how to deploy Llama3.1 405B on Trn2 with EAGLE
speculative decoding.

.. note::
   To use this example, you must provide an EAGLE-trained Llama3.1 405B
   checkpoint to use for EAGLE speculative decoding. For more information
   about EAGLE checkpoint compatibility with NxD Inference, see :ref:`nxd-eagle-speculative-decoding`.

This example uses the following configuration options:

- Sequence length: 2048 tokens
- Max context length: 1024 tokens
- Speculation length: 6 tokens
- Flash attention, QKV, and MLP kernels
- On-device sampling with greedy sampling
- Sequence parallelism enabled
- Auto-bucketing enabled, which automatically selects buckets to use.
  For more information about bucketing and how to customize the buckets used,
  see :ref:`nxdi-bucketing`.

::

   import copy
   import os
   import torch
   
   from transformers import AutoTokenizer, GenerationConfig
   
   from neuronx_distributed_inference.models.config import FusedSpecNeuronConfig, NeuronConfig, OnDeviceSamplingConfig
   from neuronx_distributed_inference.models.llama.modeling_llama import LlamaInferenceConfig, NeuronLlamaForCausalLM
   from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
   
   model_path = "/home/ubuntu/models/llama-3.1-405b-Instruct/"
   draft_model_path = "/home/ubuntu/models/EAGLE-llama-3-405b/"
   compiled_model_path = "/home/ubuntu/neuron_models/llama-3-405b-instruct-EAGLE/"
   
   # Set environment variables for Trn2.
   os.environ["XLA_DENSE_GATHER_FACTOR"] = "0"
   os.environ["NEURON_RT_EXEC_TIMEOUT"] = "600"
   
   def run_llama_generate():
       top_k = 1
       do_sample = False
   
       # Initialize tokenizer.
       tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
       tokenizer.pad_token = tokenizer.eos_token
   
       # Initialize target model config.
       neuron_config = NeuronConfig(
           torch_dtype=torch.bfloat16,
           tp_degree=64,
           batch_size=1,
           max_context_length=1024,
           seq_len=2048,
           on_device_sampling_config=OnDeviceSamplingConfig(
               dynamic=False,
               do_sample=do_sample,
               top_k=top_k
           ),
           enable_eagle_speculation=True,
           enable_fused_speculation=True,
           speculation_length=6,
           trace_tokengen_model=False,
           enable_bucketing=True,
           fused_qkv=True,
           sequence_parallel_enabled=True,
           attn_kernel_enabled=True,
           qkv_kernel_enabled=True,
           mlp_kernel_enabled=True,
           cc_pipeline_tiling_factor=1,
       )
       config = LlamaInferenceConfig(
           neuron_config,
           load_config=load_pretrained_config(model_path),
       )
   
       # Initialize draft model config.
       draft_neuron_config = copy.deepcopy(neuron_config)
       draft_neuron_config.trace_tokengen_model = True
       draft_neuron_config.enable_fused_speculation = False
       draft_neuron_config.is_eagle_draft = True
       draft_neuron_config.sequence_parallel_enabled = False
       draft_config = LlamaInferenceConfig(
           draft_neuron_config,
           load_config=load_pretrained_config(draft_model_path)
       )
   
       # Initialize fused speculation config.
       fused_spec_config = FusedSpecNeuronConfig(
           NeuronLlamaForCausalLM._model_cls,
           draft_config=draft_config,
           draft_model_path=draft_model_path,
       )
       config.fused_spec_config = fused_spec_config
           
       # Compile and save model.
       print("\nCompiling and saving model...")
       model = NeuronLlamaForCausalLM(model_path, config)
       model.compile(compiled_model_path)
       tokenizer.save_pretrained(compiled_model_path)
   
       # Load from compiled checkpoint.
       print("\nLoading model from compiled checkpoint...")
       model = NeuronLlamaForCausalLM(compiled_model_path)
       model.load(compiled_model_path)
       tokenizer = AutoTokenizer.from_pretrained(compiled_model_path)
   
       # Initialize generation config.
       generation_config = GenerationConfig.from_pretrained(model_path)
       generation_config_kwargs = {
           "do_sample": do_sample,
           "top_k": top_k,
           "pad_token_id": 0,
           "prompt_lookup_num_tokens": neuron_config.speculation_length,
       }
       generation_config.update(**generation_config_kwargs)
   
       # Generate outputs.
       print("\nGenerating outputs...")
       prompts = ["I believe the meaning of life is"]
       print(f"Prompts: {prompts}")
       inputs = tokenizer(prompts, padding=True, return_tensors="pt")
       generation_model = HuggingFaceGenerationAdapter(model)
       outputs = generation_model.generate(
           inputs.input_ids,
           generation_config=generation_config,
           attention_mask=inputs.attention_mask,
           max_length=model.config.neuron_config.max_length,
       )
       output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
       print("Generated outputs:")
       for i, output_token in enumerate(output_tokens):
           print(f"Output {i}: {output_token}")
   
   
   if __name__ == "__main__":
       run_llama_generate()