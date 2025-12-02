.. _nxdi-onboarding-models:

Onboarding models to run on NxD Inference
=========================================

This guide covers how to onboard a model to get it to run on NxD Inference
for the first time. To learn more about how to optimize a model on Neuron,
see the :ref:`nxdi-feature-guide`.

.. contents:: Table of contents
   :local:
   :depth: 2


Overview
--------

This guide demonstrates how to adapt an existing PyTorch model to run on
Neuron with the NeuronX Distributed (NxD) Inference library. At a
high-level, you will do the following:

1. Define configuration classes. NxD Inference models include a
   NeuronConfig, which defines Neuron-specific configuration parameters,
   and an InferenceConfig, which defines model configuration parameters.
   When adapting a model that works with HuggingFace, InferenceConfig is
   synonymous to PretrainedConfig.
2. Define model classes. When you define model classes, you replace
   linear layers with parallel layers that are optimized for distributed
   inference on Neuron. NxD Inference also provides modules for
   attention, KV cache management, and more, which you can use to write
   model classes that work with Neuron. Model classes are compiled to
   run effectively on Neuron.
3. Define application heads. Application heads orchestrate passing
   inputs to the correct compiled model. Application heads also provide
   the interface to compile and load the model.
4. Convert weights to a supported format. NxD Inference supports
   safetensors and pickle formats.


1. Define a NeuronConfig class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a Neuron configuration class, which extends NeuronConfig.
NeuronConfig includes Neuron-specific configuration parameters. In the
config class for your model, you can define any additional
Neuron-specific configuration parameters that your model requires.

- For MoE models, you can extend MoENeuronConfig instead of
  NeuronConfig. This class includes configuration parameters specific to
  MoE models.

::

   from neuronx_distributed_inference.models.config import NeuronConfig

   class NeuronLlamaConfig(NeuronConfig):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Set any args/defaults

2. Define an InferenceConfig class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define an inference configuration class, which extends InferenceConfig.
InferenceConfig includes model parameters, such as those from a
HuggingFace PretrainedConfig (like LlamaConfig). When users initialize
your config, they can provide required attributes directly, or they can
populate the config from a HuggingFace PretrainedConfig. You can also
override ``get_required_attributes`` to enforce that certain attributes
are present.

::

   from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig

   class LlamaInferenceConfig(InferenceConfig):
       def get_required_attributes(self) -> List[str]:
           return [
               "hidden_size",
               "num_attention_heads",
               "num_hidden_layers",
               "num_key_value_heads",
               "pad_token_id",
               "vocab_size",
               "max_position_embeddings",
               "rope_theta",
               "rms_norm_eps",
               "hidden_act",
           ]
           
       @classmethod
       def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
           return NeuronLlamaConfig

3. Define a Neuron model
~~~~~~~~~~~~~~~~~~~~~~~~

Define a Neuron model. This class is a subclass of NeuronBaseModel,
which is a PyTorch module.

1. In this class, you provide implementations for
   ``setup_attr_for_model(self, config)`` and
   ``init_model(self, config)``.

   1. In ``setup_attr_for_model``, set values for the following
      attributes. You can set these attributes from values in ``config``
      and ``config.neuron_config``.

      1. self.on_device_sampling
      2. self.tp_degree
      3. self.hidden_size
      4. self.num_attention_heads
      5. self.num_key_value_heads
      6. self.max_batch_size
      7. self.buckets

   2. In ``init_model``, initialize the modules that make up the model.

      1. For attention modules, extend NeuronAttentionBase, which
         provides a group query attention (GQA) implementation adapted
         to Neuron.
      2. Replace linear layers (such as in attention and MLP) with
         Neuron parallel layers (RowParallelLinear and
         ColumnParallelLinear).

         1. For more information about RowParallelLinear and
            ColumnParallelLinear layers, see :ref:`tensor_parallelism_overview`.

      3. Replace embeddings with Neuron parallel embeddings
         (ParallelEmbedding).
      4. Replace any other modules that require Neuron-specific
         implementations.

Note: This example demonstrates a simplified version of NeuronLlamaModel
from from the NxDI model hub.

::

   from torch import nn
   from transformers.activations import ACT2FN

   from neuronx_distributed.parallel_layers import parallel_state
   from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding

   from neuronx_distributed_inference.models.model_base import NeuronBaseModel
   from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
   from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
   from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

   class NeuronLlamaMLP(nn.Module):
       """
       This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
       """

       def __init__(self, config: InferenceConfig):
           super().__init__()
           self.config = config
           self.neuron_config = config.neuron_config
           self.tp_degree = config.neuron_config.tp_degree
           self.hidden_size = config.hidden_size
           self.intermediate_size = config.intermediate_size
           self.act_fn = ACT2FN[config.hidden_act]

           self.gate_proj = ColumnParallelLinear(
               self.hidden_size,
               self.intermediate_size,
               bias=False,
               gather_output=False,
               dtype=config.neuron_config.torch_dtype,
               pad=True,
           )
           self.up_proj = ColumnParallelLinear(
               self.hidden_size,
               self.intermediate_size,
               bias=False,
               gather_output=False,
               dtype=config.neuron_config.torch_dtype,
               pad=True,
           )
           self.down_proj = RowParallelLinear(
               self.intermediate_size,
               self.hidden_size,
               bias=False,
               input_is_parallel=True,
               dtype=config.neuron_config.torch_dtype,
               pad=True,
           )

       def forward(self, x):
           return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


   class NeuronLlamaAttention(NeuronAttentionBase):
       """
       Compared with LlamaAttention, this class just
       1. replaces the q_proj, k_proj, v_proj with column parallel layer
       2. replaces the o_proj with row parallel layer
       3. update self.num_head to be self.num_head / tp_degree
       4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
       5. update forward() method to adjust to changes from self.num_head
       """

       def __init__(self, config: InferenceConfig):
           super().__init__()

           self.config = config
           self.neuron_config = config.neuron_config
           self.hidden_size = config.hidden_size
           self.num_attention_heads = config.num_attention_heads
           self.num_key_value_heads = config.num_key_value_heads
           self.head_dim = self.hidden_size // self.num_attention_heads
           self.max_position_embeddings = config.max_position_embeddings
           self.rope_theta = config.rope_theta
           self.padding_side = config.neuron_config.padding_side
           self.torch_dtype = config.neuron_config.torch_dtype

           self.tp_degree = parallel_state.get_tensor_model_parallel_size()

           self.fused_qkv = config.neuron_config.fused_qkv
           self.clip_qkv = None

           self.init_gqa_properties()
           self.init_rope()

       def init_rope(self):
           self.rotary_emb = RotaryEmbedding(
               self.head_dim,
               max_position_embeddings=self.max_position_embeddings,
               base=self.rope_theta,
           )


   class NeuronLlamaDecoderLayer(nn.Module):
       """
       Just replace the attention with the NXD version, and MLP with the NXD version
       """

       def __init__(self, config: InferenceConfig):
           super().__init__()
           self.hidden_size = config.hidden_size
           self.self_attn = NeuronLlamaAttention(config)
           self.mlp = NeuronLlamaMLP(config)
           self.input_layernorm = CustomRMSNorm(
               config.hidden_size,
               eps=config.rms_norm_eps,
           )
           self.post_attention_layernorm = CustomRMSNorm(
               config.hidden_size,
               eps=config.rms_norm_eps,
           )

       def forward(
           self,
           hidden_states: torch.Tensor,
           attention_mask: Optional[torch.Tensor] = None,
           position_ids: Optional[torch.LongTensor] = None,
           past_key_value: Optional[Tuple[torch.Tensor]] = None,
           **kwargs,
       ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
           residual = hidden_states
           hidden_states = self.input_layernorm(hidden_states)

           # Self Attention
           attn_outs = self.self_attn(
               hidden_states=hidden_states,
               attention_mask=attention_mask,
               position_ids=position_ids,
               past_key_value=past_key_value,
               **kwargs,
           )

           hidden_states, present_key_value = attn_outs
           hidden_states = residual + hidden_states

           # Fully Connected
           residual = hidden_states
           hidden_states = self.post_attention_layernorm(hidden_states)
           hidden_states = self.mlp(hidden_states)
           hidden_states = residual + hidden_states

           return (hidden_states, present_key_value)


   class NeuronLlamaModel(NeuronBaseModel):
       """
       The neuron version of the LlamaModel
       """

       def setup_attr_for_model(self, config: InferenceConfig):
           # Needed for init_inference_optimization()
           self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
           self.tp_degree = config.neuron_config.tp_degree
           self.hidden_size = config.hidden_size
           self.num_attention_heads = config.num_attention_heads
           self.num_key_value_heads = config.num_key_value_heads
           self.max_batch_size = config.neuron_config.max_batch_size
           self.buckets = config.neuron_config.buckets

       def init_model(self, config: InferenceConfig):
           self.padding_idx = config.pad_token_id
           self.vocab_size = config.vocab_size

           self.embed_tokens = ParallelEmbedding(
               config.vocab_size,
               config.hidden_size,
               self.padding_idx,
               dtype=config.neuron_config.torch_dtype,
               shard_across_embedding=True,
               # We choose to shard across embedding dimension because this stops XLA from introducing
               # rank specific constant parameters into the HLO. We could shard across vocab, but that
               # would require us to use non SPMD parallel_model_trace.
               pad=True,
           )
           self.lm_head = ColumnParallelLinear(
               config.hidden_size,
               config.vocab_size,
               bias=False,
               pad=True,
           )

           self.layers = nn.ModuleList(
               [NeuronLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
           )
           self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

4. Define an application/task head
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define an application/task head. Applications includes causal LM,
classification, and so on. This class extends a task-specific Neuron
application head class (such as NeuronBaseForCausalLM), or the general
NeuronApplicationHead class.

1. In this class, you provide an value for ``_model_cls`` which is the
   Neuron model class you defined.
2. You can also override any other functions as needed for your model,
   such as ``get_compiler_args(self)`` or
   ``convert_hf_to_neuron_state_dict(model_state_dict, neuron_config)``.

Note: This example demonstrates a simplified version of
`NeuronLlamaForCausalLM <https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/llama/modeling_llama.py>`__
from the NxD Inference model hub.


::

   class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
       _model_cls = NeuronLlamaModel
           
       @classmethod
       def get_config_cls(cls):
           return LlamaInferenceConfig

NxD Inference offers :ref:`nxdi_async_mode_feature_guide` as an alternative method to executing NEFFs in parallel with CPU Logic. To evaluate if your
task can utilize ``async_mode``, the following questions must be answered:

1. Does your task repeatedly execute a model for a single user request? If not, then ``async_mode`` won't offer any benefits.
    - Example: The Auto Regressive loops used in LLMs perform repeated execution of models for a given prompt, which can get some benefits from async mode.
2. Does the output of one execution get passed onto the next execution without manipulation? If not, then ``async_mode`` is incompatible.
    - NOTE: It might be possible to address this by moving some manipulation logic within the neff.
    - Example: For LLMs using on-device-sampling, we pass in the token generated as output as input to the next step in the auto regressive loop directly. Without on-device-sampling, the sampling logic will rely on logits as output, which is a data dependent compute pattern that is incompatible with async mode.
3. Is there sufficient CPU logic that is independent of the previous outputs? If not, then ``async_mode`` likely won't offer major benefits.
    - Example: In production workloads, these are typically server overheads (scheduling, logging, etc.), but this could also be some pre/post processing steps in the model execution pipeline.
  
Based on the answers above, ``async_mode`` will need to be set accordingly, and/or, be configured to work correctly with the application.

1. Convert weights to a supported format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NxD Inference supports weights stored in the model path in the following
formats:

=========== ======= ============================
Format      Sharded File name
=========== ======= ============================
Safetensors No      model.safetensors
Safetensors Yes     model.safetensors.index.json
Pickle      No      pytorch_model.bin
Pickle      Yes     pytorch_model.bin.index.json
=========== ======= ============================

If your weights are in another format, you must convert them to one of
these formats before you can compile and load the model to Neuron. See
the following references for more information about these formats:

- Safetensors:

  - https://github.com/huggingface/safetensors
  - https://huggingface.co/docs/safetensors/en/convert-weights

- Pickle:

  - https://docs.python.org/3/library/pickle.html

.. _nxdi-onboarding-models-vllm:

Integrating Onboarded Model with vLLM
-------------------------------------

After completing the model onboarding in NxDI using the steps outlined 
in this guide, you can follow these steps to run that model through vLLM.

1. **Model Architecture**: Ensure your model follows standard NxDI naming 
   conventions (e.g., ``ModelNameForCausalLM``). The model is automatically 
   recognized through NxDI's ``MODEL_TYPES`` registry.

2. **Model Directory**: Use the local directory as ``model_name_or_path`` 
   when initializing vLLM. This directory should contain:
   
   - Model weights (safetensors or pickle format)
   - ``config.json`` file compatible with your InferenceConfig class

3. **Custom Configuration**: Pass any custom NeuronConfig attributes using 
   the ``override_neuron_config`` parameter when initializing the vLLM engine.

4. **Run Inference**: Execute offline or online inference using vLLM's 
   standard APIs to get your model working with vLLM.


.. _nxdi-evaluating-models:

Evaluating Models on Neuron
---------------------------

NxD Inference provides tools that you can use to
evaluate the accuracy and performance of the models that you onboard to
Neuron.

.. _nxdi-logit-matching:

Logit Matching
~~~~~~~~~~~~~~

The logit matching evaluation tool verifies that output logits are
within certain tolerances of expected logits. With this evaluation tool,
NxD Inference runs generation on the Neuron device.
Then, it compares the output logits against expected logits, which you
can provide or generate with the HuggingFace model on CPU.

During logit validation, if the output tokens diverge, then this process
runs generation on Neuron again, using the tokens up to the point where it diverged. This
process is performed repeatedly each time the output diverges, until the
entire output matches. This process uses greedy sampling to choose the
most likely token at each index.

Once all tokens match, this process compares the logits generated on
Neuron with the expected logits. If all logits are within expected
tolerances, this accuracy check passes. Divergence difference tolerance
is used to compare the logits at the token that diverges. Absolute and
relative tolerance are used to compare the values of the logits for the
top k highest scoring tokens. For best results, use a lower relative
tolerance for smaller k values, and a higher relative tolerance for
larger k values. A top k of ``None`` means to compare logits for all
possible tokens at each index.

Logit matching uses the following tolerances by default, and you can
customize these tolerances.

- Divergence difference tolerance: ``0.001``
- Absolute tolerance:

  - Top k = 5: ``1e-5``
  - Top k = 50: ``1e-5``
  - Top k = 1000: ``1e-5``
  - Top k = None: ``1e-5``

- Relative tolerance:

  - Top k = 5: ``0.01``
  - Top k = 50: ``0.02``
  - Top k = 1000: ``0.03``
  - Top k = None: ``0.05``

If all logits are within expected thresholds, this accuracy check
passes.

- Note: Logit matching cannot be used with on-device sampling.
- Note: Generating HuggingFace model outputs on CPU can take a
  significant amount of time for larger models or large sequence
  lengths.

Example (``check_accuracy_logits_v2`` API)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   from neuronx_distributed_inference.utils.accuracy import generate_expected_logits, check_accuracy_logits_v2

   # Init Neuron model, test inputs and HuggingFace generation config.

   # Generating HuggingFace model outputs on CPU.
   expected_logits = generate_expected_logits(
        neuron_model,
        inputs.input_ids,
        inputs.attention_mask,
        generation_config
    )
    # Alternatively, you can load the expected_logits from disk to save time.
    # expected_logits = ...

    check_accuracy_logits_v2(
        neuron_model,
        expected_logits,
        inputs.input_ids,
        inputs.attention_mask,
        generation_config=generation_config
    )

Example (``check_accuracy_logits`` API)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   from neuronx_distributed_inference.utils.accuracy import check_accuracy_logits

   # Init Neuron model, HuggingFace tokenizer, and HuggingFace generation config.

   check_accuracy_logits(
       model,
       tokenizer,
       generation_config,
   )

Token Matching
~~~~~~~~~~~~~~

The token matching evaluation tool verifies that output tokens match
expected tokens. With this evaluation tool, Neuronx Distributed
Inference runs generation on the Neuron device. Then, it compares the
output against expected tokens, which you can provide or generate with
the HuggingFace model on CPU. If all tokens match, this accuracy check
passes.

- Warning: Token mismatches are acceptable in many scenarios, especially
  with large models or large sequence lengths. This tool should only be
  used for small models and small sequence lengths.
- Note: Generating HuggingFace model outputs on CPU can take a
  significant amount of time for larger models or large sequence
  lengths.

Example (``check_accuracy`` API)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   from neuronx_distributed_inference.utils.accuracy import check_accuracy

   # Init Neuron model, HuggingFace tokenizer, and HuggingFace generation config.

   check_accuracy(
       model,
       tokenizer,
       generation_config,
   )

.. _nxdi-benchmark-sampling:

Benchmarking
~~~~~~~~~~~~

NxD Inference provides a benchmarking tool that
evaluates the latency and throughput of a Neuron model and its
sub-models (context encoding, token generation, etc.).

Example (``benchmark_sampling`` API)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

   # Init Neuron model and HuggingFace generation config.

   benchmark_sampling(model, generation_config)

Example benchmarking result
^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   {
       "e2e_model": {
           "latency_ms_p50": 28890.24031162262,
           "latency_ms_p90": 28977.734088897705,
           "latency_ms_p95": 28983.17071199417,
           "latency_ms_p99": 29032.21325159073,
           "latency_ms_p100": 29044.473886489868,
           "latency_ms_avg": 28879.499554634094,
           "throughput": 283.66142510545984
       },
       "context_encoding_model": {
           "latency_ms_p50": 705.0175666809082,
           "latency_ms_p90": 705.3698301315308,
           "latency_ms_p95": 705.6618571281433,
           "latency_ms_p99": 705.8443236351013,
           "latency_ms_p100": 705.8899402618408,
           "latency_ms_avg": 705.0377488136292,
           "throughput": 5809.618005408024
       },
       "token_generation_model": {
           "latency_ms_p50": 27.20165252685547,
           "latency_ms_p90": 27.295589447021484,
           "latency_ms_p95": 27.324914932250977,
           "latency_ms_p99": 27.655515670776367,
           "latency_ms_p100": 32.74345397949219,
           "latency_ms_avg": 27.19622969277793,
           "throughput": 147.22298324644066
       }
   }

Profiling Models
~~~~~~~~~~~~~~~~

Neuron provides a profiling tool, ``neuron-profile``, which you can use
to analyze the performance of a compiled Neuron model. For more
information, see :ref:`neuron-profile-ug`.

Evaluating Models with the Inference Demo Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NxD Inference provides an ``inference_demo`` console
script, which you can run from the environment where you install
``neuronx_distributed_inference``.

Note: Before you can use a custom model with the ``inference_demo``, you
must add it to the ``MODEL_TYPES`` dictionary in ``inference_demo.py``.

This script provides the following arguments to configure evaluation
tools:

- ``--check-accuracy-mode`` - Provide one of the following values:

  - ``token-matching`` - Perform a token matching accuracy check.
  - ``logit-matching`` - Perform a logit matching accuracy check.
  - ``skip-accuracy-check`` - Do not perform an accuracy check.

- ``--num-tokens-to-check`` - The number of tokens to check when performing
  token matching or logit matching.
- ``--expected-outputs-path`` - The path to a file that contains tokens or
  logits to compare against for the accuracy check. This file must contain
  an object saved with ``torch.save()``.
- ``--benchmark`` - Run benchmarking.
- ``--on-cpu`` - Run inference on CPU. To simulate tensor parallelism, 
  initialize ``inference_demo.py`` with ``torchrun``.

Debugging Models on Neuron
--------------------------

When you debug models on Neuron, you can enable debug logging to view
information about inputs and outputs of the NeuronBaseForCausalLM
forward function, which calls the NeuronBaseModel's forward function.

::

   import logging

   logging.getLogger().setLevel(logging.DEBUG)

Because the forward function of NeuronBaseModel is compiled, you cannot
use log/print statements to debug code that is called from this forward
function (or any other compiled code).

Debugging Neuron modeling code on CPU isn't yet supported.

Writing Tests on Neuron
-----------------------

NxD Inference provides tools to help you write unit and integration tests
that validate your model works as expected. For more information, see
:ref:`nxdi-writing-tests`.
