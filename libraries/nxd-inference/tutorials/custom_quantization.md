# Custom Quantization

## Overview

The document introduces a new customizable quantization feature in the NxD framework. This feature allows users to specify modules that should not be converted during quantization, by doing this they can run inference on custom quantized models. Take an un-quantized model and apply custom quantization, selecting specific layers to quantize or keep in full precision. 

The document also details how to use external APIs like `llmcompressor` for quantization, including setting up a quantization config and applying necessary patches. It also explains how to run inference with quantized models using specific flags and parameters, and how to specify modules to not convert via command-line arguments or NeuronConfig kwargs.


## Quantization

### Quantize Using NxD 

Quantization can significantly reduce the model size and inference time, making it more suitable for deployment on resource-constrained devices. However, not all layers benefit equally from quantization. 

* Some layers, especially those involved in critical computations like normalizations or certain types of activations, may see a significant drop in accuracy if quantized. Leaving these layers in full precision helps maintain the overall performance of the model. 
* Quantization can also introduce small errors in each layer’s computation. When these errors accumulate through the network, they can lead to a noticeable degradation in performance. Keeping certain layers in full precision can mitigate this accumulation.


To leverage the customizable quantization feature in NxD, follow the steps below. This process involves importing necessary libraries, defining the model and output paths, specifying modules to not convert, and utilizing a quantization function to create a quantized model. 

```
import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuronx_distributed_inference.modules.checkpoint import prune_state_dict,save_state_dict_safetensors
from neuronx_distributed.quantization.quantization_utils import quantize_pytorch_model_per_channel_symmetric, convert_qint8_to_int8_state_dict

model_path = "/<model_path/llama-3.1-405b-instruct-4layers/" 
output_path = "<save_quantized_checkpoints>"

modules_to_not_convert = [
    "lm_head",
    "layers.0.self_attn",
    "layers.1.self_attn",
    "layers.2.self_attn",
    "layers.1.mlp"
]

def quantize(model: torch.nn.Module, dtype=torch.qint8, modules_to_not_convert: Optional[List[str]] = None) -> torch.nn.Module:
    quant_model = quantize_pytorch_model_per_channel_symmetric(model,dtype=dtype, modules_to_not_convert=modules_to_not_convert)
    model_quant_sd = quant_model.state_dict()
    convert_qint8_to_int8_state_dict(model_quant_sd)
    quantized_state_dict = prune_state_dict(model_quant_sd)
    return quantized_state_dict
    
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

state_dict = quantize(model,torch.float8_e4m3fn,modules_to_not_convert)

save_state_dict_safetensors(state_dict=state_dict,state_dict_dir=output_path)
tokenizer.save_pretrained(output_path)

```

### Quantize using external API 

In addition to the built-in quantization features of NxD, users can also leverage external APIs for more flexible and advanced quantization options. One such API is `llmcompressor`, which offers a robust set of tools for quantizing models.
To use the `llmcompressor` API for quantization, follow the steps below. 

This process involves importing necessary libraries, specifying modules to not convert, setting up a quantization recipe, and applying the quantization to create a quantized model. It is important to ensure the scale range is set from -/+ 240 if you need to run inference on the quantized model later using NxD.

The `LLaMA` model is an example where not all layers are quantized. 

* By keeping the attention layers, first and last MLP layers, and the LM head in full precision, the model maintains high accuracy in tasks like language generation and comprehension.
* Quantizing the remaining layers (e.g., intermediate MLP layers) reduces the model size and inference time without significantly compromising performance.
* This strategy allows for a balanced trade-off between model efficiency and accuracy, making the model suitable for deployment on a variety of hardware, from powerful GPUs to edge devices.

```
import torch
from llmcompressor.transformers import oneshot, SparseAutoModelForCausalLM
from transformers import AutoTokenizer
from compressed_tensors.quantization.utils.helpers import calculate_range
from compressed_tensors.quantization.quant_args import QuantizationType
import compressed_tensors.quantization.utils.helpers as helpers

model_path = "/<model_path/llama-3.1-405b-instruct-4layers/" 
output_path = "<save_quantized_checkpoints>"

modules_to_not_convert = ['lm_head',
    "model.layers.0.mlp.down_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.3.mlp.down_proj",
    "model.layers.3.mlp.gate_proj",
    "model.layers.3.mlp.up_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.1.self_attn.k_proj",
    "model.layers.1.self_attn.o_proj",
    "model.layers.1.self_attn.q_proj",
    "model.layers.1.self_attn.v_proj",
    "model.layers.2.self_attn.k_proj",
    "model.layers.2.self_attn.o_proj",
    "model.layers.2.self_attn.q_proj",
    "model.layers.2.self_attn.v_proj",
    "model.layers.3.self_attn.k_proj",
    "model.layers.3.self_attn.o_proj",
    "model.layers.3.self_attn.q_proj",
    "model.layers.3.self_attn.v_proj"]

recipe = f"""
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: {modules_to_not_convert}
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: ["Linear"]
"""

model = SparseAutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto"
)

# Monkey patch to rescale weights from -/+448 to -/+240
original_calculate_range = helpers.calculate_range
def calculate_range(*args, **kwargs):
    q_min, q_max = original_calculate_range(*args, **kwargs)
    if args[0].type == QuantizationType.FLOAT and args[0].num_bits == 8:
        return torch.tensor(-240.0, device=args[1]), torch.tensor(240.0, device=args[1])
    return q_min, q_max

# Patch it
helpers.calculate_range = calculate_range
oneshot(model=model, recipe=recipe)

for name, module in model.named_modules():
    if hasattr(module, 'weight_scale'):
        module.weight_scale.data = module.weight_scale.data.to(torch.float32)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
```

## Quantization Commands

To utilize the quantization commands in NxD, users can follow the instructions below. These commands cover the required flags to enable running inference with quantized models.

### First Quantize then Inference

If you have a model in full precision and need to quantize it on the CPU first before using it for inference, you can set the following flags to enable quantization during inference:

```
inference_demo --model-type llama --task-type causal-lm run \
--model-path /your_model_path/ \
--compiled-model-path /save_to_path/ \
--torch-dtype bfloat16 \
--tp-degree 32 \
--batch-size 1 \
--max-context-length 1024 \
--quantized \
--quantization-dtype f8e4m3 \
--quantization-type per_channel_symmetric \
--quantized-checkpoints-path /save_to_path/ \
--seq-len 2048 \
--fused-qkv \
--pad-token-id 2 \
--on-device-sampling \
--sequence-parallel-enabled \
--attn-kernel-enabled \
--prompt "I believe the meaning of life is" \
--is-continuous-batching \
--enable-fused-speculation \
--enable-eagle-speculation \
--speculation-length 4  \
--draft-model-path /your_draft_model_path \
--modules-to-not-convert-file /path/modules_to_not_convert.json
```

### Inference Using Already quantized checkpoint

To utilize the quantization commands in NxD, users can follow the instructions below. These commands cover the required flags to enable running inference with quantized models. The `modules-to-not-convert-file` allows you to specify the list of modules to not quantize, useful for quantizing models that explicitly require having some modules left in their original precision.

### How to Use

* Pass `modules_to_not_convert` using Inference Demo

```
inference_demo --model-type llama --task-type causal-lm run \
    --model-path <path> \
    --compiled-model-path <path> \
    --torch-dtype bfloat16 \
    --tp-degree <value> \
    --batch-size <value> \
    --max-context-length <value> \
    --seq-len <value> \
    --on-device-sampling \
    --mlp-kernel-enabled \
    --quantized-mlp-kernel-enabled \
    --quantization-type <type> \
    --prompt "I believe the meaning of life is" \
    --modules-to-not-convert-file /<your_path>/modules_to_not_convert.json
```

* Pass `modules_to_not_convert` using NeuronConfig kwargs

```
neuron_config = NeuronConfig(
    tp_degree=32,
    batch_size=2,
    max_context_length=32,
    seq_len=64,
    on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
    enable_bucketing=True,
    flash_decoding_enabled=False,
    modules_to_not_convert=["lm_head", "layers.0.self_attn", "layers.1.mlp", ...],
    draft_model_modules_to_not_convert=["lm_head", "layers.0.self_attn", "layers.1.mlp", ..., "fc"]
)
```

>*Note: If you are creating different NeuronConfig for draft and target models, you only need to pass the modules_to_not_convert list for both.*

### JSON File Structure

The JSON structure is a crucial component for specifying which modules should not be converted during the quantization if you are using inference demo. This section provides detailed examples of how to format the JSON file. The JSON structure depends on whether fused speculation is used. 

1. Basic Structure

For simple cases:

```
{
    "modules_to_not_convert": [
            "lm_head",
            "layers.0.self_attn",
            "layers.1.self_attn",
            "layers.2.self_attn",
            "layers.3.self_attn",
            "layers.0.mlp",
            "layers.3.mlp"
    ]}
```

#### OR

```
{
    "model": {
        "modules_to_not_convert": [
            "lm_head",
            "layers.0.self_attn",
            "layers.1.self_attn",
            "layers.2.self_attn",
            "layers.3.self_attn",
            "layers.0.mlp",
            "layers.3.mlp"
        ]
    }}
```

1. With Fused Speculation

```
{
    "model": {
        "modules_to_not_convert": [
            "lm_head",
            "layers.0.self_attn",
            "layers.1.self_attn",
            "layers.2.self_attn",
            "layers.3.self_attn",
            "layers.0.mlp",
            "layers.3.mlp"
        ]
    },
    "draft_model": {
        "modules_to_not_convert": [
            "lm_head",
            "layers.0.self_attn",
            "layers.0.mlp",
            "fc"
        ]
    }}
```

### Important Notes

* Make sure to assign partial names in modules to avoid conversion, as shown in the examples above. This is necessary due to different naming schemes between the model layers being read from the source and the model we create for inference. The above examples include the partial parts of the names which are common between the two naming schemes.
    * For example: Original model names are like `model.layers.0.self_attn.q_proj`, whereas the names we give are like `layers.0.self_attn.qkv_proj.q_proj`
* Quantization with Fused Speculation
    * We currently do not quantize the draft model, Include these in the `draft_model.modules_to_not_convert` section of your JSON file

### Backward Incompatible Changes:

* We have changed the quantization state_dict keys from `weight_scale` to `scale` to match the NxD quantization scale keys and avoid any confusion. This will require reshard the weights if running with the old sharded weights.
* Now running the quantization workflow will need the `modules-to-not-convert-file` flag while running with `inference demo` because we no longer hard-code the layers to incorporate quantized layers.

