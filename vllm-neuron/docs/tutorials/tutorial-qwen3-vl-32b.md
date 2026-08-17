# Tutorial: Deploy Qwen3-VL-32B with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying Qwen3-VL-32B with vLLM
on Neuron, covering environment setup, model download, optional MXFP8
quantization, online serving, and offline inference for the multimodal
Qwen3-VL-32B model on Trn2 (BF16) and Trn3 (BF16 or MXFP8). -->
<!-- meta: keywords: vLLM, Neuron, Qwen3-VL, Qwen3-VL-32B, multimodal, VLM,
vision-language, MXFP8, quantization, LLM-Compressor, tutorial, LLM serving,
Trn2, Trn3, Trainium -->
<!-- meta: date_updated: 2026-07-29 -->
<!-- Content type: procedural-tutorial -->
<!-- Jira: NDOC-183 -->

This tutorial walks through deploying [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) with vLLM Neuron. It covers environment setup, model download, an optional MXFP8 quantization step, online serving, and offline inference.

The model runs in two precisions:

- **BF16** — the default, on `trn2.48xlarge` or Trn3.
- **MXFP8** — micro-scaled FP8 weights for the text transformer, **Trn3-only**. Reduces the weight footprint and improves both throughput and latency while keeping accuracy close to BF16.

| Precision | Instance | Steps |
|-----------|----------|-------|
| BF16 | Trn2 or Trn3 | 1 → 2 → 4 (skip the MXFP8-only Step 3) |
| MXFP8 | Trn3 only | 1 → 2 → **3** → 4 |

BF16 is the main path throughout; each step calls out the MXFP8 differences where they apply. Step 3 (quantization) is only for MXFP8 — skip it for BF16.

**Prerequisites:**

- A `trn2.48xlarge` instance with Neuron SDK `2.31.0` or later. See
  [setup guide](../getting-started/setup-guide.md). MXFP8 requires a **Trn3** instance (the native MXFP8 kernels are Trn3-only) with Neuron SDK `2.32` or later.
- vLLM Neuron plugin `0.21.0` or above installed.
- Python 3.10+
- For MXFP8, [LLM-Compressor](https://github.com/vllm-project/llm-compressor) to quantize the checkpoint: `pip install llmcompressor compressed-tensors`

## Step 1: Set up your environment

Verify Neuron devices are visible:

```bash
neuron-ls
# Lists the Neuron devices and cores available on your instance.
```

Set environment variables before running any inference script:

```bash
# Extend timeouts for large model compilation
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export NEURON_LIBTORCH_COMPILATION_TIMEOUT=1200

# Required if your home directory is on NFS
export NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp"
mkdir -p /tmp/neuroncc_tmp
```

## Step 2: Download the model (optional)

```bash
huggingface-cli download \
    Qwen/Qwen3-VL-32B-Instruct \
    --local-dir /path/to/Qwen3-VL-32B-Instruct
```

> **Note:** This step is optional. You can pass the Hugging Face model ID (e.g. `Qwen/Qwen3-VL-32B-Instruct`) directly to `vllm serve`, and the weights will be downloaded automatically on first run.

## Step 3: Quantize the text model to MXFP8 (MXFP8 only)

**Skip this step for BF16.** Qwen3-VL is not published with MXFP8 weights, so you must quantize the BF16 checkpoint offline. This uses [LLM-Compressor](https://github.com/vllm-project/llm-compressor) and [compressed-tensors](https://github.com/vllm-project/compressed-tensors) to apply weight-only, data-free post-training quantization (PTQ) and save a checkpoint in the compressed-tensors format. For background on the quantization workflow, see [Quantize using external libraries](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/custom-quantization.html#quantize-using-external-libraries) in the quantization guide.

This is weight-only, data-free PTQ. **Run this step on CPU, not on Trainium**. Use a machine with enough memory to load the ~64 GB BF16 model, plus disk for both the base and quantized checkpoints. You can quantize once and reuse the checkpoint across Trn3 instances.

Only the **text-model linear layers** are quantized. The script builds an ignore list that keeps `lm_head` and every vision-encoder (`visual.*`) linear layer in full precision — the Neuron MXFP8 path applies to the text transformer only. This text-only scheme mirrors the published [Qwen/Qwen3-VL-32B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8) checkpoint, which likewise quantizes the text model and leaves the vision tower and `lm_head` in higher precision.

```python
import torch
from compressed_tensors.offload import dispatch_model
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Path (or Hugging Face ID) of the BF16 base checkpoint.
MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
# Where to write the quantized checkpoint.
SAVE_DIR = "/path/to/Qwen3-VL-32B-Instruct-MXFP8-text-only"

# Load the model and tokenizer.
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Build the ignore list: skip lm_head and every vision-encoder linear layer so
# only the text-model linear layers are quantized to MXFP8.
ignored_list = ["lm_head"]
for name, module in model.named_modules():
    if "visual" in name and isinstance(module, torch.nn.Linear):
        ignored_list.append(name)

print(f"Ignoring {len(ignored_list)} layers from quantization")

# Quantize all remaining Linear layers to MXFP8 via weight-only PTQ.
recipe = QuantizationModifier(targets="Linear", scheme="MXFP8", ignore=ignored_list)
oneshot(model=model, recipe=recipe)

# Confirm the quantized model still generates sane text.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("===========================================")

# Save to disk in compressed-tensors format.
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved quantized model to {SAVE_DIR}")
```

The script prints a short sample generation as a sanity check, then writes the quantized checkpoint and tokenizer to `SAVE_DIR`. Point the serving commands below at that directory.

## Step 4: Run inference

Run the model through the online serving endpoint or the offline `LLM` API — choose whichever fits your deployment.

### Online serving

Start a vLLM OpenAI-compatible server.

:::{note}
On a Trn3 instance without EFA (Elastic Fabric Adapter) installed, the server fails to start during EFA-affinity setup — this affects both BF16 and MXFP8. To work around it, prepend `NEURON_SKIP_EFA_AFFINITY=1` to the `vllm serve` command (and set it in the environment before the offline `LLM` script below).
:::

**BF16** (Trn2 or Trn3):

```bash
vllm serve /path/to/Qwen3-VL-32B-Instruct \
    --served-model-name Qwen3-VL-32B-Instruct \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --tensor-parallel-size 4 \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4],
            "on_device_sampling_config": {"all_greedy": true}
        },
        "vision_neuron_config": {
            "num_vision_tokens_buckets": [2048],
            "vision_attention_block_size": 2048
        }
    }'
```

**MXFP8** (Trn3 only) — point at the quantized checkpoint from Step 3:

```bash
vllm serve /path/to/Qwen3-VL-32B-Instruct-MXFP8-text-only \
    --served-model-name Qwen3-VL-32B-Instruct-MXFP8 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --tensor-parallel-size 4 \
    --block-size 16 \
    --tokenizer-mode slow \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --hf-overrides '{"quantization_config": {}}' \
    --additional-config '{
        "neuron_config": {
            "quantization": "mxfp8",
            "modules_to_not_convert": [],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4],
            "on_device_sampling_config": {"all_greedy": true}
        },
        "vision_neuron_config": {
            "num_vision_tokens_buckets": [2048, 4096],
            "vision_attention_block_size": 2048
        }
    }'
```

The MXFP8-specific settings, compared to the BF16 command:

- **`neuron_config.quantization: "mxfp8"`** — selects the on-device MXFP8 path. This is a Neuron-specific setting; do **not** pass vLLM's `--quantization` flag.
- **`modules_to_not_convert: []`** — an empty list runs the whole text transformer (attention **and** MLP) in MXFP8. The vision encoder stays BF16 regardless, because its weights were left unquantized in Step 3.
- **`--hf-overrides '{"quantization_config": {}}'`** — clears the checkpoint's compressed-tensors `quantization_config` so the Neuron loader owns the MXFP8 path.
- **`--tokenizer-mode slow`**, **`--no-enable-chunked-prefill`**, **`--no-enable-prefix-caching`**, **`--block-size 16`** — pin the deployment to the validated MXFP8 configuration.

**Current limitation:** at `tensor-parallel-size >= 32`, change `"modules_to_not_convert"` to `["mlp"]` (keeps the MLP layers in BF16); full MXFP8 (`[]`) is supported at `tensor-parallel-size <= 16`.

Once the server is up, send requests using the OpenAI Python SDK (use the `--served-model-name` you launched with):

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# Text-only
response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=50,
)
print(response.choices[0].message.content)

# Image + text
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        {"type": "text", "text": "Describe this image."},
    ]}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

### Offline inference

vLLM-Neuron compiles the model on the first run and caches the artifacts to `~/.cache/vllm/neuron/compile_cache`. Subsequent runs skip recompilation and load from cache.

#### Configuration

The model has two components, each with its own config object passed via `additional_config` (see the [configuration options reference](../guides/reference-configuration.md)):

- **`neuron_config`**: text decoder settings (token/sequence bucket sizes, sampling, quantization).
- **`vision_neuron_config`**: vision encoder settings (vision token buckets, attention block size, and optional vision TP/DP split).

**Bucket sizes** control the discrete padded shapes compiled into each NEFF. Each bucket adds compile time; start with one and add more as needed. The two components bucket along different dimensions, so their buckets are configured separately:

- `num_batched_tokens_buckets` (in `neuron_config`): text-decoder buckets over the number of batched **text** tokens per forward pass. See [compilation options](../guides/reference-configuration.md#compilation-options).
- `num_vision_tokens_buckets` (in `vision_neuron_config`): vision-encoder buckets over the number of **vision** patches per encoder forward pass (raw `T*H*W` patches from `image_grid_thw`, before the 2x2 spatial merge; this is the count `select_vision_bucket` matches against). The scheduler may batch images from multiple requests into one forward pass, so size the buckets for the total images processed together, not a single request. These scale with image count and resolution:

| `num_vision_tokens_buckets` | Approximate capacity |
|-----------------------------|----------------------|
| `[2048]` | 1–2 images at 448×448 px |
| `[2048, 8192]` | Up to ~8 images |
| `[2048, 8192, 20480]` | Up to ~20 images |

#### Run offline inference

The following script runs text-only, single-image, and multi-image inference. It shows the **BF16** `LLM(...)` construction; for **MXFP8**, swap in the `LLM(...)` block that follows (the rest of the script is identical).

```python
import os

os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1200"
os.environ["NEURON_LIBTORCH_COMPILATION_TIMEOUT"] = "1200"

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

MODEL_PATH = "/path/to/Qwen3-VL-32B-Instruct"

# BF16 (Trn2 or Trn3)
llm = LLM(
    model=MODEL_PATH,
    max_model_len=8192,
    max_num_batched_tokens=8192,
    max_num_seqs=4,
    tensor_parallel_size=4,
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4],
            "on_device_sampling_config": {"all_greedy": True},
        },
        "vision_neuron_config": {
            "num_vision_tokens_buckets": [2048],
            "vision_attention_block_size": 2048,
        },
    },
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
sampling_params = SamplingParams(max_tokens=200, temperature=0.0)

# --- Text-only ---
outputs = llm.generate(["What is the capital of France?"], sampling_params)
print(outputs[0].outputs[0].text)

# --- Single image ---
image = ImageAsset("cherry_blossom").pil_image.resize((640, 320))
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe this image."},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": [image]}}], sampling_params)
print(outputs[0].outputs[0].text)

# --- Multi-image ---
images = [
    ImageAsset("stop_sign").pil_image.resize((448, 448)),
    ImageAsset("cherry_blossom").pil_image.resize((448, 448)),
]
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "image"},
    {"type": "text", "text": "Compare these two images."},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": images}}], sampling_params)
print(outputs[0].outputs[0].text)
```

For **MXFP8** (Trn3 only), point `MODEL_PATH` at the quantized checkpoint from Step 3 and build the engine with the MXFP8 config instead — the mirror of the MXFP8 `vllm serve` command in Step 4:

```python
MODEL_PATH = "/path/to/Qwen3-VL-32B-Instruct-MXFP8-text-only"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=8192,
    max_num_batched_tokens=8192,
    max_num_seqs=4,
    block_size=16,
    tensor_parallel_size=4,
    tokenizer_mode="slow",
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    hf_overrides={"quantization_config": {}},
    additional_config={
        "neuron_config": {
            "quantization": "mxfp8",
            "modules_to_not_convert": [],
            "num_batched_tokens_buckets": [8192],
            "num_seqs_buckets": [4],
            "on_device_sampling_config": {"all_greedy": True},
        },
        "vision_neuron_config": {
            "num_vision_tokens_buckets": [2048, 4096],
            "vision_attention_block_size": 2048,
        },
    },
)
```

## Vision parallelism (optional)

By default the vision encoder runs as one DP replica per NeuronCore (TP1), the recommended layout for high-throughput multi-image workloads. To shard the encoder weights across cores for a single large image instead, increase `tp_size`.

Set `tp_size` inside `vision_neuron_config` to change the split. DP is derived automatically as `world_size / tp_size`, where `world_size` is your `tensor_parallel_size`. The commands above use `tensor_parallel_size=4`, so:

| `tp_size` | Vision TP | Vision DP | Best for |
|-----------|-----------|-----------|----------|
| `1` (default) | 1 | 4 | Multi-image, high throughput |
| `4` | 4 | 1 | Single-image, low latency |

The same applies at a larger `tensor_parallel_size` (e.g. `16`), which raises the default vision DP accordingly. Example — TP1, full DP:

```python
"vision_neuron_config": {
    "num_vision_tokens_buckets": [2048],
    "vision_attention_block_size": 2048,
    "tp_size": 1,  # DP = world_size / tp_size
}
```

## Conclusion

You have deployed Qwen3-VL-32B-Instruct on Trainium — in BF16 on `trn2.48xlarge` (or Trn3), or in MXFP8 on Trn3 after self-quantizing the text model. The model supports text-only, single-image, multi-image, and video inputs via both the offline `LLM` API and the OpenAI-compatible online serving endpoint. To validate accuracy after enabling MXFP8, see the [accuracy debugging guide](../model-dev/accuracy-debugging-guide.md). For feature support and accuracy validation results, see the [model card](../model-recipes/qwen3-vl.md).

## Next steps

To tune this deployment for your workload — choosing vision and text sharding with roofline analysis, profiling to find the bottleneck, and enabling the multimodal features and fused kernels — see [Optimizing a Vision-Language Model](../model-dev/optimizing-vlm-models.md).
