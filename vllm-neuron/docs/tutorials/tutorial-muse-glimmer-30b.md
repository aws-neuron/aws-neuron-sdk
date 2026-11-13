# Tutorial: Deploy Muse-Glimmer-30B with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying the multimodal
Muse-Glimmer-30B model (text, image, and video) with vLLM Neuron on Trn3 in
BF16, covering environment setup, installing the contrib model, online
serving with the reasoning parser, offline inference, and POPE accuracy. -->
<!-- meta: keywords: vLLM, Neuron, Muse-Glimmer, Muse-Glimmer-30B, multimodal,
VLM, vision-language, video, contrib model, reasoning parser, POPE, tutorial,
LLM serving, Trn3, Trainium -->
<!-- meta: date_updated: 2026-11-12 -->
<!-- Content type: procedural-tutorial -->

This tutorial walks through deploying
[Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) with
vLLM Neuron on Trn3 in BF16. It covers environment setup, installing the model,
online serving, offline inference, and an accuracy check.

Muse-Glimmer-30B accepts text, images, and video. The text decoder has 52 layers
in a repeating pattern of three sliding-window layers (window 2048, RoPE) and
one global layer (no positional encoding), with gated attention and 2 KV heads.
The vision encoder is a windowed-attention ViT that merges 2x2 patches into one
embedding. The model writes its reasoning in a separate `to=self` channel before
the final answer.

Muse-Glimmer-30B ships as a **contrib model** in the vLLM Neuron repository
(`contrib/models/muse_glimmer`). Contrib models are community contributions and
may have limited testing or performance optimization compared to the models
that vLLM Neuron supports officially. The model is plain PyTorch over the paged
KV cache, with no NKI kernels.

## Tested versions

| Component | Version |
|-----------|---------|
| Instance | Trn3 (`trn3-dev1.48xlarge`, 2 Neuron devices at logical NeuronCore config 2) |
| Neuron SDK | 2.33 |
| vLLM | 0.29.0 |
| vLLM Neuron plugin | 0.29.0.1.2.0 |
| Container | `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` |
| Python | 3.14 |
| Tensor parallelism | TP=2 |

Trn2 has not been tested.

**Prerequisites:**

- A Trn3 instance with Neuron SDK 2.33. See the
  [setup guide](../getting-started/setup-guide.md). The tested environment is
  the vLLM Neuron Deep Learning Container listed above, used as shipped (see
  [Option C: Use a container](../getting-started/setup-guide.md#option-c-use-a-container)).
- A Hugging Face token with access to the gated
  [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
  checkpoint.

## Step 1: Set up your environment

Verify Neuron devices are visible:

```bash
neuron-ls
# Lists the Neuron devices and cores available on your instance.
```

Set environment variables before running any inference script:

```bash
# The validated configuration runs at logical NeuronCore config 2. vLLM Neuron
# uses 1 when the variable is unset, which leaves too little HBM per core for
# the 30B weights.
export NEURON_LOGICAL_NC_CONFIG=2

# Select the native compile route explicitly. vLLM Neuron uses it when the
# variable is unset, but libtorch-neuronx-lite reads the variable itself: unset,
# it replaces torch.nn.functional.gelu with a wrapper that torch.compile cannot
# trace, and the vision encoder fails to compile.
export VLLM_NEURON_BACKEND=neuron_native

# The checkpoint is gated.
export HF_TOKEN=<your Hugging Face token>
```

## Step 2: Install the model

Clone vLLM Neuron and install the Muse-Glimmer contrib package:

```bash
git clone https://github.com/vllm-project/vllm-neuron.git
cd vllm-neuron
pip install -e contrib/models/muse_glimmer
```

The install adds a `vllm.general_plugins` entry point that registers
Muse-Glimmer in every vLLM process, including the tensor-parallel workers. After
the install, `vllm serve` and `LLM(...)` find the model with no other setup.
Nothing else in vLLM Neuron changes.

## Step 3: Download the model

Download the weights once. The server, the offline script, and the
integration test below all read this directory:

```bash
hf download \
    meta-models/Muse-Glimmer-30B \
    --local-dir /path/to/Muse-Glimmer-30B
```

## Step 4: Run inference

Run the model through the online serving endpoint or the offline `LLM` API.

### Online serving

Start a vLLM OpenAI-compatible server:

```bash
vllm serve /path/to/Muse-Glimmer-30B \
    --served-model-name Muse-Glimmer-30B \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 1024 \
    --num-gpu-blocks-override 512 \
    --limit-mm-per-prompt '{"image": 1, "video": 1}' \
    --media-io-kwargs '{"video": {"num_frames": -1}}' \
    --no-enable-prefix-caching \
    --reasoning-parser muse_glimmer \
    --default-chat-template-kwargs '{"reasoning_strength": "low"}' \
    --additional-config '{
        "neuron_config": {
            "num_batched_tokens_buckets": [1024],
            "num_seqs_buckets": [1]
        },
        "encoder_neuron_config": {
            "encoder_attention_block_size": 1024,
            "num_encoder_input_buckets": [1024, 1564, 1632, 2080],
            "encoder_cache_num_blocks": 16
        }
    }'
```

The first start compiles the text and vision graphs, which takes about 15
minutes. The Muse-Glimmer-specific settings:

- **`num_encoder_input_buckets`**: one vision graph per fixed shape, counted in
  patches. Each input is resized to the shape with the closest aspect ratio:

  | Input | Shape | Patches | Vision tokens |
  |-------|-------|---------|---------------|
  | Square image | 448x448 | 1024 | 256 |
  | Landscape image (for example 640x480) | 644x476 | 1564 | 391 |
  | Portrait image (for example 480x640) | 476x672 | 1632 | 408 |
  | Video | 8 frames at 480x368, real timestamps | 2080 | 520 |

- **`encoder_cache_num_blocks: 16`**: encoder-cache blocks hold 256 tokens
  (`encoder_attention_block_size` / 4), and each item pads its last block. A
  520-token video takes 3 blocks and a 391-token image takes 2. The default
  size counts tokens only, so a video and an image together can fill it; 16
  blocks leave room.
- **`--no-enable-prefix-caching`**: in vLLM Neuron 0.29.0.1.2.0, a prefix-cache
  hit can leave only an image's end marker to prefill. The runner then requires
  the image in the encoder cache and stops the engine.
- **`--media-io-kwargs '{"video": {"num_frames": -1}}'`**: vLLM passes all
  decoded frames and their timestamps. The model's processor picks 8 frames and
  writes the real timestamps into the prompt.
- **`--reasoning-parser muse_glimmer`**: puts the `to=self` reasoning in
  `reasoning_content`, so `content` holds only the final answer.
- **`--default-chat-template-kwargs '{"reasoning_strength": "low"}'`**: the chat
  template defaults to `high`. A request can set its own strength with a system
  message such as `Reasoning strength: high`.
- **`--num-gpu-blocks-override 512`**: vLLM creates 4 KV-cache groups
  (sliding-window and full-attention layers), so one 2048-token sequence needs
  4 x 64 blocks of 32 tokens.

Once the server is up, send requests using the OpenAI Python SDK. Images and
videos go in as `image_url` and `video_url` content parts, as URLs or base64
data URLs:

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# Text-only
response = client.chat.completions.create(
    model="Muse-Glimmer-30B",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=128,
)
print(response.choices[0].message.content)

# Image + text
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Muse-Glimmer-30B",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        {"type": "text", "text": "Describe this image."},
    ]}],
    max_tokens=256,
)
print(response.choices[0].message.content)

# Video + text
with open("video.mp4", "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Muse-Glimmer-30B",
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
        {"type": "text", "text": "What happens in this video?"},
    ]}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

The model's reasoning for each answer is in
`response.choices[0].message.reasoning_content`.

### Offline inference

vLLM Neuron compiles the model on the first run and caches the artifacts to
`~/.cache/vllm/neuron/compile_cache`. Later runs load from the cache.

The model has two components, each with its own config object passed through
`additional_config` (see the
[configuration options reference](../guides/reference-configuration.md)):

- **`neuron_config`**: text decoder settings (token and sequence bucket sizes).
- **`encoder_neuron_config`**: vision encoder settings (vision patch buckets,
  attention block size, and encoder-cache size).

The following script runs a text-only request and an image request, with the
same configuration as the server. It reads `image.jpg` from the current directory:

```python
import base64
import os

os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["VLLM_NEURON_BACKEND"] = "neuron_native"

from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/Muse-Glimmer-30B",
    tensor_parallel_size=2,
    max_model_len=2048,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    num_gpu_blocks_override=512,
    limit_mm_per_prompt={"image": 1, "video": 1},
    media_io_kwargs={"video": {"num_frames": -1}},
    enable_prefix_caching=False,
    additional_config={
        "neuron_config": {
            "num_batched_tokens_buckets": [1024],
            "num_seqs_buckets": [1],
        },
        "encoder_neuron_config": {
            "encoder_attention_block_size": 1024,
            "num_encoder_input_buckets": [1024, 1564, 1632, 2080],
            "encoder_cache_num_blocks": 16,
        },
    },
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=256)


def final_answer(output):
    # Offline output holds both channels; the final answer follows "to=user".
    return output.outputs[0].text.rsplit("to=user", 1)[-1].strip()


# --- Text-only ---
outputs = llm.chat(
    [[{"role": "user", "content": "What is the capital of France?"}]],
    sampling_params,
    chat_template_kwargs={"reasoning_strength": "low"},
)
print(final_answer(outputs[0]))

# --- Image + text ---
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

outputs = llm.chat(
    [[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        {"type": "text", "text": "Describe this image."},
    ]}]],
    sampling_params,
    chat_template_kwargs={"reasoning_strength": "low"},
)
print(final_answer(outputs[0]))
```

## Supported features

| Feature | Status |
|---------|--------|
| Text; single image + text; video + text; image + video + text in one prompt | Supported |
| BF16 | Supported |
| Tensor parallelism | TP=2 validated. Above 2, the 2 KV heads are replicated; not validated |
| On-device greedy sampling | Supported |
| Multi-image + text | Not supported |
| MXFP8 | Not supported |
| Encoder parallelism, EPD, segmented prefill, speculative decoding | Not supported |
| Prefix caching | Not supported on vLLM Neuron 0.29.0.1.2.0 (see [Online serving](#online-serving)) |
| Model Runner V2 | Not supported (runs on Model Runner V1) |

## Accuracy

**POPE** (Polling-based Object Probing Evaluation) asks Yes/No questions about
object presence in COCO images, in three subsets (popular, adversarial, and
random). The run uses the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
`fd3884b` request and Yes/No rule, `--max-tokens 256`, reasoning strength
`low`, Trn3, BF16, and TP=2.

On vLLM Neuron 0.29.0.1.2.0, the model was evaluated on a fixed 800-question
subset: the 400 "yes" questions, which are the same in all three subsets, and
400 adversarial "no" questions (seed 0, built with
[`scripts/pope_subset.py`](https://github.com/yahavb/Muse-Glimmer-30B/blob/main/scripts/pope_subset.py)):

| Metric | Muse-Glimmer-30B |
|--------|------------------|
| F1 | 85.99 |
| Precision / Recall | 94.33 / 79.00 |
| Accuracy | 87.12 |

All 800 answers were parsed as Yes or No.

## Test the deployment

The contrib package includes unit tests and an integration test. Run them from
the vLLM Neuron clone from Step 2. Install the test tools first:

```bash
pip install pytest pytest-forked pytest-timeout
```

The integration test serves the checkpoint from Step 3 on Trn3 and checks text,
image, and video answers. Stop the server from Step 4 first, so the
NeuronCores are free:

```bash
MUSE_GLIMMER_MODEL_PATH=/path/to/Muse-Glimmer-30B NEURON_LOGICAL_NC_CONFIG=2 \
    pytest contrib/models/muse_glimmer/test/integration/test_model.py -v -s --timeout=3600
```

The unit tests run in CPU mode, with no Neuron hardware. They compare a tiny
random-weight model with Hugging Face transformers at TP=1 and TP=2:

```bash
VLLM_NEURON_CPU_MODE=1 pytest contrib/models/muse_glimmer/test/unit/ -v --forked
```

## Conclusion

You have deployed Muse-Glimmer-30B on Trn3 in BF16 with vLLM Neuron. The
OpenAI-compatible server answers text, image, and video prompts, with the
reasoning separated from the final answer by the `muse_glimmer` reasoning
parser. The offline `LLM` API answers text and image prompts.

## Next steps

- For the model architecture and source layout, see the
  [Muse-Glimmer contrib model](https://github.com/vllm-project/vllm-neuron/tree/main/contrib/models/muse_glimmer)
  in the vLLM Neuron repository.
- To tune a vision-language deployment, see
  [Optimizing a Vision-Language Model](../model-dev/optimizing-vlm-models.md).
- For another multimodal deployment, see
  [Deploy Qwen3-VL-32B](tutorial-qwen3-vl-32b.md).
