# Tutorial: Configure disaggregated encoder inference with 1E1PD and xEyPD

<!-- meta: description: Configure encoder-disaggregated (EPD) multimodal inference
with vLLM Neuron using a simple 1E1PD example and scaling to xEyPD on AWS
Trainium. -->
<!-- meta: keywords: disaggregated encoder, EPD, 1E1PD, xEyPD, vision encoder,
prefill, decode, encoder cache, NIXL, EC connector, Qwen3-VL, vLLM, Neuron,
Trainium -->
<!-- meta: date_updated: 2026-08-13 -->
<!-- Content type: procedural-tutorial -->

This topic guides you through configuring encoder-disaggregated inference (EPD)
with vLLM Neuron using a simple 1E1PD (one vision encoder, one
prefill+decode engine) example and scaling up to a general xEyPD topology. When
you have completed it, you will have a working EPD deployment for a multimodal
model and will understand the control and data flow between the vision encoder
pool, the prefill+decode pool, and the router.

## Overview

For a vision-language model, each request first runs the prompt's images (or
video) through a **vision encoder** to produce embeddings, then runs the
**language model** (prefill + decode) over the text tokens with those
embeddings spliced in. These two stages bottleneck on different resources, and
their cost scales with different properties of the request. The encoder is
compute-bound, and its cost tracks image count and resolution — the prompt's
text length and the number of tokens generated do not affect it. The PD stage
combines a compute-bound prefill over the prompt with memory-bandwidth-bound
token generation, which loads the model weights from HBM once per token.

Encoder-disaggregated inference (EPD) separates the vision encoder from the
language model across different servers. This separation lets you:

- Scale the vision encoder pool and the prefill+decode pool independently based
  on how image-heavy your traffic is.
- Use different parallelism for each stage. This tutorial gives the encoder a
  few cores for throughput (TP=4, running DP=4 within them) and the language
  model a wider tensor-parallel group (TP=8). For how the encoder's `tp_size`
  trades multi-image throughput against single-image latency, see
  [Vision parallelism](tutorial-qwen3-vl-32b.md#vision-parallelism-optional).
- Keep expensive vision encoding off the critical path of the language model,
  and reuse encoded embeddings across the pool via a content-addressed cache.

A single multimodal server remains simpler and is the better default: EPD adds a
router, an RDMA transport, and per-pool core-placement constraints. Reach for it
when you have measured that one stage is the bottleneck.

:::{note}
EPD is a **two-way** disaggregation: it splits the vision **E**ncoder from the
combined **P**refill+**D**ecode engine. It is distinct from
[disaggregated inference (DI)](tutorial-di-1p1d-xpyd.md), which splits prefill
from decode. vLLM Neuron does **not** currently support three-way (E + P + D)
disaggregation — prefill and decode remain co-located in a single PD engine.
:::

The architecture has three components:

1. **Vision Encoder (VE) pool** — runs the vision tower only (launched with
   `--mm-encoder-only`), encodes each image/video into embeddings, and writes
   them into an on-device encoder cache. VE engines act as the EC **producer**.
2. **Prefill+Decode (PD) pool** — runs the language model only (launched with
   `"mm_language_model_only": true` inside `--additional-config`), pulls the
   vision embeddings it needs, runs prefill over the prompt, and generates
   tokens. PD engines act as the EC **consumer**.

3. **Router** — preprocesses each request once, routes each media item to a VE,
   collects the embedding locators, then drives a PD engine to generate the
   response. It exposes an OpenAI-compatible `/v1/chat/completions` endpoint.
   This repository ships a minimal reference router; the VE/PD contract it
   implements is what any replacement has to reproduce (see
   [Router arguments](#router-arguments)).

:::{note}
The two build-role switches are set differently. `--mm-encoder-only` is an
upstream vLLM CLI flag, so you pass it directly. `mm_language_model_only` is a
vllm-neuron plugin field with no upstream CLI equivalent — upstream's
disaggregated-encoder consumer loads the full model including the vision tower —
so it must be set inside the `--additional-config` JSON object. There is no
`--mm-language-model-only` flag.
:::

Embedding transfer between VE and PD uses the `NeuronNixlECConnector` over
[NIXL](https://github.com/ai-dynamo/nixl) / LIBFABRIC (EFA on AWS). The transfer
is an HBM→HBM RDMA **READ**: the PD engine pulls each vision embedding directly
from the producing VE's device-side encoder cache.

**Request flow:**

1. Client sends an OpenAI chat-completions request (text + one or more images)
   to the router.
2. The router preprocesses the request once, computing a content hash
   (`mm_hash`) and placeholder token positions for each media item.
3. The router routes each media item to a VE using rendezvous
   (highest-random-weight) hashing on its `mm_hash`, so the same image
   consistently lands on the same VE and can hit the warm encoder cache.
4. Each chosen VE encodes its items, writes the embeddings into its on-device
   encoder cache, and returns a per-`mm_hash` **EC locator** (where the
   embedding lives).
5. The router picks a PD engine (round-robin) and calls its
   `/inference/v1/generate` with the prompt token ids, the placeholder
   positions, and the EC locators.
6. The PD engine pulls each embedding from the owning VE via a NIXL RDMA READ
   over LIBFABRIC, splices them into the token embedding sequence, runs prefill,
   and generates the output tokens.
7. The router detokenizes the PD's token stream and returns an OpenAI
   ChatCompletion response (streamed when `stream=True`).

For more detail on the on-device encoder cache that VE writes and PD reads, see
the
[On-device encoder cache design document](../design/multimodal/on_device_encoder_cache.md).

## Before you start

This tutorial assumes that you have experience in the following areas:

- Serving a multimodal model on vLLM Neuron on a single instance. See
  [Deploy Qwen3-VL-32B](tutorial-qwen3-vl-32b.md).
- Running vLLM Neuron online serving. See
  [online serving quickstart](../getting-started/quickstart-online-serving.md).
- Familiarity with vision encoders and how embeddings are spliced into the
  language-model prompt.
- Understanding of tensor parallelism and data parallelism concepts.

## Prerequisites

- **Neuron instance**: A supported Trainium instance with enough NeuronCores for
  both pools. A `trn2.48xlarge` (64 NeuronCores) comfortably fits the 1E1PD
  example (12 cores) and scales up to the 2E7PD reference topology (64 cores).
- **vLLM Neuron environment**: Installed and verified. See
  [setup guide](../getting-started/setup-guide.md).
- **NIXL and its runtime libraries**: The NIXL EC transfer library. Install with:

  ```bash
  pip install nixl
  ```

  NIXL's LIBFABRIC backend also loads the `libcuda.so.1` and `libfabric.so.1`
  shared libraries at runtime. If your environment does not already ship them
  (some container images do not), follow
  [Install dependencies for disaggregated inference](../getting-started/setup-guide.md#install-dependencies-for-disaggregated-inference)
  before you start. Otherwise the servers fail at startup with
  `unsupported backend 'LIBFABRIC'`.

- **Model access**: A multimodal model that vLLM Neuron supports. This tutorial
  uses [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct).

## Prepare your environment

:::{important}
**Run this whole section in every terminal you launch a component from.** The
three components each run in the foreground, so they each need their own shell —
and `source` / `export` do not cross shells. Skipping the exports below in the VE
or PD terminal is the single most common way to break an otherwise correct
deployment: startup and compilation succeed, and the first multimodal request
then fails with `ECLoadFailure` (see [Common issues](#common-issues)).
:::

Activate the vLLM virtual environment:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_*/bin/activate
```

:::{note}
This is the pre-built environment shipped by the Neuron DLAMI. The glob avoids
pinning a vLLM version, since the directory name embeds one
(`..._vllm_0_24_0_1_0_0`) that changes between DLAMI releases. It assumes exactly
one such environment — check with `ls -d /opt/aws_neuronx_venv*` and activate the
one you want by full path if there are several. If you installed via pip or use
the container image, the path differs; see the
[setup guide](../getting-started/setup-guide.md) for the other install options.
:::

The EC connector's LIBFABRIC/EFA path expects device RDMA and no shared-memory
transport. Export these in **every** VE and PD terminal — these two are required,
not tuning knobs:

```bash
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_EFA_USE_DEVICE_RDMA=1
```

Without them, LIBFABRIC may select the shared-memory transport instead of device
RDMA. Nothing fails at startup, so both engines come up healthy; the first
multimodal request is where it breaks, because the READ of the VE's on-device
encoder cache never completes and PD raises `ECLoadFailure`. Verify with
`env | grep FI_EFA` in each engine's terminal before launching it.

First-time Neuron compilation of a 32B engine can take several minutes. If your
servers time out during compilation, raise the compile timeout:

```bash
export VLLM_NEURON_COMPILATION_TIMEOUT=6000
```

Set your model path — either a Hugging Face model id or a local checkpoint
directory:

```bash
MODEL="Qwen/Qwen3-VL-32B-Instruct"

# Fail fast if MODEL is unset or empty, rather than silently serving the wrong model.
MODEL="${MODEL:?set MODEL to a Hugging Face model id or checkpoint path first}"
```

:::{tip}
**Prefer a local checkpoint directory over a hub id.** Every engine resolves the
model id against the Hugging Face API before its engine is built, so bringing up
a pool multiplies those calls — and unauthenticated requests are rate-limited per
IP, which is shared on NAT'd or multi-tenant hosts. When the quota is exhausted,
each affected engine dies at startup with
`HfHubHTTPError: 429 Too Many Requests: you have reached your 'api' rate limit`
raised from `create_model_config`, well before any Neuron work happens. Point
`MODEL` at an already-downloaded checkpoint and pin the hub offline:

```bash
MODEL=/path/to/Qwen3-VL-32B-Instruct
export HF_HUB_OFFLINE=1
```

A local path alone skips the API lookup; `HF_HUB_OFFLINE=1` additionally
guarantees no other component reaches for the hub mid-launch. If you do use a hub
id, export an `HF_TOKEN` — authenticated requests get a far higher quota — and
stagger engine startups as described below.
:::

:::{warning}
Every command below passes `"$MODEL"` quoted, and the guard above aborts on an
empty value. Both matter: `vllm serve`'s model argument is *positional and
optional*, so if `MODEL` is empty and unquoted the shell drops the argument
entirely and vLLM silently falls back to its built-in default
(`Qwen/Qwen3-0.6B`) — a **text-only** model. Because that model has no vision
tower, the VE then fails deep in model loading with a confusing
`TypeError: Qwen3ForCausalLM.from_configs() got an unexpected keyword argument
'text_neuron_config'` rather than a clear "no model specified" error.
:::

This example places all three components on one instance, so we pin each pool to
a disjoint slice of NeuronCores with `NEURON_VISIBLE_DEVICES` and give each
server a distinct port. The layout is:

| Component | Role | Cores | HTTP port | NIXL side-channel |
|-----------|------|-------|-----------|-------------------|
| PD-0      | prefill+decode (TP=8) | `0-7`  | `18100` | — (consumer) |
| VE-0      | encoder (DP=4) | `8-11` | `18300` | `14600` |
| Router    | OpenAI front end | — | `18800` | — |

The larger pool comes first here. On `trn2`, each engine's core slice must start
on a boundary that is a multiple of its own TP degree, so the TP=8 PD engine must
start at core 0, 8, 16, … while the TP=4 VE only needs a multiple of 4. Placing
PD at `0-7` and VE at `8-11` satisfies both and packs the two pools into 12
contiguous cores with nothing stranded; leading with the TP=4 VE would push PD to
core 8 and leave cores `4-7` idle. A misaligned slice loads and compiles fine and
only fails later, during prefill warmup, when the TP group's `reduce_scatter`
cannot be mapped onto the hardware collective — see
[Common issues](#common-issues).

Each of the three commands below runs in the foreground, so run each one in its
own terminal (or under `tmux`) — and re-run "Prepare your environment" in each of
those terminals first, so the venv, the EFA variables, and `MODEL` are all set.

:::{important}
**Prefix caching must be off on every engine in both pools.** Each command below
passes `--no-enable-prefix-caching`; do not drop it. Prefix caching is enabled by
default upstream, so this is an opt-out, and the failure it causes is silent —
wrong or empty output rather than an error. Step 2 explains why.
:::

:::{tip}
**Start the VE and PD in parallel, but the router last.** Because the two pools
own disjoint core slices and do not talk to each other at startup, they can
compile concurrently — which roughly halves cold-start wall clock, and cold
compile of a 32B engine can exceed 10 minutes. Two rules of thumb, both taken
from the reference launcher (`run_epd_single_node` in `epd_common.py`):

- Stagger the launches by ~10 seconds rather than starting them at the same
  instant. The launcher sleeps `startup_delay_s = 10.0` between engines to avoid
  a thundering-herd race on `NEURON_VISIBLE_DEVICES` binding and on the shared
  compile cache.
- Start the router only after every engine reports healthy
  (`curl http://127.0.0.1:<port>/health`). The launcher health-checks all
  engines concurrently and starts the router afterward. An early router will not
  crash, but requests fail until the pools are ready.
:::

## Step 1: Launch the vision encoder (VE)

In this step, you will start a vision-encoder-only server in the `ec_producer`
role. It runs the vision tower, writes embeddings into its on-device encoder
cache, and serves their locators to PD engines over NIXL.

```bash
NEURON_VISIBLE_DEVICES=8,9,10,11 vllm serve "$MODEL" \
    --tensor-parallel-size 4 \
    --host 127.0.0.1 \
    --port 18300 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 1 \
    --dtype bfloat16 \
    --limit-mm-per-prompt '{"image": 30}' \
    --no-enable-prefix-caching \
    --mm-encoder-only \
    --no-async-scheduling \
    --additional-config '{"neuron_config": {"quantization": "bf16", "num_batched_tokens_buckets": [8192], "on_device_sampling_config": {"all_greedy": true}, "num_seqs_buckets": [1]}, "vision_neuron_config": {"num_vision_tokens_buckets": [30720], "vision_attention_block_size": 1024, "dp_size": 4, "encoder_cache_num_blocks": 128}}' \
    --ec-transfer-config '{"ec_connector": "NeuronNixlECConnector", "ec_role": "ec_producer", "engine_id": "epd-ve-0", "ec_connector_extra_config": {"backends": ["LIBFABRIC"], "side_channel_host": "127.0.0.1", "side_channel_port": 14600}}'
```

Key parameters:

- `NEURON_VISIBLE_DEVICES=8,9,10,11` — restricts this VE to NeuronCores 8–11
  (TP=4). The worker asserts that TP equals the number of visible devices.
- `--mm-encoder-only` — build and run **only** the vision tower. The language
  model is dropped, so no prefill/decode warmup runs on this engine. This one is
  an upstream vLLM CLI flag, so it is passed directly (unlike PD's
  `mm_language_model_only`, which goes inside `--additional-config`).
- `vision_neuron_config.dp_size: 4` — the vision encoder runs data-parallel
  across its 4 cores for encoder throughput. DP replicas favor multi-image
  throughput; shard the encoder weights instead (raise `tp_size`, which derives
  DP as `world_size / tp_size`) when single-image latency matters more — see
  [Vision parallelism](tutorial-qwen3-vl-32b.md#vision-parallelism-optional).
- `--ec-transfer-config` with `ec_role: "ec_producer"` — this engine produces
  embeddings. `engine_id` **must be unique per VE**; `side_channel_host` /
  `side_channel_port` is where PD engines fetch NIXL transfer metadata.
- `num_vision_tokens_buckets: [30720]` — the vision-token bucket, sized to the
  example workload as
  `num_images × (resolution / patch_size)²` = `30 × (512 / 16)²` = `30 × 1024` =
  `30720`. Recompute it for your own images: `resolution` is the (square) input
  edge length in pixels after preprocessing, `patch_size` is the vision tower's
  patch edge (16 for Qwen3-VL), and `num_images` should match
  `--limit-mm-per-prompt`. Non-square images use
  `(width / patch_size) × (height / patch_size)` per image.

Wait for the server to print `Uvicorn running on http://127.0.0.1:18300`.

## Step 2: Launch the prefill+decode engine (PD)

In this step, you will start a language-model-only server in the `ec_consumer`
role. It pulls vision embeddings from the VE, runs prefill, and generates tokens.

```bash
NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve "$MODEL" \
    --tensor-parallel-size 8 \
    --host 127.0.0.1 \
    --port 18100 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 8 \
    --dtype bfloat16 \
    --limit-mm-per-prompt '{"image": 30}' \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --additional-config '{"neuron_config": {"quantization": "bf16", "num_batched_tokens_buckets": [8192], "on_device_sampling_config": {"all_greedy": true}}, "vision_neuron_config": {"num_vision_tokens_buckets": [30720], "vision_attention_block_size": 1024, "dp_size": 1, "encoder_cache_num_blocks": 128}, "mm_language_model_only": true}' \
    --ec-transfer-config '{"ec_connector": "NeuronNixlECConnector", "ec_role": "ec_consumer", "engine_id": "epd-pd-0", "ec_connector_extra_config": {"backends": ["LIBFABRIC"]}}'
```

Key parameters:

- `NEURON_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` — uses NeuronCores 0–7 (TP=8),
  disjoint from the VE's cores. On `trn2`, a TP=8 engine must start on a multiple
  of 8, so it takes the bottom of the range and the VE follows at core 8 (see the
  layout table above).
- `"mm_language_model_only": true` — build and run **only** the language model.
  The vision tower is dropped, so no encoder warmup runs on this engine. Note
  this goes **inside `--additional-config`**, not as a CLI flag (see the note in
  the Overview). `mm_encoder_only` and `mm_language_model_only` are mutually
  exclusive — setting both raises at startup.
- `ec_role: "ec_consumer"` — this engine consumes embeddings produced by the VE.
  A consumer does not need a side-channel port; it uses the locator handed to it
  by the router to reach the producing VE.
- `--mm-processor-cache-gb 0` — the multimodal processor cache is unused on PD
  because images never reach it (the VE encodes them).

:::{note}
Prefix caching is disabled (`--no-enable-prefix-caching`) on both pools. A KV
prefix hit on tokens whose vision embedding has not been resolved would
short-circuit encoder-input scheduling, and the EC connector would never fire —
PD would run the language model on stale blocks.
:::

Wait for the server to print `Uvicorn running on http://127.0.0.1:18100`.

## Step 3: Launch the router

In this step, you will start the router that fronts the pools with an
OpenAI-compatible API and orchestrates the VE → PD flow.

```bash
python3 examples/vllm_neuron/vllm/disaggregated_encoder/server.py \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port 18800 \
    --ve-timeout-s 600 \
    --pd-timeout-s 600 \
    --ve-endpoint 127.0.0.1:18300 \
    --pd-endpoint 127.0.0.1:18100
```

The router listens on port 18800 and coordinates the request lifecycle: it
preprocesses the request, routes each image to a VE with `--ve-endpoint`, and
drives a PD engine from `--pd-endpoint` to generate the response.

### Router arguments

That is the router's complete argument surface:

| Argument | Default | Description |
| ---- | ---- | ---- |
| `--model` | required | HF model id or local path. Must be the same checkpoint the VE and PD engines were launched with — the router tokenizes and detokenizes with it. |
| `--ve-endpoint` | required | `host:port` of a Vision Encoder engine. Repeat once per VE; media items are spread over the pool by rendezvous hashing. |
| `--pd-endpoint` | required | `host:port` of a Prefill+Decode engine. Repeat once per PD; requests are assigned round-robin. |
| `--host` | `127.0.0.1` | Bind host for the router's own HTTP server. |
| `--port` | `8000` | Bind port for the router's own HTTP server. |
| `--ve-timeout-s` | `120.0` | Per-VE encode request timeout, in seconds. Raise it if a cold VE is still compiling vision buckets. |
| `--pd-timeout-s` | `600.0` | Per-PD generate request timeout, in seconds. Must cover the full generation, not just time-to-first-token. |
| `--quantization` | none | Neuron quantization of the checkpoint, for example `mxfp8`. Required for a quantized checkpoint. |

:::{important}
Pass `--quantization` whenever the checkpoint is quantized, even though the
router loads no weights. The router builds a renderer, which constructs a
`VllmConfig`, and the Neuron platform rejects the checkpoint's HF
`quantization_config` unless the CPU-dequant quantization is declared. An MXFP8
deployment therefore needs `--quantization mxfp8` on the router in addition to
`"quantization": "mxfp8"` in each engine's `neuron_config`. The value must match
what the engines use. Omitting it fails at router startup, before any request is
served. This tutorial's checkpoint is bf16, so the command above does not
set it.
:::

:::{note}
The router (`server.py`) and the reusable single-node launcher (`epd_common.py`)
are included in the vLLM Neuron repository under
`examples/vllm_neuron/vllm/disaggregated_encoder/` as **reference examples**.
They are deliberately minimal: both pools are a static list of endpoints, PD
selection is plain round-robin, and there is no retry, failover, backpressure, or
autoscaling. For production deployments, consider an orchestrator for routing,
health checking, and autoscaling.

If you bring your own router, it still has to reproduce the same contract with
the two pools. The arguments above are not only CLI conveniences:

- **Load the same checkpoint, and declare its quantization.** The router
  tokenizes and detokenizes, so it needs the tokenizer and processor for the
  exact checkpoint the engines run, and it must declare a quantized checkpoint's
  quantization even though it loads no weights (see the admonition above).
- **Preprocess once, on the router.** Compute each media item's content hash
  (`mm_hash`) and its placeholder `offset`/`length` in the prompt, then send PD
  the prompt token ids plus those placeholders — not the raw chat messages.
  Preprocessing again on PD would recompute work and can desynchronize the
  hashes the locators are keyed on.
- **Send only the LM-relevant mm kwargs.** Strip encoder-only tensors such as
  `pixel_values` — PD pulls those over NIXL, not over HTTP. For Qwen3-VL, only
  `image_grid_thw` needs to travel with the request, for M-RoPE; see
  `get_epd_kwargs` in `vllm_neuron/model/qwen3_vl/factory.py`.
- **Forward the EC locators.** Each VE returns one locator per `mm_hash`, and
  the PD request must carry them under
  `sampling_params.extra_args.kv_transfer_params.ec_locator`. Without them PD
  cannot find the embeddings, and the request fails with `ECLoadFailure`.
- **Encode before you generate.** Wait for every VE to acknowledge a request's
  items before calling PD's `/inference/v1/generate` (note PD is driven through
  that endpoint, not `/v1/chat/completions`). PD's RDMA read assumes the
  embedding already sits in the VE's on-device cache.
- **Route media items by content, not round-robin.** Hashing each item to a VE
  by its `mm_hash` — the example uses rendezvous hashing — is what lets a
  repeated image land on the VE that already has it encoded. Spraying items
  across the pool instead costs you every encoder-cache hit. PD needs no such
  affinity, since any PD can pull any embedding.
:::

## Step 4: Validate the 1E1PD deployment

In this step, you will send a multimodal request through the router and confirm
it completes end-to-end. The router speaks the OpenAI chat-completions protocol,
so any OpenAI client works against it.

The one thing to get right is image size. Both pools above set
`vision_attention_block_size: 1024`, which bounds a single image to 1024 patches
— at Qwen3-VL's patch size of 16, a 512×512 budget. An oversized image is
rejected in the encoder, so the script below downscales before sending and
inlines the result as base64 rather than depending on the server being able to
reach an external URL:

```python
import base64
import io
import urllib.request

from openai import OpenAI
from PIL import Image

MODEL = "Qwen/Qwen3-VL-32B-Instruct"   # must match what the servers loaded
IMAGE_URL = (
    "https://vllm-public-assets.s3.us-west-2.amazonaws.com/"
    "vision_model_images/cherry_blossom.jpg"
)

# vision_attention_block_size=1024 bounds one image to 1024 patches; at
# Qwen3-VL's patch size of 16 that is a 512x512 budget.
PATCH_SIZE = 16
MAX_PATCHES = 1024
MAX_EDGE = PATCH_SIZE * int(MAX_PATCHES**0.5)   # 512

req = urllib.request.Request(IMAGE_URL, headers={"User-Agent": "vllm-neuron-epd"})
with urllib.request.urlopen(req, timeout=60) as resp:
    image = Image.open(io.BytesIO(resp.read()))

image = image.convert("RGB")
print(f"source image: {image.width}x{image.height}")
image.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)   # preserves aspect ratio
patches = (image.width // PATCH_SIZE) * (image.height // PATCH_SIZE)
print(f"resized to:   {image.width}x{image.height} ({patches} patches)")
assert patches <= MAX_PATCHES, f"{patches} patches exceeds block size {MAX_PATCHES}"

buf = io.BytesIO()
image.save(buf, format="JPEG", quality=90)
image_b64 = base64.b64encode(buf.getvalue()).decode()

client = OpenAI(api_key="EMPTY", base_url="http://localhost:18800/v1")
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
    max_tokens=128,
    temperature=0,
)
print(response.choices[0].message.content)
```

Save it as `validate_epd.py` and run it (`pip install openai pillow` first if
needed):

```bash
python3 validate_epd.py
```

The source image is 1770×1180, so it is downscaled to 512×341 (672 patches)
before the request goes out:

```text
source image: 1770x1180
resized to:   512x341 (672 patches)
The image shows a tall tower ... with cherry blossoms in the foreground ...
```

:::{tip}
To use a larger image at full resolution instead of downscaling, raise
`vision_attention_block_size` on **both** the VE and PD so the block covers your
largest input. It must match across the two pools.
:::

A generated description of the image confirms:

1. The router preprocessed the request and routed the image to VE-0 (port 18300).
2. VE-0 encoded the image and produced an embedding in its encoder cache.
3. PD-0 (port 18100) pulled the embedding via NIXL over LIBFABRIC and ran prefill.
4. PD-0 generated tokens, which the router detokenized and returned.

You can confirm the router sees both pools with a health check:

```bash
curl -s http://localhost:18800/healthcheck
# {"status":"ok","ve_instances":1,"pd_instances":1}
```

## Step 5: Scale to xEyPD

In this step, you will generalize the topology to multiple VE and PD engines. As
with DI's xPyD, the two pools scale independently: the router routes each image
to a VE by content hash and picks a PD round-robin, so the pools need not be the
same size.

A **2E7PD** topology fills a `trn2.48xlarge`: 2 × VE(DP=4) = 8 cores, plus
7 × PD(TP=8) = 56 cores, for 64 cores total.

To run xEyPD on a single node, launch each engine on a disjoint core slice with
its own port and (for VEs) a distinct `engine_id` and side-channel port, then
list every endpoint on the router. The core budget must satisfy
`num_ve × ve_tp + num_pd × pd_tp ≤ node core count`.

For example, to add a second VE (VE-1 on cores 12–15, port 18301,
side-channel 14604, `engine_id` `epd-ve-1`) and a second PD (PD-1 on cores
16–23, port 18101, `engine_id` `epd-pd-1`), launch them exactly like Steps 1–2
but with those values, then start the router with all four endpoints:

```bash
python3 examples/vllm_neuron/vllm/disaggregated_encoder/server.py \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port 18800 \
    --ve-timeout-s 600 \
    --pd-timeout-s 600 \
    --ve-endpoint 127.0.0.1:18300 \
    --ve-endpoint 127.0.0.1:18301 \
    --pd-endpoint 127.0.0.1:18100 \
    --pd-endpoint 127.0.0.1:18101
```

Repeat `--ve-endpoint` / `--pd-endpoint` once per engine. The router
consistent-hashes each image's `mm_hash` across the VE list (so a repeated image
reuses the same VE's warm encoder cache) and round-robins across the PD list.
The two selections are decoupled, so any VE's embedding can be consumed by any
PD.

### Launch a whole stack programmatically

Rather than launching each engine by hand, you can derive and bring up a full
single-node topology with the `EPDTopology` / `run_epd_single_node` helpers in
`examples/vllm_neuron/vllm/disaggregated_encoder/epd_common.py`. `build()`
derives the disjoint device slices, ports, side-channel ports, and unique engine
ids from the pool counts:

```python
from examples.vllm_neuron.vllm.disaggregated_encoder.epd_common import (
    EPDTopology,
    run_epd_single_node,
)

topo = EPDTopology.build(
    model="Qwen/Qwen3-VL-32B-Instruct",
    num_ve=2, ve_dp=4,      # 2 vision encoders, DP=4 each
    num_pd=7, pd_tp=8,      # 7 prefill+decode engines, TP=8 each
    max_model_len=8192,
    max_num_seqs=8,
    vision_seq_len=30720,   # num_images × (resolution/patch_size)² = 30 × (512/16)²
    num_images=30,
    quantization="bf16",
)

with run_epd_single_node(topo, artifacts_dir="/tmp/epd") as stack:
    print(stack.base_url)   # http://127.0.0.1:18800 — send requests here
```

The context manager brings up every engine and the router, waits for them to
report healthy, and tears the stack down on exit. The 2E7PD topology above needs
all 64 cores of a `trn2.48xlarge`.

#### Overriding the device slices

`build()` lays the pools out **contiguously** from `base_device`, so it cannot
express a gap. To skip specific cores — a faulted device, or cores another job
owns — build the topology and then rewrite each engine's `device_slice`. It is
only ever iterated into that engine's `NEURON_VISIBLE_DEVICES`, so any set of
cores works, and `EngineSpec` is frozen, hence `dataclasses.replace`:

```python
from dataclasses import replace

# Skip cores 20-23. Every PD start stays a multiple of pd_tp (trn2).
VE_STARTS = [0, 4]
PD_STARTS = [8, 24, 32, 40, 48, 56]

topo = EPDTopology.build(..., num_pd=len(PD_STARTS), pd_tp=8, ...)
topo = replace(
    topo,
    ve=[replace(e, device_slice=range(s, s + e.tp))
        for e, s in zip(topo.ve, VE_STARTS, strict=True)],
    pd=[replace(e, device_slice=range(s, s + e.tp))
        for e, s in zip(topo.pd, PD_STARTS, strict=True)],
)

for e in topo.ve + topo.pd:                      # fail fast, not 20 min into compile
    cores = list(e.device_slice)
    assert cores[0] % e.tp == 0, f"{e.label} start {cores[0]} not aligned to tp={e.tp}"
```

Two constraints still apply, and neither is checked for you: each slice must
avoid the excluded cores, and on `trn2` each must start on a multiple of its own
TP degree. Assert both before launching — a misaligned slice compiles fine and
only fails in prefill warmup minutes later. Excluding cores also shrinks the
budget, so a pool may need to lose an engine: dropping one device from a 64-core
node leaves 60 cores, which no longer fits 2E7PD (64) but does fit 2E6PD (56).
Reduce `num_pd` rather than `pd_tp` — changing TP invalidates the compile cache.
Cores that cannot form an aligned slice of the required width are simply left
idle.

:::{note}
The example router and launcher target a **single node** — all endpoints default
to `127.0.0.1`, and `EPDTopology` partitions the cores of one instance. The
router's `--ve-endpoint` / `--pd-endpoint` flags accept an arbitrary `host:port`,
so a multi-node layout is possible in principle, but it is not covered by the
reference launcher. A multi-node deployment additionally requires
EFA connectivity and security-group rules between instances (see the
[DI tutorial's multi-node section](tutorial-di-1p1d-xpyd.md#step-5-scale-to-xpyd-multi-node)).
:::

## Confirmation

You have a working EPD deployment that:

- Separates the vision encoder from the prefill+decode language model across
  different NeuronCore slices.
- Transfers vision embeddings from VE to PD via the `NeuronNixlECConnector` over
  NIXL / LIBFABRIC (HBM→HBM RDMA READ).
- Routes each image to a VE by content hash (warm-cache affinity) and each
  request to a PD round-robin, through an OpenAI-compatible router.
- Supports asymmetric scaling (different numbers of VE and PD engines, and
  different DP/TP per pool).

## Common issues

- **`TypeError: Qwen3ForCausalLM.from_configs() got an unexpected keyword
  argument 'text_neuron_config'`**: The server is loading a **text-only** model
  while `vision_neuron_config` is set, so the runner builds a multimodal model
  from a text-only architecture. The usual cause is an empty or unset `MODEL`:
  `vllm serve`'s model argument is positional and optional, so an unquoted empty
  `$MODEL` disappears from the command line and vLLM falls back to its default
  `Qwen/Qwen3-0.6B`. Check the startup banner's `model` line — it shows what was
  actually loaded — and confirm your checkpoint's `config.json` lists
  `"architectures": ["Qwen3VLForConditionalGeneration"]`.

- **`HfHubHTTPError: 429 Too Many Requests: you have reached your 'api' rate
  limit`**: Not an EPD problem — vLLM resolves the model id against the Hugging
  Face API in `create_model_config`, before the engine is built, and
  unauthenticated requests are rate-limited **per IP** (shared on NAT'd or
  multi-tenant hosts). Launching a pool multiplies those calls, so a programmatic
  bring-up trips the limit more easily than a single server. Serve from a local
  checkpoint directory with `HF_HUB_OFFLINE=1`, or export an `HF_TOKEN` for the
  higher authenticated quota (see the tip in
  [Prepare your environment](#prepare-your-environment)). The error message states
  how many seconds to wait before retrying.

- **Server fails at startup with `unsupported backend 'LIBFABRIC'`**: NIXL's
  LIBFABRIC backend can't load `libcuda.so.1` / `libfabric.so.1`. Install the DI
  dependencies (see Prerequisites) and confirm `python -c "import nixl"` works.

- **`No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'`**: Benign on
  Neuron. PyTorch's `cpp_extension` probes for CUDA at import; nothing in the EPD
  path uses it, and NIXL transfers ride LIBFABRIC/EFA. The same applies to
  `Failed to import from vllm._C` warnings.

- **`Connection refused` from the router**: Both the VE and PD servers must be
  fully started before the router can serve requests. First-time Neuron
  compilation can take several minutes — wait for the `Uvicorn running` message
  on each server, and confirm `/healthcheck` reports the expected VE and PD
  counts.

- **Embedding transfer stalls or times out**: Ensure LIBFABRIC/EFA is available
  (`fi_info -p efa`) and that `FI_EFA_ENABLE_SHM_TRANSFER=0` and
  `FI_EFA_USE_DEVICE_RDMA=1` are set on both pools. Confirm each VE has a unique
  `engine_id` and a reachable `side_channel_port`.

- **`TP != len(NEURON_VISIBLE_DEVICES)` assertion**: The tensor-parallel size
  must equal the number of cores in that engine's `NEURON_VISIBLE_DEVICES`
  slice. For a VE, `--tensor-parallel-size` is the core count; `dp_size` is a
  separate vision data-parallel degree inside `vision_neuron_config`.

- **`Collective operation REDUCE_SCATTER ... is currently not supported`,
  followed by `Failed to schedule neff execution. status=2 message=Invalid` and
  `Model warmup failed for prefill bucket ...`**: On `trn2`, the engine's core
  slice is **misaligned**. A slice must start on a multiple of its own TP degree,
  so a TP=8 engine must start at core 0, 8, 16, …; a slice like `4-11` has the
  right core *count* but the wrong *offset*, and the TP group's sequence-parallel
  `reduce_scatter` cannot be mapped onto a hardware collective. This is a property
  of the `trn2` device interconnect, whose cores form a 2D torus that the runtime
  matches against a fixed set of collective algorithms; a slice that straddles the
  torus rows the algorithms expect falls through to no algorithm at all. Nothing
  catches this at startup — the engine loads and compiles normally, then fails
  during prefill warmup. Note the error names the collective and not the core
  layout, so it is easy to misread as a size or bucket problem: shrinking
  `num_batched_tokens_buckets` (or enabling segmented prefill with
  `kv_segment_size_buckets`) changes the reported `total_size` but reproduces the
  same failure. Fix the offset instead — move the engine to an aligned slice such
  as `0-7`, `8-15`, or `56-63`.

  This also applies to `EPDTopology.build` in `epd_common.py`, which lays the VE
  pool out first and gives the PD pool whatever follows. That is aligned when the
  VE pool's total width is a multiple of the PD's TP degree — as in 2E7PD, where
  2 × VE(TP=4) = 8 cores puts PD-0 at core 8 — but a 1E1PD built the same way
  yields `VE 0-3` and `PD 4-11`, which is exactly the misaligned case above. When
  a topology has a single small VE pool ahead of a wider PD pool, pass
  `base_device` or lay the pools out by hand so each PD slice starts on a multiple
  of `pd_tp`.

  Later platforms whose cores are connected through a switch rather than a torus
  may not need the offset to be aligned, since any set of cores can form a
  collective group. That has not been verified for EPD — the topologies exercised
  in CI are aligned on every platform — so keep the slices aligned unless you have
  measured otherwise.

- **`ECLoadFailure: Encoder-cache load failed for mm_hashes: [...]`, and the PD
  engine core dies**: PD could not read a vision embedding out of the VE's
  on-device encoder cache. The failure always surfaces on the *consumer* side as a
  cache-load error and takes the engine core down with it, but the cause is usually
  upstream of PD. In order of likelihood:

  1. **The EFA environment variables are missing on one of the engines.** Check
     `FI_EFA_ENABLE_SHM_TRANSFER=0` and `FI_EFA_USE_DEVICE_RDMA=1` in the VE *and*
     PD terminals — `env | grep FI_EFA` in each. This is easy to miss because the
     exports are set once in "Prepare your environment" while each engine runs in
     its own shell, and because nothing complains until the first request. If the
     PD log shows `EC wait_for_load timed out after ...`, the READ was issued and
     never landed, which points here.
  2. **An oversized image.** One image exceeding `vision_attention_block_size`
     patches (1024 in this tutorial, a 512×512 budget at patch size 16) fails to
     encode on the VE, so there is nothing for PD to read. Check the VE's log for
     the encode error, then downscale the image (see
     [Step 4](#step-4-validate-the-1e1pd-deployment)) or raise
     `vision_attention_block_size` on **both** pools.
  3. **A genuinely broken transport** — see the embedding-transfer entry above.

- **One engine hangs for minutes, then `TDRV:tdrv_init_wait_ncs_ready Failed to
  reset nd <N> nc_map 0x… timeout (result=-110)` → `RuntimeError: Engine core
  initialization failed`**: A Neuron device is in a faulted state and is not
  responding to reset (`-110` is `ETIMEDOUT` from the driver). This is a hardware
  problem, not a configuration one — restarting the stack will not clear it.
  Confirm which device by comparing reset counters across devices:

  ```bash
  B=/sys/devices/virtual/neuron_device
  for d in $(seq 0 15); do for c in 0 1 2 3; do
    printf "nd%s nc%s req=%s fail=%s\n" $d $c \
      "$(cat $B/neuron$d/neuron_core$c/stats/other_info/reset_req_count/total 2>/dev/null)" \
      "$(cat $B/neuron$d/neuron_core$c/stats/other_info/reset_fail_count/total 2>/dev/null)"
  done; done
  ```

  A healthy device reports `fail=0` on every core; a faulted one reports a nonzero
  `fail` on all of them. Note the failure takes down the **whole stack**, not just
  the affected engine: `run_epd_single_node` health-checks engines concurrently and
  the first exception unwinds the `ExitStack`, which terminates every other engine.
  So a clean `SIGTERM`/shutdown log on a healthy engine is a *symptom*, not the
  cause — find the one engine whose log ends in a traceback. Until the device is
  reset or the instance replaced, route around it by assigning explicit device
  slices (see [Step 5](#step-5-scale-to-xeypd)), keeping every slice free of the
  faulted device and still aligned to its own TP degree.

- **Core-slice binding race on single node**: Launch engines a few seconds apart
  (the `run_epd_single_node` launcher staggers Popens by 10 s) so they don't race
  on `NEURON_VISIBLE_DEVICES` binding and the shared compile cache.

- **Wrong or empty output on repeated images**: Keep prefix caching **off** on
  both pools (`--no-enable-prefix-caching`). With APC on, a KV prefix hit can
  bypass encoder-input scheduling so the embedding is never pulled.

## Clean up

Stop all processes:

```bash
pkill -f "vllm serve"
pkill -f "disaggregated_encoder/server.py"
```

If you launched EC2 instances specifically for this tutorial, terminate them to
avoid ongoing charges.

## Next steps

- [Deploy Qwen3-VL-32B](tutorial-qwen3-vl-32b.md) — serve the multimodal model
  on a single instance.
- [Disaggregated inference: 1P1D and xPyD](tutorial-di-1p1d-xpyd.md) — split
  prefill from decode (the other disaggregation axis).
- [On-device encoder cache design document](../design/multimodal/on_device_encoder_cache.md)
  — the block-based HBM cache that VE writes and PD reads.
