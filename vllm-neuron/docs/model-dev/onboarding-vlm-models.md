# How to onboard a Vision-Language Model to vLLM Neuron

<!-- meta: description: Onboard a new vision-language model (VLM) architecture
to vLLM Neuron, adding a vision encoder tower on top of the text-decoder
onboarding flow. -->
<!-- meta: keywords: vLLM, Neuron, VLM, vision-language model, multimodal,
vision encoder, Qwen3-VL, Trainium, block packing, M-RoPE, encoder cache -->
<!-- meta: date_updated: 2026-07-31 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: NMI-325 -->

## Task overview

This topic covers the **vision-language-model-specific** parts of onboarding a
model to vLLM Neuron. A VLM adds a **vision encoder** tower that turns
images/videos into embeddings, which are merged into the text decoder's token
sequence. Everything about the text decoder — config, attention/MLP patterns,
KV cache, weight loading, registration, compilation, accuracy validation, and
benchmarking — is identical to a text-only model.

:::{important}
This guide is a **companion** to the base
[How to onboard a model to vLLM Neuron](onboarding-models.md). It follows the
same structure (steps 1a–5) but only documents what differs for a VLM. For every
step, read the base guide first — the sections below add the vision-specific
delta on top of it. Steps not listed here (e.g., the text decoder's attention
and MLP patterns) are unchanged.
:::

The [Qwen3-VL implementation](https://github.com/aws-neuron/vllm-neuron)
(`vllm_neuron/model/qwen3_vl/`) is the canonical reference throughout this guide:
a dense text decoder plus a ViT vision encoder, with deepstack feature injection,
block-packed vision attention, M-RoPE, and an on-device encoder cache.

## Prerequisites

Same as the [base guide](onboarding-models.md#prerequisites). In addition:

- **Familiarity with the multimodal design docs**, referenced per-section below:
  [block packing vision attention](../design/multimodal/block_packing_attention.md),
  [M-RoPE](../design/multimodal/mrope.md), and the
  [on-device encoder cache](../design/multimodal/on_device_encoder_cache.md).

## End-to-end onboarding flow

A VLM follows the same five-stage flow as the base guide, with a vision encoder
tower added alongside the text decoder:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Implement model     2. Register with   3. Compile &   4. Validate   │
│     text decoder  ───┐      vLLM registry      smoke test     accuracy  │
│     vision encoder ──┘                                                  │
│     (+ encoder cache, block packing, M-RoPE)             5. Benchmark   │
└─────────────────────────────────────────────────────────────────────────┘
```

The text decoder and vision encoder are independent modules — you can implement
and validate them in parallel, each against its own HF reference. They converge
only at the
[top-level model class](#top-level-model-class-yourvlmodelforconditionalgeneration),
which composes both towers once each is working. Validating each tower in
isolation keeps text-decoder bugs, vision-encoder bugs, and vision↔text merge
bugs separable.

## Instructions

### 1. Implement the model

A VLM directory should have vision-encoder modules and multimodal utilities alongside
the text-decoder files:

```text
vllm_neuron/model/your_vlm/
├── __init__.py
├── config.py               # Composed config: text_config + vision_config
├── factory.py              # Factory for vLLM ModelRegistry (validates vision + text)
├── model.py                # Text decoder + top-level multimodal model
├── vision_encoder.py       # Vision encoder (ViT) implementation
├── weight_loaders.py       # Weight loaders for both towers
└── utils/
    ├── block_packing.py    # FFD packing of images into fixed-size attention blocks
    ├── preprocessing.py    # CPU: vision RoPE, position embeddings, attention bounds
    ├── mrope.py            # 3D (M-RoPE) position-id computation
    └── merge_vision_embeds.py  # Scatter vision embeddings into the text sequence
```

#### 1a. Define the model config (`config.py`)

The text decoder config is unchanged from the
[base guide](onboarding-models.md#1a-define-the-model-config-configpy). A VLM
composes **two** sub-configs — one per tower — each carrying its own Neuron
config object. This mirrors the HuggingFace VLM config structure (a nested
`text_config` + `vision_config`) and decouples the two towers' Neuron configs, so
each can be tuned independently — for example, running the vision encoder at a
different parallelism degree than the text decoder:

```python
from dataclasses import dataclass
from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig

@dataclass
class YourVLMTextConfig:
    # Text decoder architecture params (same as a text-only model)
    hidden_size: int = 5120
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    # ...
    neuron_config: NeuronConfig | None = None

@dataclass
class YourVLMVisionConfig:
    # Vision encoder architecture params
    hidden_size: int = 1152
    num_heads: int = 16
    depth: int = 27                       # number of ViT blocks
    patch_size: int = 16
    spatial_merge_size: int = 2
    out_hidden_size: int = 5120           # must match text hidden_size after merge
    deepstack_visual_indexes: list | None = None
    vision_neuron_config: VisionNeuronConfig | None = None

@dataclass
class YourVLMConfig:
    text_config: YourVLMTextConfig
    vision_config: YourVLMVisionConfig
    # Special token IDs used to locate vision placeholders in the token sequence
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig,
                     neuron_config: NeuronConfig = None,
                     vision_neuron_config: VisionNeuronConfig = None):
        # Parse hf_config.text_config / hf_config.vision_config into the two
        # sub-configs, attaching neuron_config and vision_neuron_config.
        ...
```

The `VisionNeuronConfig` dataclass (provided by the plugin at
`vllm_neuron.model.neuron_config`) is the vision-encoder counterpart to
`NeuronConfig`. It carries vision-specific settings: parallelism degrees
(`tp_size`, `dp_size`), vision bucket configurations
(`num_vision_tokens_buckets`), block-packed attention (`vision_attention_block_size`),
on-device encoder cache sizing (`encoder_cache_num_blocks`,
`encoder_cache_min_hold_time_ms`). See the
[configuration reference](../guides/reference-configuration.md#vision-encoder-options-vision_neuron_config)
for details. See [`vllm_neuron/model/qwen3_vl/config.py`](https://github.com/vllm-project/vllm-neuron/blob/HEAD/vllm_neuron/model/qwen3_vl/config.py) for a concrete example.

#### 1b. Implement model components (`model.py`)

The text decoder components — attention, MLP, decoder layer, backbone — follow
the [base guide](onboarding-models.md#1b-implement-model-components-modelpy)
unchanged. The subsections below add the vision-specific components.

##### Vision encoder class pattern

The vision encoder is a Vision Transformer (ViT) that consumes packed image
patches and produces embeddings in the text decoder's hidden dimension. Unlike
the text decoder, it has **no KV cache, no decode path, and no causal mask** — it
runs a single forward pass over the full sequence. The reference pipeline
(`vllm_neuron/model/qwen3_vl/vision_encoder_bf16.py`):

```text
pixel_values [num_blocks, block_size, patch_dim]
  → PatchEmbed (convolution)  + position embeddings
  → N × ViT blocks (LayerNorm → Attention → LayerNorm → MLP)
       ↳ at deepstack layer indices: extract intermediate features
  → PatchMerger (spatial merge → project to text hidden_size)
  → write embeddings into the encoder cache buffer
```

Key differences from the text decoder's attention/MLP patterns:

- **Independent parallelism.** The vision encoder uses its own TP/DP process
  group (e.g. `get_neuron_vision_tp_group()`), decoupled from the text decoder's
  TP. Block-level DP scatters image blocks across ranks and `all_gather`s the
  merged output. There is no sequence parallelism — every rank sees the full
  sequence. See
  [vision encoder parallelism](../design/parallelism/vision_encoder_parallelism.md).
- **Bidirectional attention with bounds masking.** Vision attention is full
  (non-GQA) and non-causal. Instead of a causal mask it uses per-image/frame
  `bound_min`/`bound_max` to isolate attention within each image or video frame.
- **Custom weight loaders.** Fused vision QKV checkpoints often store
  `[Q_all | K_all | V_all]`, which a naive column shard would split incorrectly.
  Use an interleaved-head loader (see
  {ref}`1d: weight loading <vlm-implement-weight-loading>`).

**Block-packing vision attention.** Multi-image requests produce
variable-length token sequences. Rather than padding every image to the largest
size (batch packing) or computing a full `seq_len × seq_len` attention matrix
(sequence packing), the encoder packs images into fixed-size blocks using
First-Fit-Decreasing (FFD) bin packing and processes each block as an
independent batch element with no cross-block attention. This yields 2×–2.5×
efficiency on heterogeneous workloads. The block size is a compile-time constant
(`vision_attention_block_size`, a multiple of 128) that must be ≥ the largest
single image's token count. FFD packing, position shuffling, and
`bound_min`/`bound_max` computation happen on CPU in preprocessing before kernel
dispatch. For the algorithm, block-size tuning, and efficiency analysis, see
[block packing vision attention](../design/multimodal/block_packing_attention.md).

**Multimodal rotary position embedding (M-RoPE).** Some VLM text decoders (for
example Qwen3-VL) use a multi-axis RoPE that assigns positions along temporal,
height, and width axes.
Text tokens receive identical sequential positions on all three axes (reducing
to standard 1D RoPE), while vision tokens receive their spatial grid coordinates.
The 3D position IDs are computed on CPU from each item's `grid_thw` in the
`NeuronModelRunner` (during input preparation) and threaded into the model's
`forward()` as the `positions` tensor; the decoder's rotary embedding module
then applies per-axis rotation using the model's `mrope_section` split. For the
position-id layout, the runner integration, and the modeling-code contract, see
[M-RoPE](../design/multimodal/mrope.md).

##### Top-level model class (`YourVLModelForConditionalGeneration`)

The VLM top-level class composes both towers (`self.visual` +
`self.language_model` + `self.lm_head`) and implements the same runner interface
methods as the text-only
{ref}`YourModelForCausalLM <top-level-model-class-yourmodelforcausallm>`
(`get_kv_spec`, `bind_kv_cache`, `load_weights`, `from_configs`). It adds these
**vision-specific** methods and behaviors on top:

```python
class YourVLModelForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.visual = YourVLMVisionModel(config.vision_config)      # vision tower
        self.language_model = YourVLMTextModel(config.text_config)  # text tower
        self.lm_head = neuron_nn.ColumnParallelLinear(...)
        ...

    def embed_multimodal(self, encoder_cache, mm_hashes, pixel_values, grid_thw):
        """VLM-only. Runs the vision encoder and writes embeddings to the cache.

        Called by the NeuronModelRunner before prefill for cache-miss items:
        allocate cache blocks, select the vision bucket, FFD-pack the images,
        run CPU preprocessing (vision RoPE, position embeddings, attention
        bounds), dispatch the vision encoder, and scatter-write its output
        directly into the encoder cache buffer.
        """
        ...

    def forward(self, input_ids, positions, attn_metadata, sampling_positions,
                sampling_params, *, vision_embedding_blocks=None,
                vision_positions=None, ...):
        """Text decoder forward, extended to merge vision embeddings.

        During prefill, `merge_vision_embeddings` scatters the cached vision
        embedding blocks into the token hidden states at the vision-token
        positions. Deepstack features (if any) are added to the hidden states
        after the corresponding early decoder layers.
        """
        ...

    def load_weights(self, checkpoint_path, device, cache_dir):
        """Loads both towers. Delegates the vision encoder to its own loader,
        which uses the independent vision TP group (see step 1d)."""
        ...
```

Compared to `YourModelForCausalLM`, the additions are:

| Method / behavior | Purpose |
| --- | --- |
| `embed_multimodal()` | Run the vision encoder, pack images, write embeddings to the encoder cache. Not present in text-only models. |
| Vision-aware `forward()` | Accepts vision embedding blocks + positions; calls `merge_vision_embeddings`; injects deepstack features into early decoder layers. |
| Two-tower composition | `self.visual` + `self.language_model` instead of a single backbone. |
| `load_weights()` delegation | Loads the vision tower via its own loader on the independent vision TP group, in addition to the text decoder. |

##### Encoder cache

The **on-device encoder cache** bridges the vision encoder to the text decoder
and lets repeated media (e.g. the same image across multi-turn chat) skip
re-encoding. It is a pre-allocated HBM buffer of fixed-size blocks
(`[num_blocks, block_size, fat_dim]`) keyed by a per-item content hash
(`mm_hash`). The vision encoder writes its output directly into cache blocks; at
prefill, the runner passes zero-copy block views into the graph, which scatters
them into the token sequence via a fixed-shape position map. Keeping the cache
outside the compiled graph (a runner-managed device tensor) avoids the CPU↔device
round-trip that would otherwise gate the async dispatch pipeline.

Integration is split between the model and the runner, mirroring the KV cache
contract:

1. **Model (`embed_multimodal`)** — allocates cache blocks for cache-miss items,
   runs the vision encoder, and scatter-writes the output into the buffer.
2. **`NeuronModelRunner`** — owns the buffer and slot allocator; on the
   scheduler's encode signal it calls `embed_multimodal`, and on the evict signal
   it frees blocks back to the free queue. Before prefill it gathers the block
   views and builds the position map consumed inside the graph.

Eviction is driven by vLLM's upstream `EncoderCacheManager` (unchanged), which
ref-counts items and signals which `mm_hash`es to evict. For the buffer layout,
gather/merge mechanism, sizing, and eviction design, see the
[on-device encoder cache design](../design/multimodal/on_device_encoder_cache.md).

#### 1c. Define the factory (`factory.py`)

Same pattern as the
[base guide](onboarding-models.md#1c-define-the-factory-factorypy). The
factory's `_validate_config` should additionally reject unsupported vision
configurations — for example, quantization modes the vision encoder doesn't
support, or a `tp_size`/`dp_size` split that violates
`tp_size * dp_size == world_size`. Optionally implement any multimodal capability
protocols the runtime expects (e.g. spatial-merge factor and max-pixels helpers
so vLLM can size vision placeholders correctly).

(vlm-implement-weight-loading)=

#### 1d. Implement weight loading

The text decoder's weight-loading system —
{ref}`mappings, weight loaders, and the checkpoint reader <text-model-implement-weight-loading>`
— is unchanged. Two VLM additions:

- **Vision encoder mappings.** Add a full `...visual.*` checkpoint-key →
  parameter-name mapping for the vision tower, and load it through the vision
  encoder's own loader on the **independent vision TP group** (not the text TP
  group).
- **Interleaved vision QKV loader.** Vision encoders commonly store fused QKV as
  a single `[3*H, H]` matrix laid out as `[Q_all | K_all | V_all]`. A naive
  column shard would give rank 0 all of Q. Instead, use an interleaved-head
  loader that, for each rank, slices the corresponding head range from each of
  Q, K, and V. See `vis_qkv_weight_loader` in
  `vllm_neuron/model/qwen3_vl/weight_loaders.py` for the reference
  implementation (and the matching bias loader).

#### 1e. Write the model README

Same as the [base guide](onboarding-models.md#1f-optional-write-the-model-readme). For a
VLM, document **both** towers: add a vision-encoder architecture table (hidden
size, heads, depth, patch size, spatial merge size, deepstack indices,
activation, normalization, position embedding) and note the vision-specific
feature status (block-packed attention, deepstack, vision TP/DP, encoder cache).
See `vllm_neuron/model/qwen3_vl/README.md` for the reference.

### 2. Register the model with vLLM

Same as the [base guide](onboarding-models.md#2-register-the-model-with-vllm).
Register the top-level `YourVLModelForConditionalGeneration` class; the string
key must match the `architectures` field in the model's `config.json`.

### 3. Compile and run a smoke test

The [warmup, compilation, and runtime-padding](onboarding-models.md#3-compile-and-run-a-smoke-test)
mechanics are unchanged, with one addition: a VLM compiles **two sets of NEFFs** —
the text decoder (bucketed over `num_batched_tokens_buckets` /
`num_seqs_buckets`) and the vision encoder (bucketed over
`num_vision_tokens_buckets`). Each vision bucket is a discrete padded shape over
the number of vision patches per encoder forward pass; more buckets mean longer
warmup but less padding waste. Start with a single vision bucket sized for your
target workload and add more as needed.

A minimal multimodal smoke test passes both text and an image through the
offline `LLM` API to trigger compilation of both towers. For a runnable
deployment walkthrough (online serving, offline inference, and the full
`additional_config` with `neuron_config` + `vision_neuron_config`), follow the
[Deploy Qwen3-VL-32B tutorial](../tutorials/tutorial-qwen3-vl-32b.md) rather than
duplicating it here.

### 4. Validate accuracy

The systematic accuracy workflow, CPU-mode debugging, and
`tensor_capture`/`tensor_replacement` tooling from the
[base guide](onboarding-models.md#4-validate-accuracy) apply unchanged, and the
[accuracy debugging guide](accuracy-debugging-guide.md) remains the reference for
isolating drift. VLM-specific additions:

- **Validate the vision encoder separately.** Add module-level three-way
  comparisons (FP32 HF, BF16 HF, BF16 vLLM Neuron) for each vision component —
  patch embedding, attention, MLP, patch merger, and the full ViT — mirroring the
  text-decoder module tests. This isolates encoder bugs from decoder bugs and
  from the vision↔text merge.
- **Validate CPU preprocessing against the HF reference.** The vision RoPE,
  position-embedding interpolation, and attention-bounds computations run on CPU;
  test them directly against the reference implementation, since a subtle
  mismatch here corrupts attention without any runtime error.
- **Extend the end-to-end logit test to multimodal inputs.** Run the base guide's
  end-to-end logit test with image *and* video prompts, so the whole
  vision→merge→decode path is exercised — the vision encoder, the vision-embedding
  merge, and the M-RoPE/vision-position mapping — not just the text decoder. Build
  the HF goldens from multimodal prompts and pass a multimodal generate function
  to the same `multi_prompt_logit_validation` helper:

  ```python
  # 1. HF reference goldens (FP32 baseline + BF16 target) for image and/or video prompts
  goldens = compute_multimodal_reference_goldens(
      model_checkpoint=checkpoint,
      target_dtype=torch.bfloat16,
      prompts=prompts,
      images=prompt_images,      # and/or videos=prompt_videos
      output_length=OUTPUT_LEN,
  )

  # 2. vLLM Neuron generate fn (raw_logits mode), then three-way compare
  llm = LLM(model=checkpoint, **vllm_args)          # vision_neuron_config in additional_config
  result = multi_prompt_logit_validation(
      prompts_input_ids=goldens["input_ids"],
      generate_fn=create_multimodal_vllm_generate_fn(llm, OUTPUT_LEN),
      prompts_expected_logits=goldens["dtype_logits"],   # BF16 HF
      prompts_baseline_logits=goldens["fp32_logits"],    # FP32 HF
      tol_map=MM_TOL_MAP,        # looser than text-only: vision + many vision tokens add BF16 noise
  )
  ```

  You may need a looser top-K tolerance than the text-only model — the vision
  encoder plus thousands of vision tokens can accumulate extra BF16 rounding
  before the LM head. Logit validation is a stricter, more sensitive check; the
  accuracy evaluation benchmark below is the final accuracy gate.
- **Validate end-to-end with a VLM benchmark.** Serve the model and run a
  vision-language eval harness (e.g. [VLMEvalKit](https://github.com/open-compass/VLMEvalKit))
  against the OpenAI-compatible endpoint, then compare the score to the HF
  reference:

  ```bash
  # 1. Serve the model (see the deployment tutorial for the full command)
  vllm serve <checkpoint> --served-model-name <name> [neuron + vision flags] &

  # 2. Run the eval harness against the running server over its OpenAI endpoint
  python VLMEvalKit/run.py --data POPE --model <name> --verbose
  ```

  The Neuron score should match the GPU/HF reference score within a small
  tolerance.

### 5. Benchmark and tune performance

The [base guide's](onboarding-models.md#5-benchmark-and-tune-performance)
benchmarking flow and text-decoder tuning parameters apply unchanged. For a VLM,
use the `random-mm` dataset so `vllm bench serve` generates synthetic image
inputs alongside the random text, exercising the vision encoder under load:

```bash
# Benchmark with synthetic multimodal requests (server started as in the base guide)
vllm bench serve \
    --model /path/to/model \
    --dataset-name random-mm \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 100 \
    --random-mm-base-items-per-request 1 \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --random-mm-limit-mm-per-prompt '{"image": 4, "video": 0}' \
    --random-mm-bucket-config '{(512, 512, 1): 0.5, (1024, 1024, 1): 0.5}'
```

- `--random-mm-base-items-per-request` — baseline number of multimodal items
  (images/videos) per request.
- `--random-mm-num-mm-items-range-ratio` — how much the per-request item count
  varies around that baseline (`0.0` = fixed, every request gets the base count).
- `--random-mm-limit-mm-per-prompt` — per-modality cap on items in a single
  prompt; must stay within the server's `--limit-mm-per-prompt`.
- `--random-mm-bucket-config` — a `{(H, W, T): probability}` map that sets the
  sampled media resolutions (`T=1` is an image; `T>1` is a video with that many
  frames). Match these to your target workload — resolution drives the vision
  token count, and therefore which `num_vision_tokens_buckets` and
  `vision_attention_block_size` you should compile for.

VLM-specific optimization options:

- **`vision_attention_block_size`** — trade intra-block padding against
  fully-padded trailing blocks; see the block-size guidance in
  [block packing vision attention](../design/multimodal/block_packing_attention.md).
- **`num_vision_tokens_buckets`** — size for the most images a **single request**
  carries. The Neuron scheduler admits one request into prefill per step (batch
  size 1) and schedules its vision-encoder inputs with that prefill, so the encoder
  processes one request's images per forward pass (a request may still contain many
  images or video frames).
- **Vision `tp_size` / `dp_size`** — full-DP favors multi-image throughput;
  higher vision TP favors single-large-image latency. See
  [vision encoder parallelism](../design/parallelism/vision_encoder_parallelism.md).
- **Encoder cache sizing** (`encoder_cache_num_blocks`) — larger caches raise the
  cross-request/multi-turn hit rate at the cost of HBM shared with the KV cache.

To profile where time goes, see
[how to profile workloads](../guides/how-to-profile-workloads.md). For a VLM this
is worth doing on image/video-heavy inputs, where the vision encoder can take a
large share of prefill. Look across the three profiling levels for
vision-specific costs:

- **Torch profile (CPU/host):** the HF multimodal pre-processing time, the
  Neuron-specific host work in `embed_multimodal()` before and after the vision
  encoder graph executes, and the CPU-side prefill input preparation
  (`_gather_mm_embeddings`).
- **NRT profile (runtime):** bubbles between graph executions, and the encoder
  cache data transfer — `nrt_tensor_write` (encoder output into the cache buffer)
  and `nrt_tensor_read`.
- **Device profile:** on-device activity within the vision encoder and text
  decoder graphs themselves (compute vs. idle per engine per NeuronCore).

For how to reason about VLM performance in depth — setting a roofline latency
target, choosing vision and text sharding, and closing the gap with the general
multimodal features and fused kernels — see
[Optimizing a Vision-Language Model](optimizing-vlm-models.md).

## Confirm your work

In addition to the [base guide's checks](onboarding-models.md#confirm-your-work)
(compilation, text accuracy, performance, serving), confirm:

1. **Vision compilation:** Both the text-decoder and vision-encoder NEFFs compile
   and load from cache on subsequent runs.
2. **Multimodal accuracy:** Single-image, multi-image, and (if supported) video
   inputs produce correct outputs, and the vision encoder matches the HF reference
   at the module level.
3. **Encoder cache:** Repeated media reuses cached embeddings (no re-encode), and
   text-only requests are unaffected.

## Common issues

The [base guide's common issues](onboarding-models.md#common-issues)
(non-compilable ops, weight-loading `KeyError`, accuracy drift) apply to both
towers. VLM-specific:

### M-RoPE position IDs not offset correctly

- **Possible solution:** M-RoPE uses a **different** set of position IDs from the
  absolute sequential positions. The sequential positions are used for KV cache
  slot mapping (and the attention mask), while the 3D rotary (M-RoPE) position IDs
  are used only for the rotary embedding. Because vision tokens share grid
  coordinates, the M-RoPE max is smaller than the sequential max after an image,
  so the two diverge — the gap is captured by `mrope_position_delta`. Common
  symptoms: using M-RoPE values for cache indexing collapses multiple tokens onto
  the same slot and corrupts the KV cache; forgetting to apply
  `mrope_position_delta` when continuing into decode leaves generated tokens at
  the wrong positions. Keep the two position streams separate and verify the
  decode offset. See [M-RoPE](../design/multimodal/mrope.md) for the dual
  position-id design and the delta computation.

## Related information

- [How to onboard a model to vLLM Neuron](onboarding-models.md) — The base guide;
  read it first. This guide only covers the vision-specific delta.
- [Block packing vision attention](../design/multimodal/block_packing_attention.md)
  — FFD packing, block-size tuning, efficiency analysis.
- [M-RoPE](../design/multimodal/mrope.md) — Multimodal rotary position embeddings
  and their runner/model integration.
- [On-device encoder cache](../design/multimodal/on_device_encoder_cache.md) — Buffer layout,
  gather/merge, sizing, and eviction.
- [Vision encoder parallelism](../design/parallelism/vision_encoder_parallelism.md)
  — Vision TP/DP, independent of text TP.
- [Deploy Qwen3-VL-32B tutorial](../tutorials/tutorial-qwen3-vl-32b.md) — Runnable
  deployment walkthrough for the reference VLM.
- [Qwen3-VL model card](../model-recipes/qwen3-vl.md) — Feature matrix and accuracy
  (VLMEvalKit POPE) results.
- [Accuracy debugging guide](accuracy-debugging-guide.md) — Diagnose accuracy
  issues in either tower.
