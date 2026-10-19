# Tutorial: Deploy Qwen3-ASR-1.7B with vLLM Neuron

<!-- meta: description: Deploy Qwen3-ASR-1.7B for online and offline speech
transcription with vLLM Neuron on AWS Trainium2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3-ASR, Qwen3-ASR-1.7B, ASR, speech
recognition, audio transcription, online serving, offline inference, Trn2,
Trainium -->
<!-- meta: date_updated: 2026-10-01 -->
<!-- Content type: procedural-tutorial -->

This tutorial guides you through deploying
[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) with vLLM
Neuron. You will transcribe audio through an online API or the offline Python
API.

## Overview

Qwen3-ASR converts speech into text. This tutorial uses BF16 on Trn2 with eight
NeuronCores.

Choose one deployment path after you prepare the environment:

| Deployment | Interface | Use case |
| --- | --- | --- |
| Online | `POST /v1/audio/transcriptions` | Applications that send audio requests to a server |
| Offline | `LLM.generate()` | Local batch jobs without an HTTP server |

Both paths use the same model and Neuron configuration.

## Tested versions

This tutorial was validated with the following components.

| Component | Version |
| --- | --- |
| Neuron SDK | 2.33 |
| vLLM Neuron plugin | Shipped with the Neuron 2.33 release |
| vLLM (upstream) | The version pinned by the vLLM Neuron plugin for Neuron 2.33 |
| Qwen3-ASR-1.7B checkpoint | `Qwen/Qwen3-ASR-1.7B` (Hugging Face) |
| Instance | `trn2.48xlarge` |

:::{note}
The current release validates Qwen3-ASR-1.7B on Trn2. It does not validate the
Qwen3-ASR-0.6B checkpoint.
:::

## Prerequisites

- A Trn2 instance with enough NeuronCores for the configured parallelism. The
  examples use eight NeuronCores.
- A working vLLM Neuron environment. See the
  [setup guide](../getting-started/setup-guide.md).
- Access to the public `Qwen/Qwen3-ASR-1.7B` checkpoint.
- The optional vLLM audio dependencies.

Install the audio dependencies if your environment does not include them:

```bash
pip install "vllm[audio]"
```

## Step 1: Prepare the environment

Confirm that the instance can access the Neuron devices:

```bash
neuron-ls
```

You can use the Hugging Face model ID directly. You can also download the
checkpoint before deployment:

```bash
hf download Qwen/Qwen3-ASR-1.7B \
    --local-dir /path/to/Qwen3-ASR-1.7B
```

Use `/path/to/Qwen3-ASR-1.7B` in the following commands if you download the
checkpoint.

## Step 2: Review the deployment configuration

The examples use TP8, batch size 1, and a maximum model length of 1024 tokens.
Change these values to match your request size and concurrency requirements.

The command-line options control the text decoder:

- `--tensor-parallel-size` sets the text decoder tensor parallel degree.
- `--max-model-len` sets the maximum text sequence length.
- `--max-num-seqs` sets the maximum number of concurrent sequences.
- `--max-num-batched-tokens` sets the maximum number of tokens in one batch.

The text decoder and audio encoder use separate Neuron configuration objects:

```json
{
    "neuron_config": {
        "quantization": "bf16",
        "num_batched_tokens_buckets": [512, 1024],
        "num_seqs_buckets": [1],
        "on_device_sampling_config": {
            "all_greedy": true
        }
    },
    "encoder_neuron_config": {
        "num_encoder_input_buckets": [4],
        "encoder_attention_block_size": 128,
        "dp_size": 8
    }
}
```

The transcription endpoint splits audio longer than 30 seconds into chunks of
30 seconds or less. At batch size 1, one 30-second chunk fits the `[4]` encoder
bucket. Increase the bucket when one encoder batch contains multiple chunks.
The direct offline interface does not split audio. Its input must fit the
largest configured bucket. See
{ref}`audio encoder sizing <audio-encoder-sizing>`
for the calculation. See the
[configuration reference](../guides/reference-configuration.md) for all decoder,
sampling, and encoder settings.

:::{note}
The upstream Qwen3-ASR vLLM backend uses greedy decoding by default. The
`all_greedy` setting matches this behavior on Neuron. It ignores nonzero
sampling values from a client.
:::

## Step 3: Run inference

Choose the online or offline path. You do not need to run both paths.

### Deploy an online transcription server

Start the OpenAI-compatible server:

```bash
vllm serve Qwen/Qwen3-ASR-1.7B \
    --served-model-name qwen3-asr-1.7b \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --max-model-len 1024 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 1024 \
    --limit-mm-per-prompt '{"audio": 1}' \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "num_batched_tokens_buckets": [512, 1024],
            "num_seqs_buckets": [1],
            "on_device_sampling_config": {"all_greedy": true}
        },
        "encoder_neuron_config": {
            "num_encoder_input_buckets": [4],
            "encoder_attention_block_size": 128,
            "dp_size": 8
        }
    }'
```

The first server start compiles the configured shapes. Later starts can reuse
the Neuron compile cache.

Open another terminal. Check the server:

```bash
curl --fail http://localhost:8000/health
```

Send a WAV file to the transcription endpoint:

```bash
curl --fail-with-body http://localhost:8000/v1/audio/transcriptions \
    -F "file=@/path/to/speech.wav" \
    -F "model=qwen3-asr-1.7b" \
    -F "language=en" \
    -F "temperature=0" \
    -F "response_format=json"
```

The response contains a `text` field:

```json
{
    "text": "The transcribed speech appears here."
}
```

### Deploy offline transcription

The following script loads one audio file and transcribes it without a server:

```python
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.multimodal.media.audio import load_audio

MODEL = "Qwen/Qwen3-ASR-1.7B"
AUDIO_FILE = "/path/to/speech.wav"

llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    tensor_parallel_size=8,
    max_model_len=1024,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    limit_mm_per_prompt={"audio": 1},
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "num_batched_tokens_buckets": [512, 1024],
            "num_seqs_buckets": [1],
            "on_device_sampling_config": {"all_greedy": True},
        },
        "encoder_neuron_config": {
            "num_encoder_input_buckets": [4],
            "encoder_attention_block_size": 128,
            "dp_size": 8,
        },
    },
)

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
audio = load_audio(AUDIO_FILE, sr=16000)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio"},
            {"type": "text", "text": "Transcribe the audio."},
        ],
    }
]
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

outputs = llm.generate(
    [
        {
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio]},
        }
    ],
    SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0),
)

print(outputs[0].outputs[0].text)
```

The `sr=16000` argument also normalizes integer PCM audio from the PyAV
decoder. vLLM resamples other input rates before inference.

## Confirmation

The online deployment is ready when the health check succeeds. A transcription
request must return a nonempty `text` value.

The offline deployment is ready when the script prints the audio transcript.

## Common issues

### Offline inference rejects a long audio file

The audio needs more blocks than the largest encoder input bucket. Add a larger
value to `num_encoder_input_buckets`. See
{ref}`audio encoder sizing <audio-encoder-sizing>`
for sizing instructions.

### The server does not use the requested temperature

The tutorial config sets `all_greedy` to `true`. Remove that setting to enable
sampling.

### Compilation reaches the timeout

Increase `NEURON_LIBTORCH_COMPILATION_TIMEOUT`. Keep the server process active
while the first compilation completes.

## Next steps

- [Configuration reference](../guides/reference-configuration.md) — Configure
  decoder and encoder buckets.
- [Features guide](../guides/features-guide.md) — Review vLLM Neuron features.
- [Offline serving quickstart](../getting-started/quickstart-offline-serving.md)
  — Review the general offline flow.
- [Online serving quickstart](../getting-started/quickstart-online-serving.md) —
  Review the general server flow.
