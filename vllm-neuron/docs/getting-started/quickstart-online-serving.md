# Quickstart: Launch an OpenAI-compatible API server with vLLM Neuron

<!-- meta: description: Launch an OpenAI-compatible API server with vLLM on
Neuron and serve your first LLM request on AWS Trainium or Inferentia. -->
<!-- meta: keywords: vLLM, Neuron, online serving, OpenAI API, LLM serving,
Trainium, Inferentia, quickstart -->
<!-- meta: date_updated: 2026-05-28 -->
<!-- Content type: procedural-quickstart -->
<!-- Jira: NDOC-182 -->

This quickstart guides you through launching an OpenAI-compatible API server
with vLLM Neuron and validating it with your first request.

## Prerequisites

- Completed the [setup guide](setup-guide.md) (vLLM Neuron installed and verified).
- Permission to pull the model you plan to serve (for example, an accepted
  Hugging Face license).
- Familiarity with basic shell and HTTP clients such as curl or the OpenAI
  Python SDK.

## Step 1: Launch the API server

In this step, you start an OpenAI-compatible endpoint serving GPT-OSS 20B.

First, set compilation and execution timeouts. GPT-OSS compilation can exceed
the default 600-second budget:

```bash
export VLLM_NEURON_COMPILATION_TIMEOUT=1200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
```

Then launch the server:

```bash
vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 8 \
  --max-model-len 4096 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 4 \
  --port 8000 \
  --hf-overrides '{"quantization_config": {}}' \
  --additional-config '{
    "neuron_config": {
      "on_device_sampling_config": {"all_greedy": "true"}
    }
  }'
```

What each argument does:

- `--tensor-parallel-size 8` — shards the model across 8 NeuronCores (fits a
  20B model comfortably on Trn2).
- `--max-model-len 4096` — maximum sequence length (prompt + generation) the
  server will accept.
- `--max-num-seqs 4` — maximum concurrent requests batched together.
- `--additional-config` — passes Neuron-specific configs; here enabling greedy
  on-device sampling.

The first launch compiles the model for Neuron, which takes several minutes.
Once ready you will see:

```text
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 2: Verify the endpoint with `curl`

In this step, you confirm the server is healthy and responding to chat
completion requests.

First, check the health endpoint:

```bash
curl http://localhost:8000/health
```

You should receive an empty `200 OK` response. Next, send a chat completion
request:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "openai/gpt-oss-20b",
        "messages": [
          {"role": "user", "content": "What is the capital of France? Reply in one sentence."}
        ],
        "max_tokens": 30,
        "temperature": 0
      }' | python -m json.tool
```

Expected output (generation text will vary):

```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1715000000,
    "model": "openai/gpt-oss-20b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 18,
        "completion_tokens": 7,
        "total_tokens": 25
    }
}
```

If you see a JSON response with a `choices` array containing generated text, the
server is working correctly.

## Step 3: Call the API with the OpenAI Python SDK

In this step, you confirm that standard OpenAI client libraries work unchanged
against the Neuron endpoint.

Install the SDK if you haven't already:

```bash
pip install openai
```

Then run the following script:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "Explain what Neuron cores are in two sentences."}
    ],
    max_tokens=60,
    temperature=0,
)

print(response.choices[0].message.content)
```

Expected output:

```text
Neuron cores are custom-designed machine learning accelerators built by AWS,
optimized for high-throughput inference and training workloads. They are the
processing units inside AWS Trainium chips.
```

If the script prints a coherent completion, you have confirmed end-to-end OpenAI
SDK compatibility.

## Confirmation

You have launched a vLLM server on Neuron, validated it over HTTP with `curl`,
and called it from the OpenAI Python SDK. Any application or framework that
targets the OpenAI chat completions API can now point to
`http://<your-instance-ip>:8000/v1` and use Neuron-accelerated inference without
code changes.

## Common issues

- **Server fails to start:** Verify the Neuron SDK and vLLM Neuron plugin
  versions match. See [setup guide](setup-guide.md).
- **Model pull fails:** Confirm your Hugging Face token is set and you have
  access to the model.
- **Out-of-memory at startup:** Reduce `--max-num-seqs` or `--max-model-len`, or
  use a smaller model for the quickstart.

## Clean up

Stop the server with `Ctrl+C` in the terminal where it's running.

## Next steps

- [Quickstart: Offline serving](quickstart-offline-serving.md) — Offline batch
  inference with vLLM Neuron.
- [Features guide](../guides/features-guide.md) — Detailed feature coverage.
