# Quickstart: Online video generation with Wan2.2 on Neuron

<!-- meta: description: Serve the Wan2.2-T2V-A14B model with the vLLM Omni
OpenAI-compatible video API on AWS Trainium using the vLLM Omni Neuron plugin.
Covers a quick synchronous smoke test and a full asynchronous generation. -->
<!-- meta: keywords: vLLM Omni, vLLM Omni Neuron plugin, Wan2.2,
Wan2.2-T2V-A14B, text-to-video, video generation, online serving, OpenAI API,
AWS Trainium, trn2, NeuronCores, quickstart -->
<!-- meta: date_updated: 2026-09-14 -->
<!-- meta: content_type: procedural-quickstart -->
<!-- Jira: NMI-413 -->

This quickstart shows you how to serve
[Wan2.2-T2V-A14B](../models/wan22-t2v-14b.md) on a `trn2.48xlarge` instance with
the vLLM Omni Neuron plugin. When you finish, you can submit text-to-video jobs,
check their status, and download the generated MP4 files through the vLLM Omni
video API. The request flow follows the
[upstream vLLM Omni Videos API](https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/videos_api.md).
This guide adds the Neuron-specific container and server configuration.

## Prerequisites

Before you start, make sure that you have the following:

- One SSH-accessible `trn2.48xlarge` or `trn3` instance.
- The environment prepared in the [setup guide](setup-guide.md). The commands
  below use the DLC flow and its `vllm-omni-neuron` container.
- For the DLC flow, port `8091` published on the instance loopback interface.
  Add `-p 127.0.0.1:8091:8091` to the setup guide's `docker run` command.
- Network access to download the Wan2.2 weights
  (`Wan-AI/Wan2.2-T2V-A14B-Diffusers`) from Hugging Face.
- `curl` and `jq` installed on the host.

The server uses the vLLM Omni entrypoint and reads its Neuron parallelism
configuration from
`/workspace/plugin/examples/wan22/wan22_stage.yaml`.

## Step 1: Start the server

In an SSH session on the instance host, run the server in the foreground inside
the `vllm-omni-neuron` container:

```bash
sudo docker exec -it \
  -e TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1 \
  -e VLLM_SLEEP_WHEN_IDLE=1 \
  -e NEURON_LOGICAL_NC_CONFIG=2 \
  -e VLLM_NEURON_COMPILATION_TIMEOUT=1800 \
  -e NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=167772160 \
  -e NEURON_SCRATCHPAD_PAGE_SIZE=2048 \
  -e NEURON_RT_DBG_CC_DMA_PACKET_SIZE=2048 \
  -e "NEURON_CC_FLAGS=-O1 --hbm-scratchpad-page-size=2048" \
  -e OMP_NUM_THREADS=24 \
  -e MKL_NUM_THREADS=24 \
  vllm-omni-neuron \
  vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --omni \
    --host 0.0.0.0 \
    --port 8091 \
    --stage-configs-path /workspace/plugin/examples/wan22/wan22_stage.yaml \
    --stage-init-timeout 3600 \
    --init-timeout 3600
```

For 720P serving, replace `--stage-configs-path` with
`/workspace/plugin/examples/wan22/wan22_stage_tp4cp8cfg2_720p.yaml`
([720P stage config](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage_tp4cp8cfg2_720p.yaml))
and request `width=1280` and `height=720`. This config enables VAE tiling and
32-way VAE patch parallelism; the 480P examples below use `wan22_stage.yaml`.

For manual installation, run `vllm serve` directly in the activated environment.
Set the Docker `-e` variables with shell `export` commands, omit the Docker wrapper,
replace `/workspace` with `$VLLM_OMNI_HOME`, and use `--host 127.0.0.1` instead
of `--host 0.0.0.0` so the unauthenticated API remains on instance loopback.

Keep this SSH session open. The server is ready when the log reports that
application startup is complete. Open a second SSH session to the same instance
and run the client commands below there; `localhost` refers to the instance.

```bash
curl --fail http://localhost:8091/health
```

To run the client commands on your local machine instead, forward the port in
a separate local terminal, replacing the example username and address:

```bash
ssh -N -L 8091:127.0.0.1:8091 ubuntu@INSTANCE_ADDRESS
```

Use the same SSH key/options as your normal connection. Keep the tunnel open;
the existing `http://localhost:8091` URLs then reach the remote API. No public
inbound rule for port 8091 is needed.

> **Note:** Initial startup loads approximately 116 GB of model weights. The
> first request for a new video shape also compiles Neuron graphs.

## Step 2: Run a quick smoke test

Use the synchronous endpoint to verify request handling, Neuron execution, and
MP4 encoding in one command:

```bash
curl --fail-with-body --show-error \
  -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=A red balloon floating over green hills" \
  -F "width=208" \
  -F "height=128" \
  -F "num_frames=5" \
  -F "fps=8" \
  -F "num_inference_steps=2" \
  -F "guidance_scale=1.0" \
  -F "guidance_scale_2=1.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=5.0" \
  -F "seed=42" \
  --output wan22-smoke.mp4
```

The command returns the generated video as `wan22-smoke.mp4`. The synchronous
endpoint blocks until generation and encoding finish.

## Step 3: Submit a full asynchronous generation job

The asynchronous API creates a job immediately. The following request generates
81 frames at 832x480 over 40 denoising steps:

```bash
create_response=$(
  curl --fail-with-body --silent --show-error \
    -X POST http://localhost:8091/v1/videos \
    -H "Accept: application/json" \
    -F "prompt=A red hot air balloon rising over green hills at sunrise" \
    -F "width=832" \
    -F "height=480" \
    -F "num_frames=81" \
    -F "fps=16" \
    -F "num_inference_steps=40" \
    -F "guidance_scale=5.0" \
    -F "guidance_scale_2=5.0" \
    -F "boundary_ratio=0.875" \
    -F "flow_shift=5.0" \
    -F "seed=42"
)

video_id=$(printf '%s' "$create_response" | jq -r '.id')
printf 'Created video job %s\n' "$video_id"
```

Poll the job until it completes:

```bash
while true; do
  status_response=$(
    curl --fail-with-body --silent --show-error \
      "http://localhost:8091/v1/videos/${video_id}"
  )
  status=$(printf '%s' "$status_response" | jq -r '.status')
  printf 'Video job %s status: %s\n' "$video_id" "$status"

  case "$status" in
    completed)
      break
      ;;
    failed)
      printf '%s\n' "$status_response" | jq .
      exit 1
      ;;
    queued|in_progress)
      sleep 5
      ;;
    *)
      printf '%s\n' "$status_response" | jq .
      exit 1
      ;;
  esac
done
```

Download the completed video:

```bash
curl --fail-with-body --location \
  "http://localhost:8091/v1/videos/${video_id}/content" \
  --output wan22-online.mp4
```

By default, the asynchronous API stores generated files in `/tmp/storage`
inside the container. To use another local or mounted directory, add
`-e VLLM_OMNI_SERVER_STORAGE__PATH=/path/to/videos` to the `docker exec` command
that starts the server. Delete a job and its stored output with the `DELETE`
endpoint in [Clean up](#clean-up). vLLM Omni 0.24.0 has no S3 storage backend;
upload the downloaded MP4 separately if you need remote storage.

## Common issues

- **The health check cannot connect:** For Docker, confirm the container was started
  with `-p 127.0.0.1:8091:8091` and the server uses `--host 0.0.0.0`. Check that the
  server is running and, for local clients, that the SSH tunnel is still open.
- **Silent crash or `SIGBUS` while returning a video:** The container
  `/dev/shm` is too small. Start the container with `--shm-size=2g`, as shown in
  the setup guide.
- **The first request takes a long time:** The Neuron compiler builds graphs
  for the requested shape on the first run, which can take several minutes.
  An asynchronous job can remain `in_progress` with progress `0` during compilation.
- **The job status is `failed`:** Retrieve the job with
  `GET /v1/videos/{video_id}` and inspect its error field. Check the server
  terminal for the corresponding worker error.
- **Weights download fails:** Confirm network access and Hugging Face permission
  for `Wan-AI/Wan2.2-T2V-A14B-Diffusers`.

## Clean up

Delete the stored job and output:

```bash
curl --fail-with-body --request DELETE \
  "http://localhost:8091/v1/videos/${video_id}" | jq .
```

Press `Ctrl+C` in the server terminal to stop the API server. To stop and remove
the container, run:

```bash
docker stop vllm-omni-neuron
docker rm vllm-omni-neuron
```

## Next steps

- [Optimizing offline video generation](../model-dev/optimizing-offline-video-generation.md) —
  parallelism selection, 720p and VAE tiling, and quality tuning.
- [Wan2.2-T2V-A14B model card](../models/wan22-t2v-14b.md) — architecture, feature
  support, and [VBench](https://arxiv.org/abs/2311.17982) accuracy.
- [Stage config (`wan22_stage.yaml`)](https://github.com/aws-neuron/vllm-omni-neuron/blob/release-0.24.0.0.1.0/examples/wan22/wan22_stage.yaml) —
  the parallelism and model configuration used by the server.
