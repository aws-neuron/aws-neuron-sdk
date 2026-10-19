# How to set up vLLM Omni Neuron

<!-- meta: description: Set up vLLM Omni Neuron on AWS Trainium with manual
installation or an AWS Neuron Deep Learning Container. -->
<!-- meta: keywords: vLLM Omni, Neuron, setup, manual installation, DLC, Trainium, Wan2.2 -->
<!-- meta: date_updated: 2026-10-13 -->
<!-- Content type: procedural-how-to -->

## Overview

This guide assumes you already have one SSH-accessible `trn2.48xlarge` or
`trn3` instance. Choose **manual installation** on that host or
**DLC** for a container on the same host. This branch targets Neuron SDK 2.33
with vLLM 0.29.0, vLLM Neuron 0.29.0.1.x, and vLLM Omni 0.29.0rc1.
The exact vLLM package requirements are in
[`requirements/core.txt`](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/requirements/core.txt).

Both flows start from a compatible Neuron SDK environment that already contains
the compiler, NKI, `libtorch-neuronx-lite`, PyTorch, vLLM, and vLLM-Neuron.
The commands below install this plugin, Omni, and their remaining dependencies.

Pass the local `stubs/cosmos-guardrail` package on the same pip command as the
plugin. It satisfies Omni's Cosmos requirement without installing the Cosmos
implementation or its dependency chain, which conflicts with the current
compiler's NetworkX requirement. This uses normal pip dependency resolution;
neither uv nor a modified Omni wheel is required. Neuron Wan does not use
Cosmos3 guardrails, and this installation does not provide them.

## Prerequisites

- One `trn2.48xlarge` or `trn3` instance with SSH access and
  permission to install host packages and run Docker. The recommended Wan2.2
  configuration uses 64 NeuronCores on that instance.
- A host driver and runtime compatible with the selected Neuron SDK. A matching
  [Neuron DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/environments/dlami.html)
  provides the driver and tools. For a bare Ubuntu 24.04 host, complete
  **Step 2: Install drivers and tools** in the
  [Neuron host setup instructions](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/manual.html).
  Manual installation also requires the host runtime and collectives libraries from
  that step; the DLC supplies those libraries inside the container. Pip cannot
  install the kernel driver. Do not follow the legacy `torch-neuronx` framework
  installation steps.
- **Manual installation:** an existing Python 3.13 Neuron SDK environment
  containing the packages listed above. **DLC flow:** Docker installed and
  running, and a compatible SDK image.
- Git and access to the source repository, Neuron package index, PyPI, and
  Hugging Face. The DLC flow also needs access to the selected image registry.
- Storage for approximately 116 GB of Wan2.2 weights, plus compilation caches
  and generated videos. Accept any required model license before downloading.

## Connect to the instance

Connect from your local machine using your instance's SSH username, address,
and key, following the [EC2 SSH connection instructions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connect-linux-inst-ssh.html).
Run the commands below in that SSH session on the instance, not on your laptop.
For the DLC flow, only the final verification commands run inside the container.

## Prepare the source and persistent storage

Run on the instance host. Both flows install the plugin from this checkout,
which also provides the examples and stage configurations:

```bash
export VLLM_OMNI_HOME="$HOME/vllm-omni-neuron"
mkdir -p "$VLLM_OMNI_HOME"/{huggingface,vllm-cache,nki-cache,output}
git clone https://github.com/aws-neuron/vllm-omni-neuron.git "$VLLM_OMNI_HOME/plugin"
```

If you already have the release checkout, use it at that path instead of cloning
again. The commands below install the plugin and stub from this source checkout.

## Option A: Manual installation

Activate your prepared Neuron SDK Python environment, then install the plugin
and local stub together:

```bash
python -m pip install --extra-index-url https://pip.repos.neuron.amazonaws.com \
  "$VLLM_OMNI_HOME/plugin" \
  "$VLLM_OMNI_HOME/plugin/stubs/cosmos-guardrail"
```

The local distribution reports `cosmos-guardrail==0.3.1+stub` and contains no
Python modules or runtime dependencies. Pip resolves all other dependencies
from the original package metadata. Include the stub path on the same command
when upgrading or reinstalling; see the
[stub documentation](https://github.com/aws-neuron/vllm-omni-neuron/blob/main/stubs/cosmos-guardrail/README.md).

Start from a clean compatible SDK environment. Installing the stub does not
remove dependencies left behind by an earlier installation of the real Cosmos
package. For development tests, use `"$VLLM_OMNI_HOME/plugin[test]"` in the
command above. Rerun the installation command after editing the plugin source.

In each new shell, activate the same SDK environment and set these variables
before inference:

```bash
export VLLM_OMNI_HOME="$HOME/vllm-omni-neuron"
export PATH="/opt/aws/neuron/bin:$PATH"
export LD_LIBRARY_PATH="/opt/aws/neuron/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_NEURON_BACKEND=neuron_native
export VLLM_NEURON_LIBTORCH_NEURONX_LITE=1
export VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND=1
export TORCH_CACHING_PRECOMPILE=0
export HF_HOME="$VLLM_OMNI_HOME/huggingface"
export VLLM_CACHE_ROOT="$VLLM_OMNI_HOME/vllm-cache"
export NKI_COMPILE_CACHE_URL="$VLLM_OMNI_HOME/nki-cache"
```

Continue to [Verify the installation](#verify-the-installation).

## Option B: Install with the Neuron DLC

Use the `pytorch-inference-vllm-omni-neuronx` Neuron Deep Learning Container
(DLC). It ships vLLM Omni Neuron with the compatible vLLM, compiler, NKI, and
runtime libraries described above. Copy the image URI for your Neuron SDK
version from the
[Neuron DLC image list](https://github.com/aws-neuron/deep-learning-containers#vllm-omni-inference-neuronx)
into `IMAGE` below. Pull it and start a container, mounting the source, caches,
and output directory:

```bash
export IMAGE="public.ecr.aws/neuron/pytorch-inference-vllm-omni-neuronx:<tag-from-image-list>"
docker pull "$IMAGE"
docker run -d --name vllm-omni-neuron \
  --device=/dev/neuron0 --device=/dev/neuron1 \
  --device=/dev/neuron2 --device=/dev/neuron3 \
  --device=/dev/neuron4 --device=/dev/neuron5 \
  --device=/dev/neuron6 --device=/dev/neuron7 \
  --device=/dev/neuron8 --device=/dev/neuron9 \
  --device=/dev/neuron10 --device=/dev/neuron11 \
  --device=/dev/neuron12 --device=/dev/neuron13 \
  --device=/dev/neuron14 --device=/dev/neuron15 \
  --cap-add SYS_ADMIN --cap-add IPC_LOCK --shm-size=2g \
  --entrypoint /bin/bash --workdir /workspace \
  -e VLLM_OMNI_HOME=/workspace \
  -e VLLM_NEURON_BACKEND=neuron_native \
  -e VLLM_NEURON_LIBTORCH_NEURONX_LITE=1 \
  -e VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND=1 \
  -e TORCH_CACHING_PRECOMPILE=0 \
  -e HF_HOME=/workspace/huggingface \
  -e VLLM_CACHE_ROOT=/workspace/vllm-cache \
  -e NKI_COMPILE_CACHE_URL=/workspace/nki-cache \
  -v "$VLLM_OMNI_HOME:/workspace" \
  "$IMAGE" -c 'sleep infinity'
docker exec vllm-omni-neuron python -m pip install \
  --extra-index-url https://pip.repos.neuron.amazonaws.com \
  /workspace/plugin /workspace/plugin/stubs/cosmos-guardrail
```

The last command installs this plugin, vLLM Omni, and the local Cosmos stub,
using the SDK packages already in the image.
Use `sudo docker` if your host requires it. For online serving, add
`-p 127.0.0.1:8091:8091` when creating the container and use SSH port forwarding
for access from your local machine, as described in the online quickstart.

The device mappings and capabilities follow the
[Neuron DLC quickstart](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/environments/quickstart-deploy-dlc.html#quickstart-vllm-dlc-step2).
`--shm-size=2g` avoids Docker's 64 MB shared-memory limit, which is too small
for video output.
Package installation changes this container, not the base image; repeat the
install when recreating it or put that step in a derived Dockerfile.

Enter the container for the verification commands below:

```bash
docker exec -it vllm-omni-neuron bash
```

## Verify the installation

Run the following commands in the activated environment for manual installation
or inside the DLC container:

```bash
neuron-ls
python - <<'PY'
from importlib.metadata import version

import vllm_omni  # Load Omni before importing the Neuron platform module.
from vllm_omni_neuron import neuron_omni_platform_plugin
from vllm_omni_neuron.platform import NeuronOmniPlatform

assert neuron_omni_platform_plugin() is not None, "Neuron platform not detected"
assert NeuronOmniPlatform.get_device_count() > 0, "No Neuron devices available"
for package in ("vllm-neuron", "vllm-omni", "vllm-omni-neuron", "cosmos-guardrail"):
    print(package, version(package))
assert version("cosmos-guardrail") == "0.3.1+stub"
PY
```

`neuron-ls` must list the expected devices and the Python check must print the
installed package versions without an assertion or import error. The stub
satisfies package metadata only; it does not supply Cosmos guardrail
functionality. If a model requires authentication, run `hf auth login` in the
same environment.

Run a short text-to-video generation through the Omni engine:

```bash
python "$VLLM_OMNI_HOME/plugin/examples/wan22/run.py" \
  --dev --output "$VLLM_OMNI_HOME/output/wan22-dev.mp4"
```

The first run downloads weights and compiles graphs. Model weights are stored in
the prepared Hugging Face directory. With either flow, the video is saved at
`$VLLM_OMNI_HOME/output/wan22-dev.mp4`. Copy it to your local machine with `scp`
using the same SSH connection details and the absolute path on the instance.

## Common issues

- **No devices:** check the host driver and `/dev/neuron*`; for Docker, also
  check device exposure. Follow the host setup instructions above rather than
  installing a driver inside the container.
- **Missing compiler or NKI:** activate the prepared SDK environment or use a
  compatible SDK image. The plugin installation assumes those packages exist.
- **Cosmos/NetworkX dependency conflict:** include the local stub path on the same
  pip command as the plugin. If real Cosmos dependencies were installed earlier,
  restart from a clean compatible SDK environment.
- **Imports fail after an upgrade:** recreate the matching environment. Do not
  mix this stack with a standalone `torch-neuronx` installation.
- **Worker exits with `SIGBUS`:** use `--shm-size=2g` for Docker; for manual
  installation, ensure the host has sufficient `/dev/shm` space.

## Next steps

- [Offline quickstart](quickstart-offline-serving-wan22.md) — full-size generation.
- [Online quickstart](quickstart-online-serving-wan22.md) — serve the video API.
- [Wan2.2 model card](../models/wan22-t2v-14b.md) — supported model configurations.
- [Optimizing high-quality offline video generation](../model-dev/optimizing-offline-video-generation.md)
  — select output shapes, quality settings, and parallelism after setup.
