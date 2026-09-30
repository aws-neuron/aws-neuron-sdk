# How to set up vLLM Omni Neuron

<!-- meta: description: Set up vLLM Omni Neuron on AWS Trainium with manual
installation or an AWS Neuron Deep Learning Container. -->
<!-- meta: keywords: vLLM Omni, Neuron, setup, manual installation, DLC, Trainium, Wan2.2 -->
<!-- meta: date_updated: 2026-09-23 -->
<!-- Content type: procedural-how-to -->

## Overview

This guide assumes you already have one SSH-accessible `trn2.48xlarge` or
`trn3` instance. Choose **manual installation** on that host or
**DLC** for a container on the same host. Both flows use Neuron SDK 2.32.0,
vLLM Neuron 0.24.0.1.1.0, and vLLM Omni 0.24.0. No additional nodes are required.

Installing this plugin with pip also installs vLLM Neuron, vLLM, vLLM Omni,
PyTorch, the Neuron compiler, and NKI.

## Prerequisites

- One `trn2.48xlarge` or `trn3` instance with SSH access and
  permission to install host packages and run Docker. The recommended Wan2.2
  configuration uses 64 NeuronCores on that instance.
- A host prepared for Neuron SDK 2.32.0. For bare Ubuntu 24.04, complete
  **Step 2: Install drivers and tools** in the
  [Neuron host setup instructions](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/manual.html).
  Manual installation also requires the runtime and collectives libraries from
  that step. The DLC supplies these libraries inside the container. Pip cannot
  install the kernel driver or system libraries.
- **Manual installation:** Python 3.13 with `venv` support on Ubuntu 24.04.
  **DLC flow:** Docker installed and running.
- Git and access to the source repository, Neuron package index, PyPI, and
  Hugging Face. The DLC flow also needs access to Public ECR.
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
git clone -b release-0.24.0.0.1.0 https://github.com/aws-neuron/vllm-omni-neuron.git "$VLLM_OMNI_HOME/plugin"
```

If you already have the release checkout, use it at that path instead of cloning
again.

## Option A: Manual installation

On bare Ubuntu 24.04, install the compiler's host library on the instance:

```bash
sudo apt-get install -y --no-install-recommends libarchive13t64
```

Create and activate a Python 3.13 environment, then install the plugin and its
declared dependencies in one command:

```bash
python3.13 -m venv "$VLLM_OMNI_HOME/venv"
source "$VLLM_OMNI_HOME/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com \
  -e "$VLLM_OMNI_HOME/plugin"
```

In each new shell, run this block and activate the manual venv before inference:

```bash
export VLLM_OMNI_HOME="$HOME/vllm-omni-neuron"
export PATH="/opt/aws/neuron/bin:$PATH"
export LD_LIBRARY_PATH="/opt/aws/neuron/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_NEURON_BACKEND=neuron_native
export VLLM_NEURON_LIBTORCH_NEURONX_LITE=1
export VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND=1
export HF_HOME="$VLLM_OMNI_HOME/huggingface"
export VLLM_CACHE_ROOT="$VLLM_OMNI_HOME/vllm-cache"
export NKI_COMPILE_CACHE_URL="$VLLM_OMNI_HOME/nki-cache"
```

Continue to [Verify the installation](#verify-the-installation).

## Option B: Install with the Neuron DLC

The published vLLM Neuron DLC contains vLLM, the compiler, NKI, and
Neuron runtime libraries. Pull it and start a container, mounting the source,
caches, and output directory created above:

```bash
export IMAGE="public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04"
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
  -e HF_HOME=/workspace/huggingface \
  -e VLLM_CACHE_ROOT=/workspace/vllm-cache \
  -e NKI_COMPILE_CACHE_URL=/workspace/nki-cache \
  -v "$VLLM_OMNI_HOME:/workspace" \
  "$IMAGE" -c 'sleep infinity'
docker exec vllm-omni-neuron python -m pip install \
  --extra-index-url=https://pip.repos.neuron.amazonaws.com -e /workspace/plugin
```

The last command installs the plugin and its Python dependencies in the
container. Public ECR does not require authentication for this image. Use
`sudo docker` if your host requires it. For online serving, add
`-p 127.0.0.1:8091:8091` when creating the container and use SSH port
forwarding as described in the online quickstart.

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
import torch, torch_xla, libtorch_neuronx_lite, vllm, vllm_neuron, vllm_omni
from vllm_omni_neuron import neuron_omni_platform_plugin
from vllm_omni_neuron.platform import NeuronOmniPlatform

assert neuron_omni_platform_plugin() is not None, "Neuron platform not detected"
assert NeuronOmniPlatform.get_device_count() > 0, "No Neuron devices available"
for package in ("vllm-neuron", "vllm-omni", "vllm-omni-neuron"):
    print(package, version(package))
PY
```

`neuron-ls` must list the expected devices and the Python check must print the
installed package versions without an assertion or import error. If a model
requires authentication, run `hf auth login` in the same environment.

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
- **Missing compiler or NKI:** activate the correct environment and reinstall
  the plugin with dependencies. Installing `vllm-neuron` alone does not install
  these packages.
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
