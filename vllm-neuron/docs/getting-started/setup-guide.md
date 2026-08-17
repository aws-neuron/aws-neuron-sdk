# How to set up vLLM Neuron

<!-- meta: description: Install and configure the vLLM Neuron plugin on AWS
Trainium or Inferentia instances so you can serve LLMs with vLLM Neuron. -->
<!-- meta: keywords: vLLM, Neuron, setup, install, vLLM Neuron plugin,
Trainium, Inferentia, DLAMI -->
<!-- meta: date_updated: 2026-07-23 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: NDOC-179 -->

## Overview

This topic discusses how to install and configure the vLLM Neuron plugin on AWS
Trainium or Inferentia so you can serve large language models with vLLM on
Neuron. When you have completed it, you will have a working environment ready for
the online or offline serving quickstarts.

:::{note}
This guide is for the **vLLM Neuron** plugin introduced in Neuron 2.31. If you
are using vLLM with the NxD Inference library, see the
[vLLM + NxD Inference documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/index.html).
For migration guidance from NxD Inference to vLLM Neuron, see
[migrate from NxD Inference to vLLM Neuron](migration-nxdi-to-vllm-neuron.md).
:::

## Prerequisites

- **Instance**: A supported Trainium EC2 instance (for example,
  `trn2.48xlarge`).
- **Neuron SDK version**: Neuron 2.31 or later.
- **Python**: Python 3.11 or later.
- **Hugging Face access (optional)**: An accepted model license and a Hugging
  Face token if you plan to pull gated models such as Llama.

## Instructions

### Option A: Install from source

Clone the repository and install the plugin from source:

```bash
git clone https://github.com/vllm-project/vllm-neuron.git
cd vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
```

This installs the vLLM Neuron plugin along with vLLM. The Neuron framework and
compiler packages (`neuronxcc`, `torch_neuronx`, `nki`) are **not** included and must
be installed separately (e.g., via the Neuron DLAMI in Option B, or by following the
[Neuron installation guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html)).

### Option B: Use the Neuron DLAMI

Launch an instance with the Neuron Multi-Framework Deep Learning AMI. It ships
with the Neuron SDK, PyTorch NeuronX, and a pre-built virtual environment that
includes vLLM Neuron.

Launch an instance with the DLAMI by following the
[Neuron DLAMI setup instructions](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/environments/dlami.html).
After you connect, activate the vLLM virtual environment:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0/bin/activate
```

### Option C: Use a container

Use the [vLLM Inference NeuronX DLC](https://github.com/aws-neuron/deep-learning-containers#vllm-inference-neuronx)
which bundles the SDK and all dependencies.

### Verify the installation

Confirm that vLLM discovers the Neuron platform plugin:

```bash
python -c "import vllm; from vllm.platforms import current_platform; print(current_platform.device_name)"
```

The output should be:

```text
neuron
```

Confirm the Neuron runtime is visible:

```bash
neuron-ls
```

You should see output listing the available NeuronCores on your instance.

### Install dependencies for disaggregated inference

:::{note}
This section is only required if you plan to run
[disaggregated inference (DI)](../guides/features-guide.md#disaggregated-inference),
which requires the NIXL LIBFABRIC transport (`"backends": ["LIBFABRIC"]`) for KV
cache transfer over EFA. If you are not using it, skip ahead to
[Configure environment variables](#configure-environment-variables).
:::

To use the NIXL LIBFABRIC backend for disaggregated inference, the following two
shared libraries must be installed:

- **`libcuda.so.1`** — the CUDA driver library. The plugin links against it for its
  optional GPUDirect path, which is never exercised on Neuron, so a stub is
  sufficient.
- **`libfabric.so.1`** — the Libfabric/EFA library that carries the actual RDMA
  transfer.

The backend ships as a plugin (`libplugin_LIBFABRIC.so`) inside the `nixl` wheel
that NIXL `dlopen`s when you start a server with `"backends": ["LIBFABRIC"]`. If
either library is missing from the loader path, `dlopen` fails and NIXL reports
`createBackend: unsupported backend 'LIBFABRIC'` followed by
`nixlNotFoundError: NIXL_ERR_NOT_FOUND`, and every server worker exits.

Some environments already ship one or both of these libraries. Run the checks
below and provision only the libraries that are actually missing.

#### 1. Check for `libcuda.so.1`

```bash
ldconfig -p | grep libcuda.so
```

If this prints a `libcuda.so.1` line, the CUDA stub is already present — skip to
step 2. If it prints nothing, install the official CUDA driver stub (as root)
and expose it as `libcuda.so.1`:

```bash
# Add the NVIDIA CUDA apt repo (the ubuntu2204 keyring works on Ubuntu 24.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb && rm -f cuda-keyring_1.0-1_all.deb
apt-get update

# Minimal package that ships the CUDA driver stub (the full toolkit is not needed)
apt-get install -y --no-install-recommends cuda-cudart-dev-12-6

# Expose the stub as libcuda.so.1 on the system loader path.
# /usr/local/lib is already on the default ldconfig search path, so the
# symlink plus ldconfig is enough — no ld.so.conf.d entry is needed.
ln -sf /usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs/libcuda.so /usr/local/lib/libcuda.so.1
ldconfig
```

If you cannot install packages (for example, a read-only or network-isolated
container), put the directory containing a `libcuda.so.1` stub on
`LD_LIBRARY_PATH` in the same shell that runs `vllm serve` instead of using
`ldconfig`:

```bash
export LD_LIBRARY_PATH=/path/to/dir/with/libcuda.so.1:$LD_LIBRARY_PATH
```

:::{note}
A `CUDA driver is a stub library (Error 34)` warning during server startup is
**expected and harmless** — nothing on the Neuron/EFA path calls into the CUDA
driver.
:::

#### 2. Check for `libfabric.so.1` (EFA)

```bash
ldconfig -p | grep libfabric.so
fi_info -p efa   # should list at least one EFA provider
```

If `libfabric.so.1` resolves and `fi_info -p efa` lists a provider, EFA is ready —
skip to the verification step. If `libfabric.so.1` is not found, install the
[AWS EFA installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html)
(userspace libraries only — the host supplies the kernel driver):

```bash
cd /tmp
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
tar -xf aws-efa-installer-latest.tar.gz
cd aws-efa-installer
# --skip-kmod installs userspace libraries only (the host supplies the EFA
# kernel driver); --no-verify skips the installer's kernel-module verification
# step, which otherwise reports a false failure inside a container even though
# --skip-kmod is set.
bash efa_installer.sh --yes --skip-kmod --no-verify
```

This installs Libfabric under `/opt/amazon/efa`. Add its library directories to
`LD_LIBRARY_PATH` in the same shell that runs `vllm serve` so the loader can find
`libfabric.so.1`:

```bash
export LD_LIBRARY_PATH=/opt/amazon/efa/lib64:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export PATH=/opt/amazon/efa/bin:$PATH   # for fi_info and other EFA tools
```

If Libfabric is installed but simply not on the loader path (for example,
`fi_info -p efa` works but `ldconfig -p | grep libfabric.so` is empty), setting
`LD_LIBRARY_PATH` as above is all that is needed.

#### 3. Verify the LIBFABRIC backend loads

Before launching the server, confirm the backend instantiates end-to-end:

```bash
python -c "from nixl._api import nixl_agent, nixl_agent_config; \
nixl_agent('probe', nixl_agent_config(backends=['LIBFABRIC'])); print('LIBFABRIC OK')"
```

A successful run prints `NIXL INFO Backend LIBFABRIC was instantiated` followed by
`LIBFABRIC OK`. If you still see `unsupported backend 'LIBFABRIC'`, run
`ldd` on the plugin to see which library is still unresolved:

```bash
ldd "$(find "$(python -c 'import site; print(site.getsitepackages()[0])')" \
    -name libplugin_LIBFABRIC.so | head -1)" | grep 'not found'
```

Resolve any `not found` entry by adding its directory to `LD_LIBRARY_PATH`.
Entries such as `libcudart.so.13` are used only by the GPUDirect path and can be
left unresolved on Neuron.

### Configure environment variables

The following environment variables control common behaviors:

| Variable | Purpose | Default |
| --- | --- | --- |
| `VLLM_CACHE_ROOT` | Root directory for the Neuron compile cache. Set to local NVMe for best performance. | `~/.cache/vllm` |
| `VLLM_NEURON_LOG_LEVEL` | Plugin log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `INFO` |
| `HF_TOKEN` | Hugging Face access token for gated model downloads. | (none) |

```bash
export VLLM_CACHE_ROOT=/local/cache/vllm
export HF_TOKEN=hf_your_token_here
```

## Confirm your work

Run a minimal `vllm serve` command to confirm the full stack is wired up. This
example uses TinyLlama (a small open model) to keep compilation time short:

```bash
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tensor-parallel-size 2 \
    --max-num-seqs 1 \
    --max-model-len 128 \
    --max-num-batched-tokens 128 \
    --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [128], "num_seqs_buckets": [1]}}'
```

Wait for the server to print `Uvicorn running on ...`. Then, in a second terminal
on the same instance:

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "messages": [{"role": "user", "content": "Hello, Neuron!"}],
      "max_tokens": 16
    }'
```

A JSON response containing generated text confirms that the plugin, Neuron
runtime, and model compilation are all working. Stop the server with `Ctrl+C`.

## Next steps

- [Quickstart: Online serving](quickstart-online-serving.md) — Launch an
  OpenAI-compatible API server.
- [Quickstart: Offline serving](quickstart-offline-serving.md) — Run batch
  inference with the Python API.
- [Features guide](../guides/features-guide.md) — Configure features like
  prefix caching, speculative decoding, data parallelism, and quantization.
- [Configuration options](../guides/reference-configuration.md) — Full
  parameter reference.

## Common issues

### Plugin fails to load the Neuron platform

- **Possible solution**: Confirm you activated the correct virtual environment and
  that `pip show vllm-neuron` reports version 0.21.0.1.0.0 or later. If you installed
  vLLM separately before the plugin, reinstall the plugin to ensure version
  alignment.

### `neuron-ls` shows no devices

- **Possible solution**: Verify you are on a Neuron-supported instance type
  (`trn2` or `trn3`). On a fresh instance, the Neuron runtime may need a
  reboot after driver installation. Check `dmesg | grep neuron` for driver
  messages.

### Compilation times out or is very slow on first run

- **Possible solution**: First-time model compilation on Neuron can take several
  minutes depending on model size. Subsequent runs use the compile cache. Ensure
  `VLLM_CACHE_ROOT` points to fast local storage (NVMe) rather than a network
  filesystem.

### Python version mismatch

- **Possible solution**: The vLLM Neuron plugin requires Python 3.11 or later. Run
  `python --version` to check. The DLAMI virtual environment ships with a
  compatible Python version.

## Related information

- [Quickstart: Online serving](quickstart-online-serving.md) — Launch an
  OpenAI-compatible API server with vLLM Neuron.
- [Quickstart: Offline serving](quickstart-offline-serving.md) — Run offline batch
  inference with vLLM Neuron.
- [Features guide](../guides/features-guide.md) — Detailed coverage of vLLM on
  Neuron features and configuration.
- [Neuron DLAMI options and setup](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/environments/dlami.html)
- [Neuron Deep Learning Containers](https://github.com/aws-neuron/deep-learning-containers#vllm-inference-neuronx)
  for Docker-based workflows.
