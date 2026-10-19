# How vLLM is integrated with TorchNeuron

<!-- meta: description: How vLLM Neuron uses the TorchNeuron torch.compile lowering to serve models on AWS Trainium, and when to use vLLM Neuron or torch.compile directly. -->
<!-- meta: date_updated: 2026-10-13 -->

This topic explains how vLLM Neuron uses TorchNeuron to serve models on AWS
Trainium. It also helps you choose between vLLM Neuron and a direct
`torch.compile` workflow for inference.

vLLM Neuron is the Neuron plugin for vLLM. It compiles your model through the
same native `torch.compile` lowering that TorchNeuron uses. You do not call
`torch.compile` yourself. vLLM calls it for you when the server starts.

## Applies to

This concept is applicable to:

- Inference serving with vLLM on Trn2 and Trn3 instances
- Migration of a serving workload from the PyTorch/XLA route to the native route

## How vLLM Neuron uses TorchNeuron

vLLM Neuron prepares each model graph in the same way that a TorchNeuron
`torch.compile` workflow does:

1. vLLM captures the model as an FX graph with TorchDynamo.
2. The native backend lowers the graph through AOTAutograd, with the Neuron
   decompositions, and then through torch-mlir to StableHLO.
3. The Neuron Compiler (`neuronx-cc`) compiles the StableHLO into a NEFF for
   each bucket.
4. The Neuron Runtime runs the NEFFs while the server handles requests.

vLLM Neuron does not use the `torch-neuronx` package that you install. vLLM
Neuron compiles through the `libtorch-neuronx-lite` package, which includes its
own copy of the TorchNeuron lowering. Because of this, the version of
`torch-neuronx` in your environment does not change how vLLM Neuron compiles
your model.

The native backend of vLLM Neuron is named `neuron_native`, the default value
of `VLLM_NEURON_BACKEND` (`neuron` is accepted as an alias). This name selects
the compilation route inside `libtorch-neuronx-lite`; it is not the PyTorch
`neuron` device that TorchNeuron registers. You can install `torch-neuronx` in
the same environment. In that case, set `TORCH_DEVICE_BACKEND_AUTOLOAD=0`
before you start the vLLM Neuron server:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

`torch-neuronx` loads automatically when Python imports `torch`, and only one
package can own the `neuron` device in a process. Without the variable, the
vLLM Neuron server stops at startup with the error `Tried to register multiple
backend fallbacks for the same dispatch key PrivateUse1`. PyTorch reads the
variable only while it imports `torch`, so set it in the environment, not in
Python code. Processes that use `torch-neuronx` directly do not need the
variable.

## The native route and the XLA route

vLLM Neuron has two compilation routes:

- **Native route (default).** The route that this topic describes. It uses the
  TorchNeuron lowering.
- **XLA route (deprecated).** The earlier route. It lowers the graph through
  PyTorch/XLA to HLO.

Both routes start from the same FX graph and run on the same Neuron Runtime.
Your model code and your serving API stay the same. To use the XLA route, set
`VLLM_NEURON_BACKEND=vllm_neuron` before the server starts.

The native route keeps the vLLM default compiler optimization level, O2. The
XLA route lowers it to O1. To get the same level on each route, set
`--optimization-level` explicitly.

## When to use vLLM Neuron or torch.compile

Use vLLM Neuron when you serve a model to many users. vLLM Neuron gives you:

- An OpenAI-compatible API server and an offline batch API
- Continuous batching, a paged KV cache, and prefix caching
- Tensor, data, and expert parallelism across NeuronCores
- Bucketing, so that each request shape uses a compiled NEFF

Use TorchNeuron with `torch.compile` directly when you write your own
inference loop. For example, use it for a model that vLLM does not support, for
a model that is not a language model, or for research code that calls the model
from your own script. To serve a diffusion or multimodal generation model, first
check [vLLM Omni Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-omni-neuron/docs/index.html), which serves these
models.

A model that runs with TorchNeuron `torch.compile` goes through the same kind
of lowering in vLLM Neuron. vLLM Neuron uses its own copy of the lowering, so
the two versions can be different.

## Related information

- [Inference on Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/inference.html) — The inference options on Neuron.
- [Native PyTorch inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/inference-native-pytorch.html) — Running inference with native PyTorch on Trainium.
- [vLLM Neuron](../../index.md) — The vLLM Neuron documentation.
- [torch.compile on Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/compile/index.html) — Using `torch.compile` on Neuron.
- [Migrating inference from torch-xla to torch.compile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/torch/compile/xla-to-torch-compile-difference.html) — Migrating inference
  from torch-xla to `torch.compile`.
- [vLLM Omni Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-omni-neuron/docs/index.html) — Multimodal (omni)
  inference serving on Neuron.
- [Features guide](../../guides/features-guide.md) — The vLLM Neuron features
  guide, including compilation.
