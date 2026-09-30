# vLLM Omni Neuron (Beta)

vLLM Omni Neuron is a Neuron hardware plugin for [vLLM Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
that runs diffuser-type multimodal generation models — such as video and image
generation — on AWS Trainium. Diffuser models are not supported by upstream vLLM,
so they are not served by vLLM Neuron, the Neuron LLM plugin. They live in vLLM
Omni, an extended library under the vLLM project for multimodal generation. vLLM
Omni Neuron adds the Neuron backend that library needs.

The plugin provides Neuron-optimized reference implementations — including NKI
kernels, custom compilation, and hardware-aware scheduling — for models whose
upstream implementations live in vLLM Omni. It uses native PyTorch
(`torch.compile`) through the `libtorch_neuronx_lite` package, the same native
PyTorch infrastructure used by vLLM Neuron.

```{note}
vLLM Omni Neuron is in Beta and under active development. It is supported on
Trainium 2 (trn2) and Trainium 3 (trn3) instances only.
```

The source code for the vLLM Omni Neuron plugin is hosted in the
[vLLM Omni Neuron GitHub repository](https://github.com/aws-neuron/vllm-omni-neuron).

---

## Get started

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Get started
:link: getting-started/index
:link-type: doc

Choose manual installation or a Neuron DLC, then run the online and offline serving quickstarts.
:::

:::{grid-item-card} Guides
:link: guides/index
:link-type: doc

Features guide and offline video generation optimization on Trainium.
:::

:::{grid-item-card} Model recipes
:link: models/index
:link-type: doc

Production-ready deployment recipes for supported models on Trainium.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

End-to-end walkthrough for deploying Wan2.2-A14B on Trainium.
:::

:::{grid-item-card} Model development
:link: model-dev/index
:link-type: doc

Model onboarding, accuracy evaluation and debugging, and Neuron kernel implementation references.
:::

:::{grid-item-card} Concepts & architecture
:link: design/index
:link-type: doc

Engine and model integration, and context parallelism for diffusion inference.
:::

::::
