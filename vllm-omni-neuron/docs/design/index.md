# Concepts & architecture

How vLLM Omni Neuron works under the hood — engine and model integration, and context parallelism for diffusion inference on Trainium.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} vLLM Omni Neuron overview
:link: vllm_omni_neuron_overview
:link-type: doc

Engine, worker, and model integration — how the plugin attaches to vLLM Omni.
:::

:::{grid-item-card} Context parallelism
:link: context_parallelism
:link-type: doc

Sequence-parallel context handling for diffusion inference on Neuron.
:::

::::

:::{toctree}
:maxdepth: 1
:hidden:

vLLM Omni Neuron overview <vllm_omni_neuron_overview>
Context parallelism <context_parallelism>
:::
