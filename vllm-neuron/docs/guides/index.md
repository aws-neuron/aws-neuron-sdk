# Deploy & serve

Configure, tune, and operate vLLM Neuron for production workloads — features, profiling, and configuration reference.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Features guide
:link: features-guide
:link-type: doc

Configure and tune all serving features — bucketing, quantization, DI, speculation, and more.
:::

:::{grid-item-card} Configuration reference
:link: reference-configuration
:link-type: doc

All Neuron-specific options in `additional_config` and environment variables.
:::

:::{grid-item-card} Profiling workloads
:link: how-to-profile-workloads
:link-type: doc

Capture Neuron Runtime profiles via built-in profiler endpoints.
:::

:::{grid-item-card} Compilation route
:link: how-to-choose-compilation-route
:link-type: doc

Keep the default native route, go back to the XLA route, and confirm which route runs.
:::

:::{grid-item-card} Precompile cache
:link: how-to-precompile-cache
:link-type: doc

Understand the Precompile Cache, confirm a warm hit, and know what invalidates it.
:::

:::{grid-item-card} Model recipes
:link: ../model-recipes/index
:link-type: doc

Supported models and their feature tables.
:::

::::

:::{toctree}
:maxdepth: 1
:hidden:

Features guide <features-guide>
Configuration reference <reference-configuration>
Profiling workloads <how-to-profile-workloads>
Compilation route <how-to-choose-compilation-route>
Precompile cache <how-to-precompile-cache>
Feature–model compatibility <reference-feature-model-compatibility>
:::
