# Model development

Onboard new models, evaluate and debug accuracy, and study the Neuron-optimized kernels used in diffusion model development on AWS Trainium.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Onboard a new model
:link: onboarding-models
:link-type: doc

Implement the Neuron pipeline and model components, register them, compile, and validate.
:::

:::{grid-item-card} Accuracy evaluation and debugging
:link: accuracy-evaluation-debugging
:link-type: doc

Evaluate output quality, detect regressions, and debug numerical accuracy.
:::

:::{grid-item-card} Kernel implementations
:link: kernels/index
:link-type: doc

Per-kernel implementation references for the Neuron-optimized kernels used by vLLM Omni Neuron.
:::

:::{grid-item-card} Optimizing offline video generation
:link: optimizing-offline-video-generation
:link-type: doc

Reason about the three-stage cost model and choose a parallelism scheme to maximize quality per compute.
:::

::::

:::{toctree}
:maxdepth: 1
:hidden:

Onboarding a model <onboarding-models>
Accuracy evaluation and debugging <accuracy-evaluation-debugging>
Kernel implementations <kernels/index>
Optimizing offline video generation <optimizing-offline-video-generation>
:::
