# Model Recipes

Production-ready deployment recipes for specific models on AWS Trainium and Inferentia. Each recipe includes instance sizing, configuration, and performance guidance.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Deploy Llama 3
:link: llama-3
:link-type: doc

Model recipe for the Llama 3 family (1B, 8B, 70B) on Trn2/Trn3.
:::

:::{grid-item-card} Deploy GPT-OSS
:link: gpt-oss
:link-type: doc

Model recipe for GPT-OSS 20B and 120B (MoE) on Trn2/Trn3.
:::

:::{grid-item-card} Deploy Qwen3-VL 32B
:link: qwen3-vl
:link-type: doc

Model recipe for Qwen3-VL 32B (multimodal) on Trn2/Trn3.
:::

:::{grid-item-card} Deploy Qwen3-Embedding 8B
:link: qwen3-embedding-8b
:link-type: doc

Model recipe for Qwen3-Embedding 8B (pooling / embeddings) on Trn2/Trn3.
:::

::::

:::{toctree}
:maxdepth: 1
:hidden:

Llama 3 <llama-3>
GPT-OSS <gpt-oss>
Qwen3-VL <qwen3-vl>
Qwen3-Embedding <qwen3-embedding-8b>
:::
