# Tutorials

End-to-end guided walkthroughs for specific deployment scenarios and performance optimization.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Deploy Llama 3.3 70B (FP8)
:link: tutorial-llama3-70b
:link-type: doc

Deploy Llama 3.3 70B in static FP8 on a single Trn2/Trn3 instance.
:::

:::{grid-item-card} EAGLE3 speculative decoding (Llama 3.1)
:link: tutorial-eagle3-speculative-decoding-llama-3-1
:link-type: doc

Run Llama 3.1 8B with an EAGLE3 draft model for higher throughput.
:::

:::{grid-item-card} EAGLE3 speculative decoding (GPT-OSS)
:link: tutorial-eagle3-speculative-decoding-gpt-oss
:link-type: doc

Run GPT-OSS-120B with an EAGLE3 draft model for higher throughput.
:::

:::{grid-item-card} DFlash speculative decoding
:link: tutorial-dflash-speculative-decoding
:link-type: doc

Run Llama 3.1 8B with a block-parallel DFlash draft model.
:::

:::{grid-item-card} Disaggregated inference: 1P1D and xPyD
:link: tutorial-di-1p1d-xpyd
:link-type: doc

Configure disaggregated inference topologies.
:::

:::{grid-item-card} Deploy gpt-oss
:link: tutorial-gpt-oss
:link-type: doc

Deploy gpt-oss 20B and 120B, single-instance or disaggregated.
:::

:::{grid-item-card} High-performance GPT-OSS 120B 8K/1K deployment
:link: tutorial-gpt-oss-120b-8k1k-trn3
:link-type: doc

Serve GPT-OSS 120B at 8K/1K on two Trn3 instances (6P1D), with all decoder layers of a decode step fused into one megakernel, and run the benchmark.
:::

:::{grid-item-card} Prefix caching benchmark
:link: tutorial-prefix-caching-gpt-oss-benchmarking
:link-type: doc

Measure TTFT improvement from prefix caching with GPT-OSS.
:::

:::{grid-item-card} Deploy dense Qwen3-32B
:link: tutorial-qwen3-32b
:link-type: doc

Prepare BF16 and MXFP8 checkpoints and validate serving on Trn2/Trn3.
:::

:::{grid-item-card} Deploy Qwen3-VL-32B
:link: tutorial-qwen3-vl-32b
:link-type: doc

Serve the multimodal Qwen3-VL-32B model (BF16 or MXFP8).
:::

:::{grid-item-card} Disaggregated encoder: 1E1PD and xEyPD
:link: tutorial-epd-1e-1pd-xeypd
:link-type: doc

Configure encoder-disaggregated (EPD) multimodal topologies.
:::

:::{grid-item-card} Deploy Qwen3-Embedding-8B
:link: tutorial-qwen3-embedding-8b
:link-type: doc

Serve embeddings via `/v1/embeddings` with a pooling model.
:::

:::{grid-item-card} Deploy Qwen3-ASR-1.7B
:link: tutorial-qwen3-asr-1-7b
:link-type: doc

Transcribe audio through online and offline interfaces.
:::

::::

:::{toctree}
:maxdepth: 1
:hidden:

Deploying Llama 3.3 70B <tutorial-llama3-70b>
EAGLE3 speculative decoding (Llama 3.1) <tutorial-eagle3-speculative-decoding-llama-3-1>
EAGLE3 speculative decoding (GPT-OSS) <tutorial-eagle3-speculative-decoding-gpt-oss>
DFlash speculative decoding <tutorial-dflash-speculative-decoding>
Disaggregated inference (1P1D and xPyD) <tutorial-di-1p1d-xpyd>
Deploying gpt-oss <tutorial-gpt-oss>
High-performance GPT-OSS 120B 8K/1K deployment <tutorial-gpt-oss-120b-8k1k-trn3>
Benchmarking prefix caching (GPT-OSS) <tutorial-prefix-caching-gpt-oss-benchmarking>
Deploying dense Qwen3-32B <tutorial-qwen3-32b>
Deploying Qwen3-VL-32B <tutorial-qwen3-vl-32b>
Disaggregated encoder (1E1PD and xEyPD) <tutorial-epd-1e-1pd-xeypd>
Deploying Qwen3-Embedding-8B <tutorial-qwen3-embedding-8b>
Deploying Qwen3-ASR-1.7B <tutorial-qwen3-asr-1-7b>
:::
