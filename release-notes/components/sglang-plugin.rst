.. meta::
    :description: Complete release notes for the SGLang-Neuron plug-in component across all AWS Neuron SDK versions.
    :keywords: sglang, sglang plug-in, sglang-neuron, neuron sglang, sglang diffusion, release notes, aws neuron sdk, trainium
    :date-modified: 2026-11-11

.. _sglang_rn:

Component Release Notes for the SGLang-Neuron plug-in
=====================================================

**Latest version**: 0.1.0 (Public Preview)

The release notes for SGLang-Neuron. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _sglang-public-preview-release:

SGLang-Neuron (11/12/2026 Public Preview Release)
-------------------------------------------------

SGLang-Neuron 0.1.0 (``sglang-neuron``) is the first release of the plug-in. It brings the SGLang serving stack to AWS Trainium as a native PyTorch plug-in. In this public preview, large language models (LLMs) run on Trn3 instances on one fully asynchronous runtime, and multimodal diffusion models run on Trn2 instances through SGLang Diffusion. To run the plug-in, use the :ref:`SGLang-Neuron container image <sglang-0-1-0-container-image>`.


.. note::

   This release is a public preview. Features, APIs, and supported models can change in later releases. SGLang-Neuron is released by RadixArk in collaboration with the AWS Neuron team. SGLang-Neuron enables developers to serve open models and generate reinforcement learning (RL) rollouts with SGLang on AWS Trainium, using the same SGLang interface they use today.

Highlights
^^^^^^^^^^

* SGLang loads the plug-in through a single plug-in entry point. The plug-in runs on TorchNeuron (``torch-neuronx``).
* All supported LLMs use one fully asynchronous execution path with prefill-decode disaggregation.
* GPT-OSS-120B decodes with a giga kernel on Trainium3. The giga kernel runs every layer of a decode step in one kernel call.
* Qwen3.5-397B-A17B runs on a whole-model megakernel on Trainium3, in aggregated and disaggregated serving.
* GLM-5.2 (MXFP4) serves a 1M-token context on Trainium3, with prefill-decode disaggregation and multi-token prediction (MTP) speculative decoding.
* Kimi-K3 serves a 1M-token context on Trainium3, with prefill-decode disaggregation and DSPARK speculative decoding.
* HiCache L2 keeps prefixes in host memory, so the server can reuse them across requests. A remote DRAM tier below HiCache L2 extends the cache over NIXL (libfabric).
* You can serve multimodal diffusion models on Trn2. FLUX.1-dev, FLUX.3-Action, MiniMax-H3, Wan2.2 A14B, and LingBot-World-V2 run on a new Neuron platform for SGLang Diffusion, with NKI kernels. For each model, see :ref:`sglang-0-1-0-multimodal-highlights`.

.. _sglang-0-1-0-multimodal-highlights:

Multimodal model support and optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The multimodal models run on Trn2 instances through SGLang Diffusion on the Neuron platform:

* **FLUX.1-dev** generates 1024 x 1024 and 2048 x 2048 images from a text prompt. It uses fused K/V all-gather attention and const-max ring attention kernels.
* **Wan2.2 A14B** generates 81-frame videos at 480p, 720p, and 1080p from a text prompt or an input image. It uses classifier-free guidance (CFG) parallelism, a const-max ring attention kernel that keeps K/V resident in SBUF (the on-chip state buffer), and a patch-parallel variational autoencoder (VAE).
* **MiniMax-H3** generates a 124-frame 768p video with its soundtrack, through text-to-video-and-audio (t2va) or first-and-last-frame conditioning (fl2va). It uses a guarded const-max ring attention kernel, an adaptive layer normalization (AdaLN) cache, and a distributed video and audio VAE.
* **FLUX.3-Action** returns the next 32 robot actions from three camera views, the robot state, and an instruction. It serves all three DROID checkpoints (``base``, ``gd``, and ``sd``) over the SGLang OpenPI WebSocket and HTTP action API. It uses const-max ring attention, a video VAE encoder sharded across chips, and UniPC sampling on the device.
* **LingBot-World-V2** streams a world in real time at 480p from one image along a camera path. It uses a flash attention kernel over paged KV history and a streaming VAE that is sharded by space. For interactive control, a session accepts prompt updates and camera actions for each chunk while it runs.

Running SGLang-Neuron with Neuron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this public preview, you run the plug-in from the :ref:`plug-in container image <sglang-0-1-0-container-image>`. The image has SGLang, the plug-in, and the patched NIXL installed.

**Prerequisites**

* For the LLMs, a trn3.48xlarge instance. Kimi-K3 needs a Trn3 UltraServer: at least two trn3.48xlarge instances connected by a switch.
* For the diffusion models, a trn2.48xlarge instance.
* The Neuron driver for Neuron SDK 2.33.0 (``aws-neuronx-dkms`` 2.31.7.0) on the host.
* Docker, and a host directory that holds the model checkpoints.

**Step 1: Pull the image**

.. code-block:: bash

   docker pull public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04

**Step 2: Start the server**

These commands come from the quickstart in the plug-in repository README. They download Qwen3-8B, start a server on four NeuronCores, wait until it is ready, and send a request. Run them in bash, because they rely on bash to split ``$DEVICES`` into separate arguments:

.. code-block:: bash

   IMAGE=public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04
   MODELS=/path/to/models     # holds Qwen3-8B, mounted at /models
   CACHE=/path/to/cache       # the compile cache: the first start compiles, later starts reuse it
   docker pull $IMAGE
   docker run --rm -v $MODELS:/models $IMAGE hf download Qwen/Qwen3-8B --local-dir /models/Qwen3-8B
   DEVICES=$(for d in /dev/neuron*; do printf -- '--device %s ' "$d"; done)

   docker run -d --name qwen3 --init --ipc=host --cap-add SYS_NICE -p 127.0.0.1:30000:30000 $DEVICES \
     -v $MODELS:/models:ro -v $CACHE:/cache \
     -e TORCH_NEURONX_NEFF_CACHE_DIR=/cache/neff -e TORCH_NEURONX_HLO_CACHE_DIR=/cache/hlo \
     -e NKI_COMPILE_CACHE_URL=/cache/nki \
     $IMAGE python3 -m sglang.launch_server --model-path /models/Qwen3-8B --tp-size 4 \
     --context-length 4096 --chunked-prefill-size 2048 --max-running-requests 64 --max-total-tokens 266240 \
     --mem-fraction-static 0.6 --host 0.0.0.0 --port 30000

   # Wait until the server answers /health. The first start on a new cache compiles the model.
   until curl -sf http://127.0.0.1:30000/health; do
     [ "$(docker inspect -f '{{.State.Running}}' qwen3)" = true ] || { echo "exited: docker logs qwen3"; break; }
     sleep 10; done

Replace ``/path/to/models`` and ``/path/to/cache`` with host directories. Do not mount a volume at ``/workspace``, because the image installs ``torch-neuronx`` from that directory.

**Step 3: Send a request**

.. code-block:: bash

   curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
     "model": "/models/Qwen3-8B", "temperature": 0, "max_tokens": 64,
     "chat_template_kwargs": {"enable_thinking": false},
     "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number."}]}'

The answer is ``17 * 23 = 391``. To stop the server, run ``docker stop -t 120 qwen3 && docker rm qwen3``.

By default, the plug-in samples greedily on the device, so each request sets ``temperature`` to 0 or ``top_k`` to 1. To serve sampled requests, set ``SGLANG_NEURON_SAMPLER=temperature`` or ``SGLANG_NEURON_SAMPLER=topk_topp`` when you start the server.

**Quickstart scripts for each model**

The `examples folder <https://github.com/sgl-project/sglang-neuron/tree/main/examples>`__ in the plug-in repository has launch scripts with the validated flags for each LLM except Qwen3-MoE. Each example has the commands to start it in the container image, on one node or on a prefill and decode pair, and a GSM8K script that checks the server's accuracy. For the diffusion models, each `model recipe <https://github.com/sgl-project/sglang-neuron/tree/main/docs/model-recipes>`__ gives the ``sglang generate`` or ``sglang serve`` command. The `container guide <https://github.com/sgl-project/sglang-neuron/blob/main/docs/guides/container.md>`__ explains each ``docker run`` option.

**Disaggregated serving**

A prefill server and a decode server must both run this image, because the KV transfer uses the patched NIXL. Each container also needs the EFA devices (``--device /dev/infiniband/uverbs<N>`` for each device, with ``--cap-add IPC_LOCK --ulimit memlock=-1``) and usually ``--network host``. Clients reach the pair through the load balancer in the image, which needs no Neuron device:

.. code-block:: bash

   docker run --rm --network host \
     public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04 \
     python3 -m sglang_neuron.disaggregation.balancer \
       --prefill http://<prefill host>:30000 --decode http://<decode host>:30000

The balancer listens on ``127.0.0.1:8000``. For the server flags of each model, see the model recipes in the `sglang-neuron repository on GitHub <https://github.com/sgl-project/sglang-neuron>`__.

For the supported models, see the following list.

Supported models
^^^^^^^^^^^^^^^^

In this public preview, the LLMs run on trn3.48xlarge instances, and the diffusion models run on trn2.48xlarge instances. Kimi-K3 needs a Trn3 UltraServer. The LLMs are served text only. In the **Details** column, TP is tensor parallelism, EP is expert parallelism, CP is context parallelism, DP is data parallelism, PP is pipeline parallelism, DCP is decode context parallelism, and CFG is classifier-free guidance parallelism.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model
     - Type
     - Details
   * - Kimi-K3
     - LLM
     - Checkpoint ``moonshotai/Kimi-K3``. Needs a Trn3 UltraServer (at least two trn3.48xlarge instances connected by a switch). Disaggregated across three nodes: one prefill node (PP8 x TP16 x DCP16) and a decode that spans two nodes (8 DP-attention groups of attention TP32, EP256, DCP32) with DSPARK speculative decoding. MXFP4 experts and an FP8 KV cache. Multi-head latent attention (MLA) with Kimi Delta Attention state. The prefill uses a chunk-keyed prefix cache with HiCache, or the ``kimi-prompt`` prefix cache, which keeps the end of each finished prompt for the next turn.
   * - GLM-5.2 (MXFP4)
     - LLM
     - Disaggregated only, one prefill server and one decode server. MLA with DeepSeek sparse attention. The prefill runs 4 pipeline stages at TP32 or 16 stages at TP8, and the decode runs a giga kernel at TP128 with 32 slots of 128K or 1M positions. Asynchronous MTP speculative decoding. Requires the Neuron SDK 2.33.0 package versions.
   * - Qwen3.5-397B-A17B
     - LLM
     - Text only, from the FP8 checkpoint. Aggregated and disaggregated (one prefill server and one decode server). TP128 as eight DP-attention replicas of attention TP16, with EP128. Gated DeltaNet and gated attention layers. A whole-model megakernel with MXFP8 experts. The chunk-keyed prefix cache and greedy sampling.
   * - Qwen3-0.6B and Qwen3-8B
     - LLM
     - Aggregated and disaggregated. TP4, and TP8 for aggregated Qwen3-8B.
   * - Qwen3-30B-A3B
     - LLM
     - Aggregated and disaggregated. TP4 with EP4.
   * - GPT-OSS-120B
     - LLM
     - Aggregated (TP8 x EP8) and disaggregated. MXFP4 experts and an FP8 KV cache. The disaggregated decode uses DP attention or the giga kernel, and two-stage pipeline parallelism is available.
   * - GLM-4-9B
     - LLM
     - Aggregated and disaggregated. TP4. Checkpoint ``zai-org/GLM-4-9B-0414``.
   * - DeepSeek-V2-Lite
     - LLM
     - Aggregated and disaggregated. TP4. MLA over a paged MLA KV cache.
   * - FLUX.1-dev
     - Diffusion
     - trn2.48xlarge. Text to image. TP8 x CP8 on 64 logical NeuronCores.
   * - FLUX.3-Action
     - Diffusion
     - trn2.48xlarge. Robot actions for the DROID checkpoints. ``sd`` and ``gd`` at TP4 or TP4 x CP2, and ``base`` at TP4 x CFG2, on 4 or 8 NeuronCores. BF16.
   * - MiniMax-H3
     - Diffusion
     - trn2.48xlarge. Text and keyframes to video and audio. TP8 x CP8 on 64 logical NeuronCores.
   * - Wan2.2 A14B
     - Diffusion
     - trn2.48xlarge. Text to video and image to video. TP4 x CP8 x CFG2 on 64 logical NeuronCores. BF16.
   * - LingBot-World-V2
     - Diffusion
     - trn2.48xlarge. Real-time image to world. The diffusion transformer runs at TP8 x CP8 on 64 NeuronCores (LNC=1), and the text encoder (UMT5) and the VAE run on 16 more NeuronCores. BF16.

Dependencies
^^^^^^^^^^^^

The plug-in is validated on Neuron SDK 2.33.0 and requires Python 3.11 or later.

* AWS Neuron SDK 2.33.0: ``neuronx-cc`` 2.28.7631.0, ``nki`` 0.7.0, ``aws-neuronx-runtime-lib`` 2.35.30.0, ``aws-neuronx-collectives`` 2.35.43.0, ``aws-neuronx-tools`` 2.33.38.0, and ``aws-neuronx-dkms`` 2.31.7.0
* ``torch-neuronx``: any build of the 2.14.3 line (PyTorch 2.14)
* SGLang: commit ``43881a7``. The plug-in stops the launch on any other SGLang revision.
* NIXL 1.5.0 (``nixl-cu13``), with patches that include ai-dynamo/nixl#2518, for disaggregated serving only.
* SGLang runtime dependencies, including ``transformers``: the versions that SGLang commit ``43881a7`` requires.

.. _sglang-0-1-0-container-image:

Container image
^^^^^^^^^^^^^^^

To run the plug-in in this public preview, use the plug-in container image. Neuron publishes the image in the Amazon ECR Public Gallery: `pytorch-inference-sglang-neuron <https://gallery.ecr.aws/neuron/pytorch-inference-sglang-neuron>`__. Pull the image:

.. code-block:: bash

   docker pull public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04

For the available tags, see the `image page in the Amazon ECR Public Gallery <https://gallery.ecr.aws/neuron/pytorch-inference-sglang-neuron>`__. For the plug-in source code and documentation, see the `sglang-neuron repository on GitHub <https://github.com/sgl-project/sglang-neuron>`__.

.. _sglang-0-1-0-known-issues:

Known issues
^^^^^^^^^^^^

The plug-in checks each launch and refuses what it does not serve. A refusal names the setting that caused it.

* **Supported models only.** The plug-in refuses to launch a model architecture that is not in its model catalog.
* **Refused SGLang features.** The plug-in refuses LoRA adapters, mixed chunks (``--enable-mixed-chunk``), embedding servers, and the SGLang weight-update endpoints, which return HTTP 400.
* **Pinned SGLang revision.** The plug-in runs only on SGLang commit ``43881a7``. On any other SGLang revision, the plug-in stops the launch with an error.
* **Pinned Neuron packages for GLM-5.2.** A GLM-5.2 launch refuses ``neuronx-cc``, ``nki``, runtime, or collectives versions other than the Neuron SDK 2.33.0 versions.

Changes in this release
^^^^^^^^^^^^^^^^^^^^^^^

**Attention backends**

* MLA and DeepSeek sparse attention (DSA) KV pools with a shared operand contract.

**MoE and expert parallelism**

* Mixture-of-experts (MoE) kernel optimizations.
* An exact node-wide ``all_to_all`` for NKI MoE kernels, on Neuron Runtime 2.35 (included in Neuron SDK 2.33.0).

**Parallelism and disaggregation**

* Data parallelism with ``--dp-size``.
* Pipeline parallelism and decode context parallelism for GPT-OSS-120B.
* Kimi-K3 prefill-decode (PD) prefill on one Trainium3 node (PP8 x TP16 x DCP16), with a chunk-keyed prefix cache and HiCache.
* Kimi-K3 PD decode across two Trainium3 nodes (DP8 x TP32 x DCP32), with DSPARK speculative decoding.

**Scheduler and runtime**

* A paged KV pool, a graph runner (compilation manager and bucketer), and a model registry.
* A Neuron sampler.
* Runs on upstream SGLang, with no fork required.
* One runtime for every model, with fully asynchronous execution and PD disaggregated serving.
* ``MegakernelSpec`` plug-in interfaces and a megakernel host.

**HiCache and radix cache**

* HiCache L2 on the ``torch-neuronx`` runtime.
* A remote DRAM tier below HiCache L2, over NIXL (libfabric).

**Model support and optimizations**

* Qwen3-MoE support and profiling utilities.
* GLM-4-9B support, with the latest plug-in optimizations.
* DeepSeek-V2-Lite serving.
* GPT-OSS-120B support and giga-kernel decode.
* Kimi-K3 prefill NKI kernels.
* GLM-5.2 (MXFP4) disaggregated serving on Trainium3 with Neuron SDK 2.33.0.
* Qwen3.5-397B-A17B aggregated and disaggregated serving, with a whole-model megakernel.

**SGLang Diffusion**

* A Neuron platform for SGLang Diffusion, validated on trn2.48xlarge.
* Tensor, context, and CFG parallelism, with ranks laid out on the Trn2 chip torus.
* NKI kernels: const-max ring attention, K/V all-gather attention, AdaLN, VAE convolution, and GroupNorm.
* Parallel VAEs for every model, and a streaming Wan VAE.
* Static whole-graph compilation with per-graph compiler flags, and launches at LNC=1.
* FLUX.1-dev text to image.
* MiniMax-H3 text and keyframes to video with audio.
* Wan2.2 A14B text to video and image to video.
* FLUX.3-Action robot policy over the SGLang OpenPI and HTTP action APIs.
* LingBot-World-V2 real-time world streaming with a paged K/V cache.
* Diffusion models in the container image.
* Model recipes for all five diffusion models.

.. TODO(11/12): The source doc lists "[Feat] Trainium2 serving" under Scheduler & Runtime.
   It is left out because LLMs are Trn3 only in this preview. Confirm with the plug-in team.

For more information, see the `sglang-neuron releases on GitHub <https://github.com/sgl-project/sglang-neuron/releases>`__.

Acknowledgments
^^^^^^^^^^^^^^^
* SGLang-Neuron was created in partnership with `Yotta Labs <https://www.yottalabs.ai/>`__. Neuron thanks their engineers who contributed to this work!

Related information
^^^^^^^^^^^^^^^^^^^

* `sglang-neuron repository on GitHub <https://github.com/sgl-project/sglang-neuron>`__
* `sglang-neuron container image in the Amazon ECR Public Gallery <https://gallery.ecr.aws/neuron/pytorch-inference-sglang-neuron>`__
* :ref:`What's new: SGLang-Neuron public preview <whats-new-2026-11-12-sglang-plugin-preview>`
* :doc:`Neuron component release notes </release-notes/components/index>`
* :ref:`TorchNeuron: Native PyTorch on AWS Neuron <torch-neuron-home>`
