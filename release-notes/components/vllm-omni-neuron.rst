.. meta::
    :description: Release notes for vLLM Omni Neuron — diffusion-based image and video generation on AWS Trainium with vLLM Omni.
    :keywords: vllm omni neuron, vllm omni, diffusion, video generation, image generation, release notes, aws neuron sdk
    :date-modified: 10/19/2026

.. _vllm-omni-neuron_rn:

vLLM Omni Neuron Release Notes
===============================

**Latest version (in 2.33.0)**: 0.29.0.0.2.0 (Beta)

The vLLM Omni Neuron plugin is the recommended serving solution for diffusion and
multimodal generation models on AWS Trainium. It extends vLLM Omni with a Neuron
backend, providing the same vLLM Omni APIs and configuration you are already familiar
with. For an introduction, see :doc:`/vllm-omni-neuron/docs/index`.

For the supported models, features, and optimized NKI kernels, see the
`vLLM Omni Neuron README on GitHub <https://github.com/aws-neuron/vllm-omni-neuron#readme>`_.
For the release notes, see the `vLLM Omni Neuron releases on GitHub <https://github.com/aws-neuron/vllm-omni-neuron/releases>`_.

.. _vllm-omni-neuron-2-33-0-rn:

vLLM Omni Neuron [0.29.0.0.2.0] (Neuron 2.33.0 Release)
-------------------------------------------------------

vLLM Omni Neuron 0.29.0.0.2.0 (Beta) moves to native PyTorch ``torch.compile``
and improves Wan2.2 text-to-video and image-to-video generation performance on
Trn2 and Trn3. This version ships with vLLM Omni 0.29.0rc1.

For the full list of changes, see the `vLLM Omni Neuron releases on GitHub <https://github.com/aws-neuron/vllm-omni-neuron/releases>`_.

.. _vllm-omni-neuron-0-24-0-0-1-0-rn:

vLLM Omni Neuron [0.24.0.0.1.0] (Beta, September 30, 2026)
-----------------------------------------------------------

The initial Beta release of vLLM Omni Neuron supports Wan2.2 T2V-A14B (text to video) and
Wan2.2 I2V-A14B (image to video) on Trn2 and Trn3. It provides offline ``Omni.generate`` and the
OpenAI-compatible ``/v1/videos`` endpoint; Neuron-optimized NKI kernels for fused QKV projection,
const-max and ring attention, fused adaptive LayerNorm with FP8 quantization, and MLP; tensor,
context, Megatron sequence, classifier-free guidance, and VAE patch parallelism; and Cache-DiT
diffusion caching for faster Wan2.2-T2V generation.

For details, see the :doc:`model recipes </vllm-omni-neuron/docs/models/index>`, the
:doc:`deployment tutorial </vllm-omni-neuron/docs/tutorials/tutorial-wan22-14b>`, and the
`v0.24.0.0.1.0 release notes on GitHub <https://github.com/aws-neuron/vllm-omni-neuron/releases/tag/v0.24.0.0.1.0>`_.
