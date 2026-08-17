# Multimodal

Design documentation for multimodal inference. For configuration guidance,
see [Multimodal support](../../guides/features-guide.md#multimodal-support)
in the features guide.

| Topic | Description |
| --- | --- |
| [Block Packing Attention](block_packing_attention.md) | FFD block packing for multi-image attention efficiency |
| [On-Device Encoder Cache](on_device_encoder_cache.md) | Block-based on-device cache for vision encoder outputs |
| [M-RoPE](mrope.md) | Spatial position embeddings for VLMs |

:::{toctree}
:maxdepth: 1
:hidden:

block_packing_attention
on_device_encoder_cache
mrope
:::
