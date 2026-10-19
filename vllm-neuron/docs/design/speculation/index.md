# Speculation

Design documentation for speculative decoding. For configuration guidance,
see [Speculative decoding](../../guides/features-guide.md#speculative-decoding)
in the features guide and the EAGLE3 tutorials for
[Llama 3.1](../../tutorials/tutorial-eagle3-speculative-decoding-llama-3-1.md) and
[GPT-OSS](../../tutorials/tutorial-eagle3-speculative-decoding-gpt-oss.md).

| Topic | Description |
| --- | --- |
| [Speculative decoding (EAGLE3)](speculative_decoding_design.md) | EAGLE3 autoregressive drafter internals on Neuron |
| [Speculative decoding (DFlash)](dflash_speculative_decoding_design.md) | DFlash block-parallel drafter internals on Neuron |

:::{toctree}
:maxdepth: 1
:hidden:

speculative_decoding_design
dflash_speculative_decoding_design
:::
