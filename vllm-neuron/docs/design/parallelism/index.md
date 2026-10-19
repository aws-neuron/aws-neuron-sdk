# Parallelism

Design documentation for parallelism strategies. For configuration guidance,
see [Tensor, data, and expert parallelism](../../guides/features-guide.md#tensor-data-and-expert-parallelism)
in the features guide.

| Topic | Description |
| --- | --- |
| [Attention DP](attention_dp.md) | Sharding Q/O weights across DP groups |
| [Component DP sharding](component_dp_sharding.md) | Per-component independent sharding |
| [Data parallelism](data_parallelism.md) | Data parallelism overview |
| [Decode Context Parallelism](dcp.md) | KV cache sequence sharding for long contexts |
| [Encoder parallelism](encoder_parallelism.md) | Independent TP/DP for multimodal encoders |
| [Expert parallelism](expert_parallelism.md) | Expert parallelism for MoE |
| [Tensor parallelism](tensor_parallelism.md) | Tensor parallelism overview |

:::{toctree}
:maxdepth: 1
:hidden:

attention_dp
component_dp_sharding
data_parallelism
dcp
encoder_parallelism
expert_parallelism
tensor_parallelism
:::
