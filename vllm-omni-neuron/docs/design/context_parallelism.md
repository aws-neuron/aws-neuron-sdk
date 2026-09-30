# Context Parallelism in vLLM Omni Neuron

<!-- meta: description: How context parallelism (CP) works in vLLM Omni Neuron —
sequence sharding on top of vLLM Omni's sequence-parallelism groups, the
ring-attention CP path used by default for the Wan2.2 DiT (with a Full K/V
AllGather fallback), how CP composes with tensor parallelism, the sequence-length
divisibility constraints, and the known kernel limitations. -->
<!-- meta: keywords: vLLM, Neuron, context parallelism, CP, sequence parallelism,
sequence sharding, ring attention, collective_permute, AllGather,
DeepSpeed-Ulysses, self-attention, rotary embeddings, tensor parallelism, NKI
kernels, Wan2.2, DiT, Trainium -->
<!-- meta: content_type: design-doc -->
<!-- meta: date_updated: 2026-08-26 -->

## Overview

Context Parallelism (CP) in vLLM Omni Neuron enables efficient processing of long sequences by distributing the sequence dimension across multiple ranks. This implementation leverages vLLM Omni's sequence parallelism infrastructure (`sequence_parallel_size`) to shard the input tokens while maintaining full attention computation. Because vLLM Omni enforces `sequence_parallel_size = ulysses_degree * ring_degree`, the CP degree is configured via `ring_degree` (with `ulysses_degree` left at its default of 1) — see [Configuration](#configuration).

Unlike traditional tensor parallelism that shards model parameters, context parallelism shards the sequence dimension, allowing each rank to process a subset of tokens while still computing attention over the full sequence. The communication strategy for attention can vary — ring attention (K/V chunks passed between ranks in a ring, so no rank materializes the full K/V), Full K/V AllGather (K/V gathered before attention so each rank computes local Q × full K/V), or DeepSpeed-Ulysses (All-to-All redistribution from sequence-partitioned to head-partitioned).

The current implementation uses **ring-attention CP** via a vendored ring-attention NKI kernel, which drives its own `collective_permute` around the CP group and keeps K/V memory at `1/cp_size`. It **falls back to Full K/V AllGather** only where the ring kernel cannot run (CPU mode, fake-tensor tracing, NKI disabled).

## Architecture

### Sequence Parallelism Integration

Context parallelism is implemented through vLLM Omni's sequence parallelism (`SP`) framework:

```python
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

sp_group = get_sp_group()
self.cp_size = sp_group.world_size
self.cp_rank = sp_group.rank_in_group
self.cp_group = sp_group if self.cp_size > 1 else None
```

### Process Groups

Context parallelism operates alongside tensor parallelism (TP):

- **TP Group**: Shards model parameters (attention heads, FFN dimensions)
- **CP Group**: Shards the sequence dimension across ranks
- **Combined**: Each rank has `(tp_rank, cp_rank)` coordinates

Example with TP=4, CP=8 (32 ranks total):
```text
Rank 0:  tp_rank=0, cp_rank=0 (heads 0-9,  tokens 0-127)
Rank 1:  tp_rank=0, cp_rank=1 (heads 0-9,  tokens 128-255)
...
Rank 7:  tp_rank=0, cp_rank=7 (heads 0-9,  tokens 896-1023)
Rank 8:  tp_rank=1, cp_rank=0 (heads 10-19, tokens 0-127)
...
Rank 31: tp_rank=3, cp_rank=7 (heads 30-39, tokens 896-1023)
```

## Model Implementation

### WanTransformer3DModel

The main transformer model implements context parallelism through sequence sharding:

```python
# Split sequence across CP ranks before transformer blocks
if self.cp_size > 1:
    S = hidden_states.shape[1]  # Total sequence length
    if S % self.cp_size != 0:
        raise ValueError(
            f"Sequence length {S} is not divisible by cp_size {self.cp_size}. "
            f"Choose a resolution/frame count that yields a divisible patch sequence length."
        )
    local_S = S // self.cp_size
    start = self.cp_rank * local_S
    hidden_states = hidden_states[:, start : start + local_S, :]
    
    # Slice rotary embeddings to match local positions
    freqs_cos, freqs_sin = rotary_emb
    freqs_cos = freqs_cos[:, start : start + local_S, :, :]
    freqs_sin = freqs_sin[:, start : start + local_S, :, :]
    rotary_emb = (freqs_cos, freqs_sin)

# Process through transformer blocks
for block in self.blocks:
    hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

# Gather full sequence after transformer blocks
if self.cp_size > 1:
    hidden_states = self.cp_group.all_gather(hidden_states.contiguous(), dim=1)
```

### Self-Attention with Context Parallelism

The `WanSelfAttention` module projects and RoPEs its local Q/K/V shard, then hands the CP policy to the shared `wan_cp_self_attention` core.

#### Ring attention (current implementation)

K/V stay sharded (`local_S` tokens per rank). The ring-attention NKI kernel drives a `collective_permute` around the CP group, streaming each rank's K/V chunk through every other rank so local Q attends the full sequence without ever materializing full K/V. Q remains sequence-partitioned — no output redistribution is needed.

#### Full K/V AllGather (fallback)

Where the ring kernel cannot run (CPU mode, fake-tensor tracing, NKI disabled), each rank AllGathers K/V from all CP ranks to materialize the full sequence, then computes local Q × full K/V attention:

```python
class WanSelfAttention(nn.Module):
    def __init__(self, ...):
        # CP group setup
        sp_group = get_sp_group()
        self.cp_size = sp_group.world_size
        self.cp_group = sp_group if self.cp_size > 1 else None
    
    def forward(self, hidden_states: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        # QKV projection (NKI kernel or matmul fallback)
        if self._use_nki_qkv:
            qkv = NF.qkv_proj(hidden_states, self.qkv_proj_weight, bias=self.qkv_proj_bias.unsqueeze(0))
        else:
            qkv = torch.matmul(hidden_states, self.qkv_proj_weight) + self.qkv_proj_bias
        q, k, v = torch.tensor_split(qkv, self.qkv_split, dim=-1)
        
        # QK-norm + reshape to multi-head: [B, S, H] -> [B, S, N, D]
        query = self.norm_q(q).unflatten(2, (self.num_heads, self.head_dim))
        key = self.norm_k(k).unflatten(2, (self.num_heads, self.head_dim))
        value = v.unflatten(2, (self.num_heads, self.head_dim))
        
        # Apply rotary embeddings to local tokens
        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb
            query = apply_rotary_emb_wan(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb_wan(key, freqs_cos, freqs_sin)
        
        query = query.transpose(1, 2)  # [B, N, local_S, D]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # CP: AllGather K/V across SP group to get full sequence
        # key/value before: [B, N, local_S, D] -> after: [B, N, S, D]
        if self.cp_size > 1:
            key = self.cp_group.all_gather(key.contiguous(), dim=2)
            value = self.cp_group.all_gather(value.contiguous(), dim=2)
        
        # Attention: local Q [B, N, local_S, D] × full K/V [B, N, S, D]
        hidden_states = _nf_attend(query, key, value, self.scale)
        
        # Output projection via NKI kernel: [B, N, local_S, D] -> [B, local_S, H]
        output = NF.o_proj(
            hidden_states.transpose(2, 3),  # [B, N, D, local_S]
            self.o_proj_weight,
            self.o_proj_bias.unsqueeze(0),
        )
        if self.tp_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)
        return output
```

#### Comparison with Other CP Strategies

| Aspect | Ring Attention (current) | Full K/V AllGather (fallback) | DeepSpeed-Ulysses |
|--------|--------------------------|-------------------------------|-------------------|
| Communication primitive | `collective_permute` in ring | AllGather on K/V | All-to-All on Q,K,V + All-to-All on output |
| Rounds per layer | cp_size send/recv | 1 AllGather | 2 All-to-All |
| K/V memory per rank | 1/cp_size of sequence | Full sequence (redundant) | Full sequence for local heads only |
| Head constraint from CP | None | None | `num_heads % cp_size == 0` |
| Compute-comm overlap | Overlaps attention with P2P | None | None |

Ring attention minimizes peak K/V memory (its main draw for long sequences) at the cost of `cp_size` communication rounds; Full K/V AllGather is a single collective with no head-count constraints, kept as the always-correct fallback; DeepSpeed-Ulysses is more memory-efficient than AllGather at large CP degrees but adds head-count constraints and is not used here.

## Communication Pattern

### Forward Pass Flow

1. **Input Embedding**: Full sequence processed locally (e.g., patch embedding for DiT)
2. **Sequence Sharding**: Split tokens across CP ranks
3. **Transformer Blocks**:
   - Each rank processes its local token subset
   - Self-attention: ring `collective_permute` of local K/V shards → local Q attends full sequence (Full K/V AllGather only on the fallback path)
   - Cross-attention uses full encoder states (no sharding)
   - FFN operates on local tokens
4. **Sequence Gathering**: Reconstruct full sequence
5. **Output Projection**: Full sequence for downstream processing (e.g., unpatchify)

### Memory and Compute Benefits

- **Memory**: Reduces activation memory by `1/cp_size` for transformer blocks
- **Compute**: Maintains full attention quality while distributing sequence processing
- **Communication**: Inter-rank communication scales with sequence length, not model size
- **Generality**: Applicable to any model with a sequence dimension (DiT, LLM, etc.)

## Configuration

Context parallelism is configured through vLLM Omni's sequence parallelism. vLLM Omni enforces `sequence_parallel_size = ulysses_degree * ring_degree`, so rather than setting `sequence_parallel_size` directly, **the stage config sets `ring_degree` and leaves the other two unset**: `ulysses_degree` defaults to 1, and `sequence_parallel_size` is derived as `ulysses_degree * ring_degree = ring_degree`. `ring_degree` is therefore the single CP knob in the stage config, and the CP degree (`cp_size`) equals it:

```yaml
# engine_args.parallel_config in the stage config
parallel_config:
  tensor_parallel_size: 4
  ring_degree: 8          # CP degree — sets sequence_parallel_size = 1 * 8 = 8
  cfg_parallel_size: 2
```

Note that despite the name, `ring_degree` only sizes the sequence-parallel group — it does not necessarily select a ring-attention algorithm. The SP group it produces is what `get_sp_group()` returns and what this implementation shards the sequence across.

### Constraints

- `sequence_length % cp_size == 0`: Sequence must be evenly divisible
- `local_S % 2 == 0`: NKI MLP kernel requires even local sequence length (enforced by padding within the MLP kernel)
- Compatible with tensor parallelism: `total_ranks = tp_size * cp_size`
- Positional embeddings must support local position slicing
- CP group ranks must map to directly connected devices on the target hardware

## Performance Characteristics

### Scaling Properties

- **Sequence Length**: Linear memory reduction with CP degree
- **Model Size**: Orthogonal to tensor parallelism scaling
- **Communication**: Ring `collective_permute` moves one K/V shard (`1/cp_size` of the sequence) per round over `cp_size` rounds, overlapped with attention compute; the AllGather fallback instead materializes full K/V on every rank

### Optimal Use Cases

- Long sequences that exceed single-device memory capacity
- Long video sequences (high frame count) in diffusion models
- High-resolution images (large spatial token count)
- Memory-constrained scenarios with long contexts
- Balanced with tensor parallelism for model parameter distribution

## Implementation Notes

### Positional Embedding Handling

Local positional embeddings (e.g., rotary embeddings) are sliced to match the local token positions:

```python
# Slice rotary_emb to match local token positions
freqs_cos = freqs_cos[:, start : start + local_S, :, :]
freqs_sin = freqs_sin[:, start : start + local_S, :, :]
```

### Distributed Normalization

`DistributedRMSNorm` computes global statistics across TP ranks only, ensuring correct normalization within each CP group:

```python
if self.tp_size > 1:
    global_sum_sq = local_sum_sq.clone()
    dist.all_reduce(global_sum_sq, group=self.tp_group)  # TP group only
```

This design ensures that context parallelism provides efficient long-sequence processing while maintaining compatibility with existing tensor parallelism and attention mechanisms.

## Sequence Length Constraints

The sequence length must be divisible by `cp_size`. If it is not, the model raises a `ValueError` at runtime. Users must choose input dimensions (resolution, frame count) that produce a compatible sequence length.

### Problem

For example, `height=240, width=432, num_frames=33` yields:
- `latent_frames = (33-1)//4 + 1 = 9`
- `post_patch_h = (240//8) // 2 = 15`
- `post_patch_w = (432//8) // 2 = 27`
- `S_total = 9 * 15 * 27 = 3645`
- `3645 % 8 = 5` — not divisible by `cp_size=8`

This will raise:
```
ValueError: Sequence length 3645 is not divisible by cp_size 8.
Choose a resolution/frame count that yields a divisible patch sequence length.
```

### NKI MLP Kernel Padding

Separately, the NKI MLP kernel uses `grid=(2,)` which requires the sequence dimension to be even. This is handled locally within `WanFeedForward.forward` — the input is padded by 1 token before the kernel call and trimmed after, so no other layer sees padded tokens:

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_shape = hidden_states.shape
    hidden_2d = hidden_states.reshape(-1, input_shape[-1])

    # NKI MLP kernel uses grid=(2,) requiring T to be even.
    T = hidden_2d.shape[0]
    pad_T = T % 2
    if pad_T:
        hidden_2d = F.pad(hidden_2d, (0, 0, 0, 1))

    output = NF.mlp(hidden_2d, ...)

    if pad_T:
        output = output[:T, :]
    ...
```

This localized padding has no impact on accuracy since the padded position is discarded immediately after the kernel.

## vLLM Omni Integration

The CP implementation depends on vLLM Omni's `get_sp_group()` for process group management. In **production**, vLLM Omni's `initialize_model_parallel()` sets up the SP group automatically. In **unit tests**, it must be initialized manually:

```python
import vllm_omni.diffusion.distributed.parallel_state as omni_ps

# SP group (required by WanTransformer3DModel.__init__ and WanSelfAttention)
ulysses_pg, ring_pg = omni_ps.set_seq_parallel_pg(
    sp_ulysses_degree=1, sp_ring_degree=sp_size,
    rank=rank, world_size=world_size, sp_group_ranks=sp_group_ranks,
)
omni_ps._SP = omni_ps.init_model_parallel_group(
    group_ranks=sp_group_ranks, ..., parallel_mode="sequence",
    ulysses_group=ulysses_pg, ring_group=ring_pg,
)
```

## Known Issues

| Issue | Description |
|-------|-------------|
| NKI QKV kernel disabled under TP4 | The `NF.qkv_proj` NKI kernel exceeds SBUF budget when the fused QKV output dimension > 2048. With TP4, each rank's QKV size is `3 * (5120 / 4) = 3840`, which exceeds this limit. The code falls back to `torch.matmul` for QKV projection. |
| Sequence length must be divisible by `cp_size` | Arbitrary input resolutions/frame counts that produce a non-divisible sequence length will raise a `ValueError`. Users must select compatible dimensions. |
| NKI MLP kernel requires even sequence length | The NKI MLP kernel uses `grid=(2,)`, requiring the flattened sequence dimension to be even. Handled by local padding in `WanFeedForward.forward`, but adds a minor overhead for odd-length inputs. |

## Related information

- [Kernel reference: `ring_attention_const_max_fwd`](../model-dev/kernels/ring-attention-const-max.md) — the design of the ring-attention kernel this page configures: why its softmax max is a runtime bound, and how that lets the K/V rotation overlap attention compute.
- [Kernel implementations](../model-dev/kernels/index.md) — other per-kernel design references.
- [Design: Engine, Worker, and Model Integration](vllm_omni_neuron_overview.md) — where CP sits in the runtime.
- [Features guide](../guides/features-guide.md) — the user-facing parallelism configuration.
