from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import BaseParallelLinear, ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from neuronx_distributed.parallel_layers.utils import divide

import torch
from torch import nn
from torch.nn.parameter import Parameter
from transformers import T5Config
from transformers.activations import ACT2FN
from transformers.pytorch_utils import find_pruneable_heads_and_indices
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention, T5LayerNorm,\
    T5LayerCrossAttention, T5LayerFF, T5DenseGatedActDense, T5DenseActDense

from transformers import T5ForConditionalGeneration
import neuronx_distributed

def prune_linear_layer(layer: BaseParallelLinear, index: torch.LongTensor,
                       dim: int = 0) -> BaseParallelLinear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`BaseParallelLinear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `BaseParallelLinear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = ColumnParallelLinear(new_size[1],
                                     new_size[0],
                                     bias=layer.bias is not None,
                                     gather_output=False).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class ParallelAttention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        # Per attention head and per partition values
        world_size = parallel_state.get_tensor_model_parallel_size()
        self.num_attention_heads_per_partition = divide(
            self.n_heads, world_size)
        self.hidden_size_per_partition = self.num_attention_heads_per_partition * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = ColumnParallelLinear(self.d_model,
                                      self.inner_dim,
                                      bias=False,
                                      gather_output=False)
        self.k = ColumnParallelLinear(self.d_model,
                                      self.inner_dim,
                                      bias=False,
                                      gather_output=False)
        self.v = ColumnParallelLinear(self.d_model,
                                      self.inner_dim,
                                      bias=False,
                                      gather_output=False)
        self.o = RowParallelLinear(self.inner_dim,
                                   self.d_model,
                                   bias=False,
                                   input_is_parallel=True)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = ParallelEmbedding(self.relative_attention_num_buckets, self.n_heads)
        self.n_heads = self.num_attention_heads_per_partition

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads_per_partition, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.num_attention_heads_per_partition = self.num_attention_heads_per_partition - len(heads)
        self.hidden_size_per_partition = self.key_value_proj_dim * self.num_attention_heads_per_partition
        self.pruned_heads = self.pruned_heads.union(heads)

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket)
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        values = values[:, :, tp_rank * self.num_attention_heads_per_partition:(tp_rank + 1)
                                                                     * self.num_attention_heads_per_partition]

        # values = self.relative_attention_bias(
        #     relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0)  # shape (1, num_heads, query_length, key_length)
        # print("Values shape is: ", values.shape)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        self.is_decoder = True
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                    len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.num_attention_heads_per_partition,
                               self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.hidden_size_per_partition)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                # import pdb; pdb.set_trace()
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states,
            past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states,
            past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.num_attention_heads_per_partition, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]

            if mask is not None:
                print(position_bias.shape, mask.shape, flush=True)
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        # print("Scores is: ", scores.shape)
        # print("position_bias_masked: ", position_bias_masked.shape)
        # print(scores.dtype, position_bias_masked.dtype)

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(
            torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        print(self.is_decoder,use_cache, flush=True)
        present_key_value_state = (key_states, value_states) if (
                self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class ParallelSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias=False)
        self.SelfAttention = ParallelAttention(config,
                                         has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


class ParallelCrossAttention(T5LayerCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        self.EncDecAttention = ParallelAttention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


class ParallelDenseActDense(T5DenseActDense):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, gather_output=False, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, input_is_parallel=True, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]


class ParallelDenseGatedActDense(T5DenseGatedActDense):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.wi_0 = ColumnParallelLinear(config.d_model,
                                      config.d_ff,
                                         gather_output=False,
                                      bias=False)
        self.wi_1 = ColumnParallelLinear(config.d_model,
                                      config.d_ff,
                                        gather_output=False,
                                      bias=False)
        self.wo = RowParallelLinear(config.d_ff,
                                    config.d_model,
                                    input_is_parallel=True,
                                    bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]


class ParallelFF(T5LayerFF):
    def __init__(self, config: T5Config):
        super().__init__(config)
        if config.is_gated_act:
            self.DenseReluDense = ParallelDenseGatedActDense(config)
        else:
            self.DenseReluDense = ParallelDenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


def load_pretrained_with_parallel_attn(model_name):
    
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")

    # Parallel implementation of Attention modules.
    from t5_model_layers import ParallelSelfAttention, ParallelFF, ParallelCrossAttention

    for index, block in enumerate(model.decoder.block):
        if index == 0:
            block.layer[0] = ParallelSelfAttention(model.config,
                                                   has_relative_attention_bias=True)
        else:
            block.layer[0] = ParallelSelfAttention(model.config)
        block.layer[1] = ParallelCrossAttention(model.config)
        block.layer[2] = ParallelFF(model.config)
    # Load the weights into the parallel layers        
    neuronx_distributed.parallel_layers.load(model_name.split("/")[-1] + ".pt", model, sharded=False)

    return model
