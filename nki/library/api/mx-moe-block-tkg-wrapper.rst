.. meta::
    :description: Wrapper that bitcasts unsigned integer weights to MX x4 dtype before calling the kernel.
    :date-modified: 05/21/2026

.. currentmodule:: nkilib.experimental.moe_block

Mx Moe Block Tkg Wrapper Kernel API Reference
=============================================

Wrapper that bitcasts unsigned integer weights to MX x4 dtype before calling the kernel.

Simulates how NxD passes MX weights — as raw uint16/uint32 tensors that need to be reinterpreted as float4_e2m1fn_x4 or float8_e4m3fn_x4 dtype.

Background
-----------

The ``mx_moe_block_tkg_wrapper`` kernel provides a wrapper that bitcasts unsigned integer weight tensors to MX x4 dtype (float4_e2m1fn_x4 or float8_e4m3fn_x4) before invoking the MoE TKG kernel, simulating how NxD passes MX weights as raw integer tensors.

API Reference
--------------

**Source code for this kernel API can be found at**: `mx_moe_block_tkg_wrapper.py <https://github.com/aws-neuron/nki-library/blob/main/src/nkilib_src/nkilib/experimental/moe_block/mx_moe_block_tkg_wrapper.py>`_

mx_moe_block_tkg_wrapper
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: mx_moe_block_tkg_wrapper(inp, gamma, router_weights, expert_gate_up_weights, expert_down_weights, shared_expert_gate_w = None, shared_expert_up_w = None, shared_expert_down_w = None, expert_gate_up_weights_scale = None, expert_down_weights_scale = None, router_bias = None, expert_gate_up_bias = None, expert_down_bias = None, shared_expert_gate_bias = None, shared_expert_up_bias = None, shared_expert_down_bias = None, eps = 1e-06, top_k = 1, router_act_fn = None, router_pre_norm = True, norm_topk_prob = False, expert_affinities_scaling_mode = None, hidden_act_fn = None, hidden_act_scale_factor = None, hidden_act_bias = None, gate_clamp_upper_limit = None, gate_clamp_lower_limit = None, up_clamp_upper_limit = None, up_clamp_lower_limit = None, router_mm_dtype = None, hidden_actual = None, skip_router_logits = False, is_all_expert = False, rank_id = None, residual = None, expert_gate_up_input_scale = None, expert_down_input_scale = None)

   Wrapper that bitcasts unsigned integer weights to MX x4 dtype before calling the kernel.


