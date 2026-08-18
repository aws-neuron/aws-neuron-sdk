# Debugging model code

<!-- meta: description: Debugging techniques for vLLM Neuron -->
<!-- meta: content_type: conceptual-deep-dive -->
<!-- meta: date_updated: 2026-06-23 -->

<!-- REVIEW: 
     This doc should be the single debugging reference for both CPU and device.
     Currently it only covers CPU mode debugging (pdb, print, torch.compile).
     
     Proposed changes:
     1. Add on-device debugging section (what's different when running on Neuron
        vs CPU — what tools work, what doesn't, how to read device errors)
     2. Clean up internal paths (/workspace/src/NxDI/, /shared/truongnp/) — use
        generic paths or vllm_neuron model paths
     3. Condense raw error logs to show only the relevant lines
     4. "Debugging with Torch Eager on CPU mode" — confirm if enforce_eager is
        actually customer-ready. The doc says "Eager mode is not yet supported
        on Neuron, but will come soon."
     5. Add this doc to model-dev/index.md
     6. cpu-development.md debugging section should link here instead of
        duplicating the same pdb/print/BdbQuit content
-->

## Debugging with Torch Eager on CPU mode

With `VLLM_NEURON_CPU_MODE=1`, and `--enforce-eager` set, print statements are supported for all models using valid CPU mode configurations. Furthermore, with `VLLM_ENABLE_V1_MULTIPROCESSING=0`, normal pdb support is also enabled for models using valid CPU mode configurations and `world_size=1`.

To use pdb for `world_size > 1`, install [forked-pdb](https://github.com/Lightning-AI/forked-pdb) with `pip install fpdb`, and insert like:

``` python
__import__('fpdb').ForkedPdb().set_trace()
```

[Original Source](https://docs.vllm.ai/en/latest/usage/troubleshooting/#breakpoints)

These flags are recommended to be set during the CPU development phase.

> Eager mode is not yet supported on Neuron, but will come soon.

## Debugging with torch.compile

### Python Debugger (pdb)

#### Usage

This approach is useful when you want to inspect all variables defined before a breakpoint.

`torch.compile` uses `dynamo` to build FX graphs before generating HLOs and compiling NEFFs. Since `dynamo` runs in multiple processes, the Python debugger does not work out of the box. You must redirect I/O streams from child processes to the parent debugging process using our custom context `original_stdio`.

``` python
def forward(
    self,
    hidden_states: torch.Tensor,
    positions: torch.LongTensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attn_metadata: object | None = None,
):

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    with original_stdio():
        breakpoint()
```

``` text
(EngineCore_DP0 pid=217) 2026-01-06 18:29:14,026 - INFO - neuron_model_runner.py:575 - Starting model forward pass with input_ids.shape=torch.Size([19]), positions.shape=torch.Size([19])
> /workspace/src/NxDI/src/nxdi/model/llama3/model.py(616)torch_dynamo_resume_in_forward_at_616()
-> with original_stdio():
(Pdb) p hidden_states.shape
torch.Size([19, 4096])
(Pdb)
```

You can also print tensor values and retrieve actual local ranks of workers when in CPU mode:

``` text
(EngineCore_DP0 pid=459) (Worker_TP0 pid=465) 2026-01-06 18:45:40,349 - INFO - neuron_model_runner.py:575 - Starting model forward pass with input_ids.shape=torch.Size([19]), positions.shape=torch.Size([19])
> /workspace/src/NxDI/src/nxdi/model/llama3/model.py(616)torch_dynamo_resume_in_forward_at_616()
-> with original_stdio():
(Pdb)             breakpoint()
(Pdb) get_tensor_model_parallel_rank()
1
(Pdb) p hidden_states
tensor([[ 1.5411e-03, -1.1597e-02, -3.0151e-02,  ...,  3.5889e-02,
          3.0136e-04,  3.6011e-03],
        [ 4.9805e-02, -2.2949e-02, -4.4141e-01,  ..., -6.1035e-02,
          4.5898e-02, -1.0376e-03],
        [ 6.1523e-02, -3.4180e-02, -1.6699e-01,  ..., -5.4688e-02,
         -8.7280e-03,  4.6082e-03],
        ...,
        [ 1.9836e-03, -2.9297e-02,  1.4648e-01,  ..., -1.0132e-02,
          7.3853e-03, -5.9204e-03],
        [ 2.0874e-02,  1.7773e-01,  7.9102e-02,  ...,  1.4062e-01,
         -1.5503e-02, -2.3804e-02],
        [ 1.9836e-03, -2.9297e-02,  1.4648e-01,  ..., -1.0132e-02,
          7.3853e-03, -5.9204e-03]], dtype=torch.bfloat16)
(Pdb)
```

#### Current Limitations

Stepping through code in debug mode is not supported:

``` text
> /workspace/src/NxDI/src/nxdi/model/llama3/model.py(623)torch_dynamo_resume_in_forward_at_616()
-> hidden_states = self.self_attn(
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) (Pdb)
ERROR 01-06 21:17:45 [multiproc_executor.py:671] WorkerProc hit an exception.
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) ERROR 01-06 21:17:45 [multiproc_executor.py:671] Traceback (most recent call last):
...
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) ERROR 01-06 21:17:45 [multiproc_executor.py:671]   File "/workspace/src/NxDI/src/nxdi/model/llama3/model.py", line 616, in forward
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) ERROR 01-06 21:17:45 [multiproc_executor.py:671]     with original_stdio():
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) ERROR 01-06 21:17:45 [multiproc_executor.py:671]   File "/workspace/src/NxDI/src/nxdi/model/llama3/model.py", line 623, in torch_dynamo_resume_in_forward_at_616
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) ERROR 01-06 21:17:45 [multiproc_executor.py:671]     hidden_states = self.self_attn(
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) ERROR 01-06 21:17:45 [multiproc_executor.py:671]   File "/usr/lib/python3.10/bdb.py", line 90, in trace_dispatch
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) ERROR 01-06 21:17:45 [multiproc_executor.py:671]     return self.dispatch_line(frame)
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) ERROR 01-06 21:17:45 [multiproc_executor.py:671]   File "/usr/lib/python3.10/bdb.py", line 115, in dispatch_line
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) ERROR 01-06 21:17:45 [multiproc_executor.py:671]     if self.quitting: raise BdbQuit
(EngineCore_DP0 pid=277) (Worker_TP1 pid=285) ERROR 01-06 21:17:45 [multiproc_executor.py:671] bdb.BdbQuit
```

When using `torch.compile`, Dynamo intercepts and modifies bytecode execution. Dynamo also breaks the compute graph when it encounters code it cannot trace. In the example above, Dynamo generates a new function `torch_dynamo_resume_in_forward_at_616()` that throws an exception, causing the Python debugger to trigger `self.quitting`.

### Printing During Tracing

#### Print Usage

The current cpu-dev flow supports printing tensor shapes and values. This is useful when you want to examine various variables throughout the modeling code.

``` python
q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
print(f'--- q  {q[:, 0]}')
# Reshape to head layout
head_layout = (tokens, self.num_attention_heads_per_rank, self.head_dim)
kv_head_layout = (tokens, self.num_key_value_heads_per_rank, self.head_dim)
```

``` text
(EngineCore_DP0 pid=820) (Worker_TP1 pid=828) 2026-01-06 22:31:32,204 - INFO - neuron_model_runner.py:575 - Starting model forward pass with input_ids.shape=torch.Size([19]), positions.shape=torch.Size([19])
(EngineCore_DP0 pid=820) (Worker_TP1 pid=828) --- q  tensor([-0.3887, -0.7422, -1.0078, -0.5312, -0.2314, -0.3594, -0.8203, -1.6250,
(EngineCore_DP0 pid=820) (Worker_TP1 pid=828)         -1.5078, -0.3359, -1.5078, -0.5156, -1.5078, -0.3379, -1.5078, -0.5156,
(EngineCore_DP0 pid=820) (Worker_TP1 pid=828)         -1.5078, -0.6055, -1.5078], dtype=torch.bfloat16)
```

#### Print Limitations

When using `torch.compile` on Neuron devices, Dynamo graph breaks can cause neuronxcc errors when printing tensor shapes:

``` python
q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
print(f'--- q {q.shape}')
```

``` text
[INTERNAL_ERROR] [NCC_ITEN406] Too many partition dimensions detected at {{0,+,38}[9],+,380}[12]. This is usually due to unsupported (strided) access pattern
```

Dynamo graph breaks can also trigger recompilation errors:

``` python
def forward(
    self,
    hidden_states: torch.Tensor,
    positions: torch.LongTensor | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attn_metadata: object | None = None,
):

    residual = hidden_states
    print(f'--- residual {residual.shape}')
    hidden_states = self.input_layernorm(hidden_states)
```

``` text
(EngineCore_DP0 pid=2929544) (Worker_TP1 pid=2929553) --- residual torch.Size([19, 4096])
(EngineCore_DP0 pid=2929544) (Worker_TP0 pid=2929551) --- residual torch.Size([19, 4096])
(EngineCore_DP0 pid=2929544) (Worker_TP0 pid=2929551) [rank0]:W0106 23:01:14.131000 2929551 torch/_dynamo/convert_frame.py:1016] [6/8] torch._dynamo hit config.recompile_limit (8)
(EngineCore_DP0 pid=2929544) (Worker_TP1 pid=2929553) [rank1]:W0106 23:01:14.131000 2929553 torch/_dynamo/convert_frame.py:1016] [6/8] torch._dynamo hit config.recompile_limit (8)
(EngineCore_DP0 pid=2929544) (Worker_TP0 pid=2929551) [rank0]:W0106 23:01:14.131000 2929551 torch/_dynamo/convert_frame.py:1016] [6/8]    function: 'torch_dynamo_resume_in_forward_at_616' (/shared/truongnp/nxdi-v2/src/NxDI/src/nxdi/model/llama3/model.py:616)
(EngineCore_DP0 pid=2929544) (Worker_TP1 pid=2929553) [rank1]:W0106 23:01:14.131000 2929553 torch/_dynamo/convert_frame.py:1016] [6/8]    function: 'torch_dynamo_resume_in_forward_at_616' (/shared/truongnp/nxdi-v2/src/NxDI/src/nxdi/model/llama3/model.py:616)
(EngineCore_DP0 pid=2929544) (Worker_TP0 pid=2929551) [rank0]:W0106 23:01:14.131000 2929551 torch/_dynamo/convert_frame.py:1016] [6/8]    last reason: 6/7: self._modules['self_attn'].layer_idx == 7               (HINT: torch.compile considers integer attributes of the nn.Module to be static. If you are observing recompilation, you might want to make this integer dynamic using torch._dynamo.config.allow_unspec_int_on_nn_module = True, or convert this integer into a tensor.)
(EngineCore_DP0 pid=2929544) (Worker_TP1 pid=2929553) [rank1]:W0106 23:01:14.131000 2929553 torch/_dynamo/convert_frame.py:1016] [6/8]    last reason: 6/7: self._modules['self_attn'].layer_idx == 7               (HINT: torch.compile considers integer attributes of the nn.Module to be static. If you are observing recompilation, you might want to make this integer dynamic using torch._dynamo.config.allow_unspec_int_on_nn_module = True, or convert this integer into a tensor.)
(EngineCore_DP0 pid=2929544) (Worker_TP0 pid=2929551) [rank0]:W0106 23:01:14.131000 2929551 torch/_dynamo/convert_frame.py:1016] [6/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
(EngineCore_DP0 pid=2929544) (Worker_TP1 pid=2929553) [rank1]:W0106 23:01:14.131000 2929553 torch/_dynamo/convert_frame.py:1016] [6/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
(EngineCore_DP0 pid=2929544) (Worker_TP0 pid=2929551) [rank0]:W0106 23:01:14.131000 2929551 torch/_dynamo/convert_frame.py:1016] [6/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
(EngineCore_DP0 pid=2929544) (Worker_TP1 pid=2929553) [rank1]:W0106 23:01:14.131000 2929553 torch/_dynamo/convert_frame.py:1016] [6/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
...
(EngineCore_DP0 pid=2929544) (Worker_TP1 pid=2929553) ERROR 01-06 23:01:14 [multiproc_executor.py:671] RuntimeError: Expected self.dtype() == dst.dtype() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
(EngineCore_DP0 pid=2929544) (Worker_TP0 pid=2929551) ERROR 01-06 23:01:14 [multiproc_executor.py:671]   File "/shared/truongnp/nxdi-v2/src/NxDI/src/nxdi/model/llama3/model.py", line 470, in forward
(EngineCore_DP0 pid=2929544) (Worker_TP1 pid=2929553) ERROR 01-06 23:01:14 [multiproc_executor.py:671] Traceback (most recent call last):
(EngineCore_DP0 pid=2929544) (Worker_TP0 pid=2929551) ERROR 01-06 23:01:14 [multiproc_executor.py:671]     hidden_states = hidden_states.to(torch.float32)
(EngineCore_DP0 pid=2929544) (Worker_TP1 pid=2929553) ERROR 01-06 23:01:14 [multiproc_executor.py:671]   File "/shared/truongnp/venv-1218/lib/python3.12/site-packages/vllm/v1/executor/multiproc_executor.py", line 666, in worker_busy_loop
(EngineCore_DP0 pid=2929544) (Worker_TP0 pid=2929551) ERROR 01-06 23:01:14 [multiproc_executor.py:671]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2929544) (Worker_TP1 pid=2929553) ERROR 01-06 23:01:14 [multiproc_executor.py:671]     output = func(*args, **kwargs)
(EngineCore_DP0 pid=2929544) (Worker_TP0 pid=2929551) ERROR 01-06 23:01:14 [multiproc_executor.py:671] RuntimeError: Expected self.dtype() == dst.dtype() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
```
