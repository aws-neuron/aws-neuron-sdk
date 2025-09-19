"""
Copyright (c) 2025, Amazon.com. All Rights Reserved
"""
from attention_kernels import *
import neuronxcc.nki as nki
from neuronxcc.nki import benchmark, baremetal, simulate_kernel
import neuronxcc.nki.language as nl
import numpy as np

WORKING_DIRECTORY = f"/home/ubuntu/attention/"

####################################################################
# v0: Using Numpy to implement self-attention
####################################################################
def numpy_attention(q, k, v):
    """NumPy reference implementation"""
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q
    
    # Not doing Q @ K.T due to NKI layout constraints which require
    # Q transposed for matmul since contraction dimension 
    # has to be mapped to the partition dimension
    # Shape: (seqlen_q, seqlen_kv)
    qk = np.matmul(q.T, k)  
    
    # Softmax
    # Shape: (seqlen_q, 1)
    row_max = np.max(qk, axis=1, keepdims=True) 
    
    # Shape: (seqlen_q, seqlen_kv)
    norm_row = qk - row_max
    exp_row = np.exp(norm_row)
    
    # Shape: (seqlen_q, 1)
    sum_row = np.sum(exp_row, axis=1, keepdims=True)  
    
    # Shape: (seqlen_q, seqlen_kv)
    scores = exp_row / sum_row  
    
    # V transpose
    v_t = v.T  # Shape: (seqlen_kv, d_head)
    
    # scores @ V
    attn_out = np.matmul(scores, v_t)  # Shape: (seqlen_q, d_head)
    
    return attn_out

####################################################################
# Function to test functionality, profile, and benchmark attention
# kernels
####################################################################
def test_attn_tutorial(mode, version):
    if version == attn_fwd_v10 and mode == "benchmark":
        pytest.xfail("alloc + no_reorder fails with nki.jit and nki.benchmark.")

    if (version == attn_fwd_v1) or (version == attn_fwd_v2):
        # No tiling support in v1 and v2
        seqlen = 128
    else:
        seqlen = 4096
    
    if mode == "profile" or mode == "benchmark":
        dtype = nl.bfloat16
    else:
        dtype = nl.float32

    d_head = 128
    #  values between -1 and 1
    q = (np.random.random_sample([d_head, seqlen]) - 0.5) * 2
    k = (np.random.random_sample([d_head, seqlen]) - 0.5) * 2
    v = (np.random.random_sample([d_head, seqlen]) - 0.5) * 2

    q = nl.static_cast(q, dtype)
    k = nl.static_cast(k, dtype)
    v = nl.static_cast(v, dtype)

    if mode == "profile":
        if version == attn_fwd_v10:
            version(q, k, v)
        else:
            profile_func = nki.profile(working_directory=os.path.join(WORKING_DIRECTORY, f"{version.__name__}-profiles"),
                                       save_neff_name='file.neff', save_trace_name='profile.ntff', profile_nth=2)(version)
            profile_func(q, k, v)
    elif mode == "benchmark":
        bench_func = benchmark(warmup=5, iters=10)(version)
        bench_func(q, k, v)
        latency_res = bench_func.benchmark_result.nc_latency
        p99 = latency_res.get_latency_percentile(50)
        print(version.__name__, ":", p99, "usec")
    else:
        numpy_output = numpy_attention(q, k, v)
        attn_out = version(q, k, v)
        assert np.allclose(attn_out, numpy_output, atol=1e-2)

test_attn_tutorial("accuracy", attn_fwd_v1)
