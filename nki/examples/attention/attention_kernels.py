import numpy as np
from neuronxcc import nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import os
import torch
import logging

from neuronxcc.nki.language import par_dim
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

####################################################################
# v1: toy example with 128 seqlen and nki.lang APIs
####################################################################
@nki.jit
def attn_fwd_v1(q, k, v):
    """nki.lang APIs"""
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q
    
    assert q.shape == k.shape == v.shape
    assert d_head == 128
    assert seqlen_q == 128
    
    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)
   
    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)
    
    # Q @ K, contract along d_head #
    qk: nt.tensor[seqlen_q, seqlen_kv] = nl.matmul(x=q_sbuf, y=k_sbuf, transpose_x=True)
    
    # Softmax #
    # reduce max along seqlen_k
    row_max = nl.max(qk, axis=1)
    
    # subtract max from row
    norm_row = nl.subtract(qk, row_max)
    
    # exponentiation
    exp_row = nl.exp(norm_row)
    
    # sum of exp results
    sum_row = nl.sum(exp_row, axis=1)
    
    # divide exp results by sum
    scores: nt.tensor[seqlen_q, seqlen_kv] = nl.divide(exp_row, sum_row)
    
    # v has the wrong layout
    v_sbuf_t: nt.tensor[seqlen_kv, d_head] = nl.transpose(v_sbuf)
    
    # scores @ V, contract along seqlen_kv
    attn_out: nt.tensor[seqlen_q, d_head] = nl.matmul(scores, v_sbuf_t, transpose_x=False)
    
    # store output
    nl.store(dst=kernel_out, value=attn_out)
    return kernel_out

####################################################################
# v2: use nki.isa APIs
####################################################################
@nki.jit
def attn_fwd_v2(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q
    assert q.shape == k.shape == v.shape
    assert d_head == 128
    assert seqlen_q == 128
    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)
    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)
    # Q @ K, contract along d_head #
    qk: nt.tensor[seqlen_q, seqlen_kv] = nisa.nc_matmul(stationary=q_sbuf,
                                                        moving=k_sbuf)
    # Softmax #
    # reduce max along seqlen_kv, dimension: [seqlen_q, 1]
    row_max = nisa.tensor_reduce(op=nl.max, data=qk, axis=1)
    # subtract max from row, dimension: [seqlen_q, seqlen_kv]
    norm_row = nisa.tensor_scalar(data=qk,
                                op0=nl.subtract,
                                operand0=row_max,
                                engine=nisa.vector_engine)
    # exponentiation, dimension: [seqlen_q, seqlen_kv]
    exp_row = nisa.activation(op=nl.exp, data=norm_row, bias=None, scale=1.0)
    # sum of exp results, dimension: [seqlen_q, 1]
    sum_row = nisa.tensor_reduce(op=nl.add,
                                data=exp_row,
                                axis=1)
    # reciprocal of sum_row, dimension: [seqlen_q, 1]
    inverse_sum_row = nisa.reciprocal(data=sum_row)
    scores: nt.tensor[seqlen_q, seqlen_kv] = nisa.tensor_scalar(data=exp_row,
                                    op0=nl.multiply,
                                    operand0=inverse_sum_row,
                                    engine=nisa.vector_engine,
                                    dtype=q.dtype)
    # v has the wrong layout
    v_psum_t = nisa.nc_transpose(v_sbuf)          # TensorE
    # dimension: [seqlen_kv, d_head]
    v_sbuf_t = nisa.tensor_copy(v_psum_t)         # ScalarE
    # scores has the wrong layout
    scores_psum_t = nisa.nc_transpose(scores)          # TensorE
    # dimension: [seqlen_kv, seqlen_q]
    scores_sbuf_t = nisa.tensor_copy(scores_psum_t)    # ScalarE
    # scores @ V, contract along seqlen_kv
    attn_out: nt.tensor[seqlen_q, d_head] = nisa.nc_matmul(stationary=scores_sbuf_t,
                                                           moving=v_sbuf_t)
    # store output
    nl.store(dst=kernel_out, value=attn_out)
    return kernel_out


####################################################################
# v3: large sequence length with tiling
####################################################################
@nki.jit
def attn_fwd_v3(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # Tile along seqlen_q #
    # for this example we assume that seqlen_q is divisible by PMAX and 
    # seqlen_kv is divisible by FMAX_MOVING, otherwise need to use mask or "final multiplication"
    qk = nl.ndarray((seqlen_q // PMAX, seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                     dtype=nl.float32, buffer=nl.psum)
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_q, i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

    # Softmax #
    # reduce max along seqlen_k
    row_max = nl.ndarray((nl.par_dim(PMAX), seqlen_q // PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_q, i_tile_kv], axis=1)
 
        row_max[:, i_tile_q, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1)

    # subtract max from row
    norm_row = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv),
                       dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        norm_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            norm_buf[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.tensor_scalar(
                data=qk[i_tile_q, i_tile_kv],
                op0=nl.subtract,
                operand0=row_max[:, i_tile_q, :],
                engine=nisa.vector_engine)
        nl.store(norm_row[i_tile_q], norm_buf[:,:])

    # exponentiation
    exp_row = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        # norm_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        exp_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        norm_buf = nl.load(norm_row[i_tile_q])
        exp_buf[:,:] = nisa.activation(op=nl.exp, data=norm_buf)
        nl.store(exp_row[i_tile_q], exp_buf[:,:])

    # sum of exp results
    sum_row = nl.ndarray((nl.par_dim(PMAX), seqlen_q // PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        exp_buf = nl.load(exp_row[i_tile_q])
        sum_row[:, i_tile_q] = nisa.tensor_reduce(op=nl.add,
                                                         data=exp_buf,
                                                         axis=1)

    # reciprocal of sum_row, tile shape is [PMAX, seqlen_q // PMAX]
    inverse_sum_row = nisa.reciprocal(data=sum_row)
    
    scores = nl.ndarray((seqlen_q // PMAX, PMAX, seqlen_kv), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        scores_buf = nl.ndarray(shape=(nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        exp_buf = nl.load(exp_row[i_tile_q])
        scores_buf[:,:] = nisa.tensor_scalar(data=exp_buf,
                                               op0=nl.multiply,
                                               operand0=inverse_sum_row[:, i_tile_q],
                                               engine=nisa.vector_engine,
                                               dtype=nl.float32)
        nl.store(scores[i_tile_q], scores_buf[:,:])
        
    # v has the wrong layout
    v_t = nl.ndarray((seqlen_kv // PMAX, PMAX, d_head), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), d_head), dtype=nl.float32, buffer=nl.sbuf)
        v_sbuf_t[:, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)                   # ScalarE
        nl.store(v_t[i_tile_kv], v_sbuf_t[:,:])

    # scores has the wrong layout
    # PMAX restriction on both free and partition dimension when performing transpose.
    # scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, seqlen_q // PMAX, PMAX),
    #                            dtype=nl.float32, buffer=nl.sbuf)
    scores_t = nl.ndarray((seqlen_kv // PMAX, seqlen_q // PMAX, PMAX, PMAX), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_tile_q in nl.affine_range(seqlen_q // PMAX):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_buf = nl.load(scores[i_tile_q, :, nl.ds(i_tile_kv*PMAX, PMAX)])
            scores_psum_t = nisa.nc_transpose(scores_buf) # TensorE
            scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), PMAX), dtype=nl.float32, buffer=nl.sbuf)
            scores_sbuf_t[:, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)    # ScalarE
            nl.store(scores_t[i_tile_kv, i_tile_q, :, :], scores_sbuf_t)

    # scores @ V, contract along seqlen_kv
    # d_head == P_MAX, no need to tile there
    for i_tile_q in nl.affine_range(seqlen_q // PMAX): # loop on stationary free
        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)
        attn_out = nl.ndarray((nl.par_dim(PMAX), d_head),
                           dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            scores_sbuf_t = nl.load(scores_t[i_tile_kv, i_tile_q, :, :])
            v_sbuf_t = nl.load(v_t[i_tile_kv, :, :])
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t,
                                            moving=v_sbuf_t)
        attn_out[:, :] = nisa.tensor_copy(attn_out_psum)
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:,:])

    return kernel_out


####################################################################
# v4: Loop fusion
# combines QK matrix multiplication, all softmax steps, and V 
# multiplication to compute attention scores & output under one 
# common loop.
####################################################################
@nki.jit
def attn_fwd_v4(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, d_head), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)     # ScalarE

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        # per i_tile_q we finish a partial block matrix for qk
        # total blocks are # (seqlen_q // FMAX_STATIONARY) * (seqlen_kv // FMAX_MOVING)
        # we do the operations of attn_fwd_v3 on each block since they are independent row-wise.
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_kv], axis=1)

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1)

        # subtract max from row
        norm_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.tensor_scalar(
                data=qk[i_tile_kv],
                op0=nl.subtract,
                operand0=row_max,
                engine=nisa.vector_engine)

        # exponentiation
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp, data=norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # sum of exp results
        sum_row_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            sum_row_kv[:, i_tile_kv] = nisa.tensor_reduce(
                op=nl.add,
                data=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], axis=1)

        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_kv, axis=1)

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        # has recriprocals of 128 rows at a time, akin to the block of
        # output each q-tile is responsible for.
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        scores = nl.ndarray((nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            scores[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.tensor_scalar(
                data=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)],
                op0=nl.multiply,
                operand0=inverse_sum_row,
                engine=nisa.vector_engine)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(scores[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])
        attn_out[...] = nisa.tensor_copy(attn_out_psum)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


####################################################################
# v5: softmax division delay
####################################################################
@nki.jit
def attn_fwd_v5(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)     # ScalarE

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_kv], axis=1)

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1)

        # subtract max from row
        norm_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.tensor_scalar(
                data=qk[i_tile_kv],
                op0=nl.subtract,
                operand0=row_max,
                engine=nisa.vector_engine)

        # exponentiation
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp, data=norm_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # sum of exp results
        sum_row_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            sum_row_kv[:, i_tile_kv] = nisa.tensor_reduce(
                op=nl.add,
                data=exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)], axis=1)

        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_kv, axis=1)

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        # CHANGE OF LOGIC COMPARED TO attn_fwd_v4, here we delay the division

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        # notice how here the division is done on the final attention output
        # directly comparing to the previous implementation, we save on having to 
        # loop all the i_tile_kvs, meaning we do less divsion operations as our
        # attention block is already collapsed.
        attn_out[...] = nisa.tensor_scalar(data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row, engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out

####################################################################
# v6: instruction combination on ScalarE
####################################################################
@nki.jit
def attn_fwd_v6(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.float32)     # ScalarE

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_kv], axis=1)

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

        # subtract max from row
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.float32, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        
        # We leverage scalar engine's hardware capability of applying reduce after activation
        # with no extra performance cost to compute the max_val subtraction and sum reduction 
        # in one step, saving on extra loops that were previously required.
        #
        # At the same time the vector engine is freed up from compute, giving it more idle time
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk[i_tile_kv],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce
            )
        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles, axis=1)

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.float32)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        attn_out[...] = nisa.tensor_scalar(data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row, engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


####################################################################
# v7: Downcast scores before transpose
# lower precision operations especially on transposes introduce
# higher performance in exchange of small precision loss.
# Furthermore, scalar engine has dtype conversion embedded,
# allowing some conversion cost to be pipelined away before
# going to the tensor engine for transposes.
####################################################################
@nki.jit
def attn_fwd_v7(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF:
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.bfloat16)     # ScalarE

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)

        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            row_max_kv[:, i_tile_kv] = nisa.tensor_reduce(op=nl.max, data=qk[i_tile_kv], axis=1)

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

        # subtract max from row
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk[i_tile_kv],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                dtype=nl.bfloat16
            )
        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles, axis=1)

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.bfloat16, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.bfloat16)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        attn_out[...] = nisa.tensor_scalar(data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row, engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out


####################################################################
# v8: Use tensor_scalar_reduce on VectorE
# In short, this evicts PSUM earlier allowing other Q@K tiles
# to potentially be computed, freeing up the tensor engine to do
# compute. This does lead to a slowdown compared to the v7 kernel, 
# which is the fastest attention kernel we have thus far, but it 
# sets us up for software-pipelining and manual allocation, which
# should outweight the cost penalty.
####################################################################
@nki.jit
def attn_fwd_v8(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax
    FMAX_STATIONARY = nl.tile_size.gemm_stationary_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=q.dtype, buffer=nl.shared_hbm)

    # load inputs into SBUF
    q_sbuf = nl.load(q)
    k_sbuf = nl.load(k)
    v_sbuf = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t = nisa.nc_transpose(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)])          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t, dtype=nl.bfloat16)     # ScalarE

    # Tile along seqlen_q #
    for i_tile_q in nl.affine_range(seqlen_q // FMAX_STATIONARY): # loop on stationary_free
        qk = nl.ndarray((seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                        dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_kv, :, :] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*FMAX_STATIONARY, FMAX_STATIONARY)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
        qk_sbuf = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING, FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)
        row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32, buffer=nl.sbuf)
        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            # previously the entire qk_sbuf row would be processed at once, so PSUM would be occupied for longer
            # here PSUM gets evicted a bit earlier, allowing us to queue the tensor engine earlier as well.
            qk_sbuf[:, i_tile_kv, :] = nisa.tensor_scalar_reduce(data=qk[i_tile_kv], op0=nl.multiply, operand0=1.0,
                                                                    reduce_op=nl.max, reduce_res=row_max_kv[:, i_tile_kv])
                                                                    
        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

        # subtract max from row
        exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                            dtype=nl.bfloat16, buffer=nl.sbuf)
        sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32, buffer=nl.sbuf)

        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk[i_tile_kv],
                bias=row_max,
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                dtype=nl.bfloat16
            )
        sum_row = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles, axis=1)

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row = nisa.reciprocal(data=sum_row)

        # scores has the wrong layout
        scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                    dtype=nl.bfloat16, buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t = nisa.nc_transpose(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)]) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t, dtype=nl.bfloat16)    # ScalarE

        # scores @ V, contract along seqlen_kv
        attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=nl.sbuf)

        attn_out_psum = nl.zeros((PMAX, PMAX), dtype=nl.float32, buffer=nl.psum)

        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])

        attn_out[...] = nisa.tensor_scalar(data=attn_out_psum, op0=nl.multiply,
                                           operand0=inverse_sum_row, engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    return kernel_out

####################################################################
# v8a_2: refactor v8 to prepare for direct allocation
# and software pipelining
####################################################################
@nki.jit
def attn_fwd_v8a(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    identity = nl.shared_constant(np.identity(128, dtype=np.int8), dtype=v.dtype)
    identity_load = nl.ndarray((par_dim(128), 128), dtype=v.dtype,
                               buffer=nl.sbuf)
    identity_load[...] = nl.load(identity)

    identity_bf16 = nisa.tensor_copy(identity_load, dtype=nl.bfloat16)

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    q_sbuf = nl.ndarray((d_head, seqlen_q), dtype=q.dtype, buffer=nl.sbuf)
    k_sbuf = nl.ndarray((d_head, seqlen_kv), dtype=k.dtype, buffer=nl.sbuf)
    v_sbuf = nl.ndarray((d_head, seqlen_kv), dtype=v.dtype, buffer=nl.sbuf)

    # load inputs into SBUF:
    q_sbuf[...] = nl.load(q)
    k_sbuf[...] = nl.load(k)
    v_sbuf[...] = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16,
                          buffer=nl.sbuf)
    v_psum_t = nl.ndarray((seqlen_kv // PMAX, nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                          buffer=nl.psum)

    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t[i_tile_kv] = nisa.nc_matmul(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)], identity_load,
                                             is_transpose=True, is_moving_onezero=True)          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t[i_tile_kv], dtype=nl.bfloat16)     # ScalarE

    num_tile_q = seqlen_q // PMAX
    qk_sbuf = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING, FMAX_MOVING),
                dtype=nl.float32, buffer=nl.sbuf)
    row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                         buffer=nl.sbuf)

    exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                        dtype=nl.bfloat16, buffer=nl.sbuf)
    
    sum_row = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                         buffer=nl.sbuf)
    inverse_sum_row = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                                 buffer=nl.sbuf)

    scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
    attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                          dtype=nl.float32, buffer=nl.sbuf)

    qk = nl.ndarray((num_tile_q, seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                    dtype=nl.float32, buffer=nl.psum)
    scores_psum_t = nl.ndarray((num_tile_q, seqlen_kv // PMAX, nl.par_dim(PMAX), PMAX),
                               dtype=nl.float32, buffer=nl.psum)
    attn_out_psum = nl.ndarray((num_tile_q, nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                               buffer=nl.psum)

    attn_out_sbuf = nl.ndarray((nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                               buffer=nl.sbuf)

    # move into here due to not wanting to continue acc
    sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32,
                       buffer=nl.sbuf)

    # move into here due to want to acc on new buffer, try reduce_cmd if does not work
    row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32,
                    buffer=nl.sbuf)

    def qk_max(i_tile_q):
        # move into here due to want to acc on new buffer
        row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32,
                        buffer=nl.sbuf)
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_q, i_tile_kv] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*PMAX, PMAX)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

            # reduce max along seqlen_k
            qk_sbuf[:, i_tile_kv, :] = nisa.tensor_scalar_reduce(data=qk[i_tile_q, i_tile_kv],
                                                                           op0=nl.multiply, operand0=1.0,
                                                                           reduce_op=nl.max,
                                                                           reduce_res=row_max_kv[:, i_tile_kv])

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

    # subtract max from row
    def exp_row_sum(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk_sbuf[:, i_tile_kv, :],
                bias=row_max[:, :],
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                dtype=nl.bfloat16
                )

    def transpose_scores(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t[i_tile_q, i_tile_kv] = nisa.nc_matmul(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)], identity_bf16) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t[i_tile_q, i_tile_kv], dtype=nl.bfloat16)    # ScalarE

    # scores @ V, contract along seqlen_kv
    def pv_matmul(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum[i_tile_q, :, :] += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                                            moving=v_sbuf_t[:, i_tile_kv, :])
        attn_out_sbuf[:, :] = nisa.tensor_copy(attn_out_psum[i_tile_q, :, :], dtype=nl.float32)

    def write_back(i_tile_q):
        sum_row[:, :] = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles[:, :], axis=1)

        # reciprocal of sum_row, tile shape is [PMAX, 1]
        inverse_sum_row[:, :] = nisa.reciprocal(data=sum_row[:, :])

        attn_out[:, :] = nisa.tensor_scalar(data=attn_out_sbuf[:, :], op0=nl.multiply,
                                        operand0=inverse_sum_row[:, :], engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    # Tile along seqlen_q #
    for i_tile_q in nl.sequential_range(num_tile_q): # loop on stationary_free
        qk_max(i_tile_q)
        exp_row_sum(i_tile_q)
        transpose_scores(i_tile_q)
        pv_matmul(i_tile_q)
        write_back(i_tile_q)

    return kernel_out


sb_mod = nki.compiler.sbuf.mod_alloc
psum_mod = nki.compiler.psum.mod_alloc

class SBufAllocator:
    def __init__(self):
        self.offset = 0

    def get_dtype_size(self, dtype):
        if dtype == nl.float32:
            return 4
        elif dtype == nl.bfloat16:
            return 2
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def allocate(self, size, dtype, num_buffers=1):
        addr = self.offset
        self.offset += size * num_buffers * self.get_dtype_size(dtype)
        return sb_mod(base_addr=addr, num_free_tiles=(num_buffers, ))
        # return nl.sbuf

allocator = SBufAllocator()

####################################################################
# v9: allocation
####################################################################
@nki.compiler.skip_middle_end_transformations
@nki.jit(additional_compile_opt="--internal-skip-backend-allocation-opt-nki --disable-internal-io-dge")
def attn_fwd_v9(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    identity = nl.shared_constant(np.identity(128, dtype=np.int8), dtype=v.dtype)

    identity_load = nl.ndarray((par_dim(128), 128), dtype=v.dtype, buffer=allocator.allocate(size=128, dtype=v.dtype))
    identity_load[...] = nl.load(identity)

    identity_bfloat16 = nl.ndarray((par_dim(128), 128), dtype=nl.bfloat16, buffer=allocator.allocate(size=128, dtype=nl.bfloat16))
    identity_bfloat16[...] = nisa.tensor_copy(identity_load, dtype=nl.bfloat16)

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    q_sbuf = nl.ndarray((d_head, seqlen_q), dtype=q.dtype, buffer=allocator.allocate(size=seqlen_q, dtype=q.dtype))
    k_sbuf = nl.ndarray((d_head, seqlen_kv), dtype=k.dtype, buffer=allocator.allocate(size=seqlen_kv, dtype=k.dtype))
    v_sbuf = nl.ndarray((d_head, seqlen_kv), dtype=v.dtype, buffer=allocator.allocate(size=seqlen_kv, dtype=v.dtype))

    # load inputs into SBUF:
    q_sbuf[...] = nl.load(q)
    k_sbuf[...] = nl.load(k)
    v_sbuf[...] = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16,
                          buffer=allocator.allocate(size=PMAX, num_buffers=seqlen_kv // PMAX, dtype=nl.bfloat16))
    v_psum_t = nl.ndarray((seqlen_kv // PMAX, nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                          buffer=psum_mod(base_bank=0, num_bank_tiles=(8, )))
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t[i_tile_kv] = nisa.nc_matmul(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)], identity_load)          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t[i_tile_kv], dtype=nl.bfloat16)     # ScalarE

    # allocations
    num_tile_q = seqlen_q // PMAX
    qk_sbuf = nl.ndarray((nl.par_dim(PMAX), 2, seqlen_kv // FMAX_MOVING, FMAX_MOVING),
                dtype=nl.float32, buffer=allocator.allocate(size=seqlen_kv, num_buffers=2, dtype=nl.float32))
    row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32,
                            buffer=allocator.allocate(size= seqlen_kv // FMAX_MOVING, dtype=nl.float32))
    row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                         buffer=allocator.allocate(size=1, dtype=nl.float32))
    exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                        dtype=nl.bfloat16, buffer=allocator.allocate(size=seqlen_kv, dtype=nl.bfloat16))
    # want 2 sum_row tiles due to write back and row sum during exp
    sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), 2, seqlen_kv // FMAX_MOVING), dtype=nl.float32,
                               buffer=allocator.allocate(size=seqlen_kv // FMAX_MOVING, num_buffers=2, dtype=nl.float32))
    sum_row = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                         buffer=allocator.allocate(size=1, dtype=nl.float32))
    inverse_sum_row = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                                 buffer=allocator.allocate(size=1, dtype=nl.float32))
    scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                dtype=nl.bfloat16, buffer=allocator.allocate(seqlen_kv, dtype=nl.bfloat16))
    attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                    dtype=nl.float32, buffer=allocator.allocate(PMAX, num_buffers=1, dtype=nl.float32))
    ## --- PSUM START ----
    qk = nl.ndarray((num_tile_q, seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                dtype=nl.float32, buffer=psum_mod(base_bank=0, num_bank_tiles=(1, 7, )))
    scores_psum_t = nl.ndarray((num_tile_q, seqlen_kv // PMAX, nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=psum_mod(base_bank=0, num_bank_tiles=(1, 7, )))
    attn_out_psum = nl.ndarray((num_tile_q, nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                               buffer=psum_mod(base_bank=7, num_bank_tiles=(1, )))
    ## --- PSUM END ----
    attn_out_sbuf = nl.ndarray((nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                               buffer=allocator.allocate(PMAX, num_buffers=1, dtype=nl.float32))

    def qk_max(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_q, i_tile_kv] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*PMAX, PMAX)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
            qk_sbuf[:, i_tile_q % 2, i_tile_kv, :] = nisa.tensor_scalar_reduce(data=qk[i_tile_q, i_tile_kv], op0=nl.multiply, operand0=1.0,
                                                                reduce_op=nl.max, reduce_res=row_max_kv[:, i_tile_kv])

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)
    # subtract max from row
    def exp_row_sum(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk_sbuf[:, i_tile_q % 2, i_tile_kv, :],
                bias=row_max[:, :],
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_q % 2, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                dtype=nl.bfloat16
                )
    # scores has the wrong layout

    def transpose_scores(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t[i_tile_q, i_tile_kv] = nisa.nc_matmul(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)], identity_bfloat16) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t[i_tile_q, i_tile_kv], dtype=nl.bfloat16)    # ScalarE

    # scores @ V, contract along seqlen_kv
    def pv_matmul(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum[i_tile_q, :, :] += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])
        attn_out_sbuf[:, :] = nisa.tensor_copy(attn_out_psum[i_tile_q, :, :], dtype=nl.float32)

    def write_back(i_tile_q):
        sum_row[:, :] = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles[:, i_tile_q % 2, :], axis=1)

        # reciprocal of sum_row [seqlen_q, 1]
        inverse_sum_row[:, :] = nisa.reciprocal(data=sum_row[:, :])

        attn_out[:, :] = nisa.tensor_scalar(data=attn_out_sbuf[:, :], op0=nl.multiply,
                                        operand0=inverse_sum_row[:, :], engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    def fused_qkmax_and_pv(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_q+2, i_tile_kv] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds((i_tile_q+2)*PMAX, PMAX)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])
            qk_sbuf[:, (i_tile_q+2) % 2, i_tile_kv, :] = nisa.tensor_scalar_reduce(data=qk[i_tile_q+2, i_tile_kv],
                                                                             op0=nl.multiply, operand0=1.0,
                                                                             reduce_op=nl.max,
                                                                             reduce_res=row_max_kv[:, i_tile_kv])
            for i_tile_kv_i in nl.affine_range(FMAX_MOVING // PMAX): # loop on contraction
                i_tile_kv_pv = i_tile_kv * FMAX_MOVING // PMAX + i_tile_kv_i
                attn_out_psum[i_tile_q, :, :] += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv_pv, :],
                                                moving=v_sbuf_t[:, i_tile_kv_pv, :])
        attn_out_sbuf[:, :] = nisa.tensor_copy(attn_out_psum[i_tile_q, :, :], dtype=nl.float32)
        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

    # Tile along seqlen_q #
    for i_tile_q in nl.sequential_range(num_tile_q): # loop on stationary_free
        qk_max(i_tile_q)
        exp_row_sum(i_tile_q)
        transpose_scores(i_tile_q)
        pv_matmul(i_tile_q)
        write_back(i_tile_q)

    return kernel_out

####################################################################
# v10: alloc + software pipelining scheduling
# Compiler issue:
# 1. nki.baremetal works, but nki.jit fails compilation
# 2. nki.benchmark and nki.profile also fail
####################################################################
@nki.compiler.skip_middle_end_transformations
@nki.baremetal(additional_compile_opt="--internal-skip-backend-allocation-opt-nki --disable-internal-io-dge",
               save_neff_name="file.neff", save_trace_name="profile.ntff")
def attn_fwd_v10(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    identity = nl.shared_constant(np.identity(128, dtype=np.int8), dtype=v.dtype)

    identity_load = nl.ndarray((par_dim(128), 128), dtype=v.dtype, buffer=allocator.allocate(size=128, dtype=v.dtype))
    identity_load[...] = nl.load(identity)

    identity_bfloat16 = nl.ndarray((par_dim(128), 128), dtype=nl.bfloat16, buffer=allocator.allocate(size=128, dtype=nl.bfloat16))
    identity_bfloat16[...] = nisa.tensor_copy(identity_load, dtype=nl.bfloat16)

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    q_sbuf = nl.ndarray((d_head, seqlen_q), dtype=q.dtype, buffer=allocator.allocate(size=seqlen_q, dtype=q.dtype))
    k_sbuf = nl.ndarray((d_head, seqlen_kv), dtype=k.dtype, buffer=allocator.allocate(size=seqlen_kv, dtype=k.dtype))
    v_sbuf = nl.ndarray((d_head, seqlen_kv), dtype=v.dtype, buffer=allocator.allocate(size=seqlen_kv, dtype=v.dtype))

    # load inputs into SBUF:
    q_sbuf[...] = nl.load(q)
    k_sbuf[...] = nl.load(k)
    v_sbuf[...] = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16,
                          buffer=allocator.allocate(size=PMAX, num_buffers=seqlen_kv // PMAX, dtype=nl.bfloat16))
    v_psum_t = nl.ndarray((seqlen_kv // PMAX, nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                          buffer=psum_mod(base_bank=0, num_bank_tiles=(8, )))
    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t[i_tile_kv] = nisa.nc_matmul(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)], identity_load)          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t[i_tile_kv], dtype=nl.bfloat16)     # ScalarE

    # allocations
    num_tile_q = seqlen_q // PMAX
    qk_sbuf = nl.ndarray((nl.par_dim(PMAX), 2, seqlen_kv // FMAX_MOVING, FMAX_MOVING),
                dtype=nl.float32, buffer=allocator.allocate(size=seqlen_kv, num_buffers=2, dtype=nl.float32))
    row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32,
                            buffer=allocator.allocate(size= seqlen_kv // FMAX_MOVING, dtype=nl.float32))
    row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                         buffer=allocator.allocate(size=1, dtype=nl.float32))
    exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                        dtype=nl.bfloat16, buffer=allocator.allocate(size=seqlen_kv, dtype=nl.bfloat16))
    sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), 2, seqlen_kv // FMAX_MOVING), dtype=nl.float32,
                               buffer=allocator.allocate(size=seqlen_kv // FMAX_MOVING, num_buffers=2, dtype=nl.float32))
    sum_row = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                         buffer=allocator.allocate(size=1, dtype=nl.float32))
    inverse_sum_row = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                                 buffer=allocator.allocate(size=1, dtype=nl.float32))
    scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                dtype=nl.bfloat16, buffer=allocator.allocate(seqlen_kv, dtype=nl.bfloat16))
    attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                    dtype=nl.float32, buffer=allocator.allocate(PMAX, num_buffers=1, dtype=nl.float32))

    ## --- BEGIN PSUM ---
    qk = nl.ndarray((num_tile_q, seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                dtype=nl.float32, buffer=psum_mod(base_bank=0, num_bank_tiles=(1, 7, )))
    scores_psum_t = nl.ndarray((num_tile_q, seqlen_kv // PMAX, nl.par_dim(PMAX), PMAX),
                            dtype=nl.float32, buffer=psum_mod(base_bank=0, num_bank_tiles=(1, 7, )))
    attn_out_psum = nl.ndarray((num_tile_q, nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                               buffer=psum_mod(base_bank=7, num_bank_tiles=(1, )))
    ## --- END PSUM ---
    attn_out_sbuf = nl.ndarray((nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                               buffer=allocator.allocate(PMAX, num_buffers=1, dtype=nl.float32))
    def qk_max(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_q, i_tile_kv] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*PMAX, PMAX)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

        # Softmax #
        # reduce max along seqlen_k
            qk_sbuf[:, i_tile_q % 2, i_tile_kv, :] = nisa.tensor_scalar_reduce(data=qk[i_tile_q, i_tile_kv], op0=nl.multiply, operand0=1.0,
                                                                reduce_op=nl.max, reduce_res=row_max_kv[:, i_tile_kv])

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)
    # subtract max from row
    def exp_row_sum(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk_sbuf[:, i_tile_q % 2, i_tile_kv, :],
                bias=row_max[:, :],
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_q % 2, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                dtype=nl.bfloat16
                )
    # scores has the wrong layout
    def transpose_scores(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t[i_tile_q, i_tile_kv] = nisa.nc_matmul(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)], identity_bfloat16) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t[i_tile_q, i_tile_kv], dtype=nl.bfloat16)    # ScalarE

    # scores @ V, contract along seqlen_kv
    def pv_matmul(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum[i_tile_q, :, :] += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                            moving=v_sbuf_t[:, i_tile_kv, :])
        attn_out_sbuf[:, :] = nisa.tensor_copy(attn_out_psum[i_tile_q, :, :], dtype=nl.float32)

    def write_back(i_tile_q):
        sum_row[:, :] = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles[:, i_tile_q % 2, :], axis=1)

        # reciprocal of sum_row [seqlen_q, 1]
        inverse_sum_row[:, :] = nisa.reciprocal(data=sum_row[:, :])

        attn_out[:, :] = nisa.tensor_scalar(data=attn_out_sbuf[:, :], op0=nl.multiply,
                                        operand0=inverse_sum_row[:, :], engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    def fused_qkmax_and_pv(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_q+2, i_tile_kv] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds((i_tile_q+2)*PMAX, PMAX)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])
            qk_sbuf[:, (i_tile_q+2) % 2, i_tile_kv, :] = nisa.tensor_scalar_reduce(data=qk[i_tile_q+2, i_tile_kv],
                                                                             op0=nl.multiply, operand0=1.0,
                                                                             reduce_op=nl.max,
                                                                             reduce_res=row_max_kv[:, i_tile_kv])
            for i_tile_kv_i in nl.affine_range(FMAX_MOVING // PMAX): # loop on contraction
                i_tile_kv_pv = i_tile_kv * FMAX_MOVING // PMAX + i_tile_kv_i
                attn_out_psum[i_tile_q, :, :] += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv_pv, :],
                                                moving=v_sbuf_t[:, i_tile_kv_pv, :])
        attn_out_sbuf[:, :] = nisa.tensor_copy(attn_out_psum[i_tile_q, :, :], dtype=nl.float32)
        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)


    # Tile along seqlen_q #
    qk_max(0)
    exp_row_sum(0)
    transpose_scores(0)
    qk_max(1)
    for i_tile_q in nl.sequential_range(num_tile_q - 2, directives=nki.compiler.no_reorder()): # loop on stationary_free
        exp_row_sum(i_tile_q+1)
        fused_qkmax_and_pv(i_tile_q)
        transpose_scores(i_tile_q+1)
        write_back(i_tile_q)
    pv_matmul(num_tile_q-2)
    write_back(num_tile_q-2)
    exp_row_sum(num_tile_q-1)
    transpose_scores(num_tile_q-1)
    pv_matmul(num_tile_q-1)
    write_back(num_tile_q-1)

    return kernel_out
    
####################################################################
# v11: software pipelining scheduling only
####################################################################
@nki.jit
def attn_fwd_v11(q, k, v):
    d_head, seqlen_q = q.shape
    seqlen_kv = seqlen_q

    PMAX = nl.tile_size.pmax
    FMAX_MOVING = nl.tile_size.gemm_moving_fmax

    assert q.shape == k.shape == v.shape
    assert d_head == PMAX
    assert seqlen_q >= 512

    identity = nl.shared_constant(np.identity(128, dtype=np.int8), dtype=v.dtype)
    identity_load = nl.ndarray((par_dim(128), 128), dtype=v.dtype,
                               buffer=nl.sbuf)
    identity_load[...] = nl.load(identity)

    identity_bf16 = nisa.tensor_copy(identity_load, dtype=nl.bfloat16)

    kernel_out = nl.ndarray((seqlen_q, d_head), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    q_sbuf = nl.ndarray((d_head, seqlen_q), dtype=q.dtype, buffer=nl.sbuf)
    k_sbuf = nl.ndarray((d_head, seqlen_kv), dtype=k.dtype, buffer=nl.sbuf)
    v_sbuf = nl.ndarray((d_head, seqlen_kv), dtype=v.dtype, buffer=nl.sbuf)

    # load inputs into SBUF:
    q_sbuf[...] = nl.load(q)
    k_sbuf[...] = nl.load(k)
    v_sbuf[...] = nl.load(v)

    # v has the wrong layout
    v_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX), dtype=nl.bfloat16,
                          buffer=nl.sbuf)
    v_psum_t = nl.ndarray((seqlen_kv // PMAX, nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                          buffer=nl.psum)

    for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
        v_psum_t[i_tile_kv] = nisa.nc_matmul(v_sbuf[:, nl.ds(i_tile_kv*PMAX, PMAX)], identity_load,
                                             is_transpose=True, is_moving_onezero=True)          # TensorE
        v_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(v_psum_t[i_tile_kv], dtype=nl.bfloat16)     # ScalarE

    num_tile_q = seqlen_q // PMAX
    qk_sbuf = nl.ndarray((nl.par_dim(PMAX), 2, seqlen_kv // FMAX_MOVING, FMAX_MOVING),
                dtype=nl.float32, buffer=nl.sbuf)
    row_max_kv = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // FMAX_MOVING), dtype=nl.float32,
                            buffer=nl.sbuf)
    row_max = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                         buffer=nl.sbuf)

    exp_row = nl.ndarray((nl.par_dim(PMAX), seqlen_kv),
                        dtype=nl.bfloat16, buffer=nl.sbuf)
    sum_row_tiles = nl.ndarray((nl.par_dim(PMAX), 2, seqlen_kv // FMAX_MOVING), dtype=nl.float32,
                               buffer=nl.sbuf)
    sum_row = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                         buffer=nl.sbuf)
    inverse_sum_row = nl.ndarray((nl.par_dim(PMAX), 1), dtype=nl.float32,
                                 buffer=nl.sbuf)

    scores_sbuf_t = nl.ndarray((nl.par_dim(PMAX), seqlen_kv // PMAX, PMAX),
                                dtype=nl.bfloat16, buffer=nl.sbuf)
    attn_out = nl.ndarray((nl.par_dim(PMAX), PMAX),
                          dtype=nl.float32, buffer=nl.sbuf)

    # START PSUM

    qk = nl.ndarray((num_tile_q, seqlen_kv // FMAX_MOVING, nl.par_dim(PMAX), FMAX_MOVING),
                    dtype=nl.float32, buffer=nl.psum)
    scores_psum_t = nl.ndarray((num_tile_q, seqlen_kv // PMAX, nl.par_dim(PMAX), PMAX),
                               dtype=nl.float32, buffer=nl.psum)
    attn_out_psum = nl.ndarray((num_tile_q, nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                               buffer=nl.psum)
    # END PSUM
    
    attn_out_sbuf = nl.ndarray((nl.par_dim(PMAX), PMAX), dtype=nl.float32,
                               buffer=nl.sbuf)

    def qk_max(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_q, i_tile_kv] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds(i_tile_q*PMAX, PMAX)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])

            # reduce max along seqlen_k
            qk_sbuf[:, i_tile_q % 2, i_tile_kv, :] = nisa.tensor_scalar_reduce(data=qk[i_tile_q, i_tile_kv],
                                                                           op0=nl.multiply, operand0=1.0,
                                                                           reduce_op=nl.max,
                                                                           reduce_res=row_max_kv[:, i_tile_kv])

        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

    # subtract max from row
    def exp_row_sum(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING):
            exp_row[:, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)] = nisa.activation(
                op=nl.exp,
                data=qk_sbuf[:, i_tile_q % 2, i_tile_kv, :],
                bias=row_max[:, :],
                reduce_op=nl.add,
                reduce_res=sum_row_tiles[:, i_tile_q % 2, i_tile_kv],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
                dtype=nl.bfloat16
                )

    def transpose_scores(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX):
            scores_psum_t[i_tile_q, i_tile_kv] = nisa.nc_matmul(exp_row[:, nl.ds(i_tile_kv*PMAX, PMAX)], identity_bf16) # TensorE
            scores_sbuf_t[:, i_tile_kv, :] = nisa.tensor_copy(scores_psum_t[i_tile_q, i_tile_kv], dtype=nl.bfloat16)    # ScalarE

    # scores @ V, contract along seqlen_kv
    def pv_matmul(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // PMAX): # loop on contraction
            attn_out_psum[i_tile_q, :, :] += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv, :],
                                                            moving=v_sbuf_t[:, i_tile_kv, :])
        attn_out_sbuf[:, :] = nisa.tensor_copy(attn_out_psum[i_tile_q, :, :], dtype=nl.float32)

    def write_back(i_tile_q):
        sum_row[:, :] = nisa.tensor_reduce(op=nl.add, data=sum_row_tiles[:, i_tile_q % 2, :], axis=1)

        # reciprocal of sum_row [seqlen_q, 1]
        inverse_sum_row[:, :] = nisa.reciprocal(data=sum_row[:, :])

        attn_out[:, :] = nisa.tensor_scalar(data=attn_out_sbuf[:, :], op0=nl.multiply,
                                        operand0=inverse_sum_row[:, :], engine=nisa.vector_engine)

        # store output
        nl.store(dst=kernel_out[nl.ds(i_tile_q*PMAX, PMAX), :], value=attn_out[:, :])

    def fused_qkmax_and_pv(i_tile_q):
        for i_tile_kv in nl.affine_range(seqlen_kv // FMAX_MOVING): # loop on moving_free
            # Q @ K, contract along d_head #
            qk[i_tile_q+2, i_tile_kv] = nisa.nc_matmul(
                stationary=q_sbuf[0:PMAX, nl.ds((i_tile_q+2)*PMAX, PMAX)],
                moving=k_sbuf[0:PMAX, nl.ds(i_tile_kv*FMAX_MOVING, FMAX_MOVING)])
            qk_sbuf[:, (i_tile_q+2) % 2, i_tile_kv, :] = nisa.tensor_scalar_reduce(data=qk[i_tile_q+2, i_tile_kv],
                                                                             op0=nl.multiply, operand0=1.0,
                                                                             reduce_op=nl.max,
                                                                             reduce_res=row_max_kv[:, i_tile_kv])
            for i_tile_kv_i in nl.affine_range(FMAX_MOVING // PMAX): # loop on contraction
                i_tile_kv_pv = i_tile_kv * FMAX_MOVING // PMAX + i_tile_kv_i
                attn_out_psum[i_tile_q, :, :] += nisa.nc_matmul(stationary=scores_sbuf_t[:, i_tile_kv_pv, :],
                                                moving=v_sbuf_t[:, i_tile_kv_pv, :])
        attn_out_sbuf[:, :] = nisa.tensor_copy(attn_out_psum[i_tile_q, :, :], dtype=nl.float32)
        row_max[:, :] = nisa.tensor_reduce(op=nl.max, data=row_max_kv[:, :], axis=1, negate=True)

    # Tile along seqlen_q #
    qk_max(0)
    exp_row_sum(0)
    transpose_scores(0)
    qk_max(1)
    for i_tile_q in nl.sequential_range(num_tile_q - 2, directives=nki.compiler.no_reorder()): # loop on stationary_free
        exp_row_sum(i_tile_q+1)
        fused_qkmax_and_pv(i_tile_q)
        transpose_scores(i_tile_q+1)
        write_back(i_tile_q)
    pv_matmul(num_tile_q-2)
    write_back(num_tile_q-2)
    exp_row_sum(num_tile_q-1)
    transpose_scores(num_tile_q-1)
    pv_matmul(num_tile_q-1)
    write_back(num_tile_q-1)

    return kernel_out
