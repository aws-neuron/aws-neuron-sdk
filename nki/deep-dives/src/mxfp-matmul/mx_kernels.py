################################################################
# NKI Kernels to demonstrate MX usage
################################################################

import nki
import nki.isa as nisa
import nki.language as nl
from mx_kernel_utils import load_scales_scattered, allocate_mx_tiles, copy_data_strided

# Matmul-MX using offline-quantized input tiles in HBM, assumed to be maximum tile sizes for the TensorE.
# MX layout requirements for data tiles are ignored. (i.e. it's assumed the data tiles are 
# already correctly laid out).
# *_mx_data inputs mimic _x4 packed types via uint. This kernel will simply view it as _x4.
# *_mx_scale inputs are uint8, with scales packed contiguous (this kernel will spread them across partition-dim).
# mx_dtype = one of nl.float8_e5m2_x4, nl.float8_e4m3fn_x4, nl.float4_e2m1fn_x4.
# Returns bfloat16 matmul result.
@nki.jit(platform_target="trn3")
def kernel_offline_quantized_mx_matmul(stationary_mx_data, stationary_mx_scale, moving_mx_data, moving_mx_scale, mx_dtype):    
  
  MAX_TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  MAX_TILE_K = nl.tile_size.pmax  # 128
  MAX_TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # View the input data as _x4 mx_dtype. This is done using an access pattern, specifying the target dtype and a simple
  # linear pattern.
  stationary_mx_data_hbm_x4 = stationary_mx_data.ap(dtype=mx_dtype, pattern=[[MAX_TILE_M,MAX_TILE_K],[1,MAX_TILE_M]], offset=0)
  moving_mx_data_hbm_x4 = moving_mx_data.ap(dtype=mx_dtype, pattern=[[MAX_TILE_N,MAX_TILE_K],[1,MAX_TILE_N]], offset=0)

  # Check that the input tiles are max-sized. This is merely for simplicity of the example but
  # smaller shapes are also supported.
  assert stationary_mx_data_hbm_x4.shape == (MAX_TILE_K, MAX_TILE_M)
  assert moving_mx_data_hbm_x4.shape == (MAX_TILE_K, MAX_TILE_N)

  # Load inputs directly from HBM to SBUF. Data is assumed to already have the 
  # layout required by MX. Scales are assumed to be contiguous in HBM therefore we use
  # load_scales_scattered() to spread them across SBUF partition-dim quadrants, as is required
  # by Matmul-MX.

  stationary_mx_data_sbuf_x4 = nl.ndarray(stationary_mx_data_hbm_x4.shape, dtype=mx_dtype, buffer=nl.sbuf)
  nisa.dma_copy(src=stationary_mx_data_hbm_x4, dst=stationary_mx_data_sbuf_x4)
  stationary_mx_scale_sbuf = load_scales_scattered(stationary_mx_data_sbuf_x4, stationary_mx_scale)

  # Load moving
  moving_mx_data_sbuf_x4 = nl.ndarray(moving_mx_data_hbm_x4.shape, dtype=mx_dtype, buffer=nl.sbuf)
  nisa.dma_copy(src=moving_mx_data_hbm_x4, dst=moving_mx_data_sbuf_x4)
  moving_mx_scale_sbuf = load_scales_scattered(moving_mx_data_sbuf_x4, moving_mx_scale)
  
  # Allocate a tile in PSUM. This could also be float32.
  result_psum = nl.ndarray((MAX_TILE_M, MAX_TILE_N), dtype=nl.bfloat16, buffer=nl.psum)

  # Matmul-MX
  nisa.nc_matmul_mx(
    dst=result_psum,
    stationary=stationary_mx_data_sbuf_x4,
    moving=moving_mx_data_sbuf_x4,
    stationary_scale=stationary_mx_scale_sbuf,
    moving_scale=moving_mx_scale_sbuf
  )

  # Copy the PSUM result back to SBUF
  result_sbuf = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
  nisa.tensor_copy(src=result_psum, dst=result_sbuf, dtype=nl.bfloat16)  

  # Store to HBM
  result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)  
  nisa.dma_copy(src=result_sbuf, dst=result_hbm)
  
  return result_hbm

# Matmul-MX using a offline-quantized stationary input tile from HBM and on-device quantized moving tile.
# Input to Quantize-MX must be bf16/fp16.
# MX layout requirements for data tiles are ignored. (i.e. it's assumed the data tiles are 
# already correctly laid out, including moving_data_bf16).
# *_mx_data inputs are float32 where each element contains 4 x quantized elements elements.
#   *_mx_data will be viewed as mx_dtype.
# *_mx_scale inputs are uint8, with scales packed contiguous (this kernel will spread them across partition-dim).
# mx_dtype = one of nl.float8_e5m2_x4, nl.float8_e4m3fn_x4, nl.float4_e2m1fn_x4.
# It's assumed TensorE max tile sizes are used.
@nki.jit(platform_target="trn3")
def kernel_on_device_quantize_matmul_mx(stationary_mx_data, stationary_mx_scale, moving_data_bf16, stationary_mx_dtype, moving_mx_dtype):

  assert moving_mx_dtype != nl.float4_e2m1fn_x4, "FP4 not supported by Quantize-MX"

  MAX_TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  MAX_TILE_K = nl.tile_size.pmax  # 128
  MAX_TILE_N = nl.tile_size.gemm_moving_fmax  # 512

  # View the input MX data as _x4 mx_dtype. This is done using an access pattern, specifying the target dtype and a simple
  # linear pattern.
  stationary_mx_data_hbm_x4 = stationary_mx_data.ap(dtype=stationary_mx_dtype, pattern=[[MAX_TILE_M,MAX_TILE_K],[1,MAX_TILE_M]], offset=0)

  # Check that the input tiles are max-sized. This is merely for simplicity of the example but
  # smaller shapes are also supported.
  assert stationary_mx_data_hbm_x4.shape == (MAX_TILE_K, MAX_TILE_M)
  # Note the factor of 4 on the N free-dim. This is unquantized data whose free-dim will be packed and
  # reduced by a factor of 4 during quantize_mx.
  assert moving_data_bf16.shape == (MAX_TILE_K, MAX_TILE_N*4)

  # Load stationary MX.
  stationary_mx_data_sbuf_x4 = nl.ndarray(stationary_mx_data_hbm_x4.shape, dtype=stationary_mx_dtype, buffer=nl.sbuf)
  nisa.dma_copy(src=stationary_mx_data_hbm_x4, dst=stationary_mx_data_sbuf_x4)
  stationary_mx_scale_sbuf = load_scales_scattered(stationary_mx_data_sbuf_x4, stationary_mx_scale)
  
  # Load moving BF16
  moving_bf16_sbuf = nl.ndarray(moving_data_bf16.shape, dtype=moving_data_bf16.dtype, buffer=nl.sbuf)
  nisa.dma_copy(src=moving_data_bf16, dst=moving_bf16_sbuf)

  # Allocate quantized moving tiles
  moving_mx_data_sbuf_x4, moving_mx_scale_sbuf = allocate_mx_tiles(moving_data_bf16.shape, moving_mx_dtype)  

  # Quantize-MX. Scales will automatically be spread across partition-dim quadrants.
  nisa.quantize_mx(src=moving_bf16_sbuf,
                  dst=moving_mx_data_sbuf_x4, 
                  dst_scale=moving_mx_scale_sbuf)  

  # Allocate a tile in PSUM
  result_psum = nl.ndarray((MAX_TILE_M, MAX_TILE_N), dtype=nl.bfloat16, buffer=nl.psum)

  # Matmul-MX
  nisa.nc_matmul_mx(
    dst=result_psum,
    stationary=stationary_mx_data_sbuf_x4,
    moving=moving_mx_data_sbuf_x4,
    stationary_scale=stationary_mx_scale_sbuf,
    moving_scale=moving_mx_scale_sbuf
  )  

  # Copy the PSUM result back to SBUF
  result_sbuf = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
  nisa.tensor_copy(src=result_psum, dst=result_sbuf, dtype=nl.bfloat16)  

  # Store to HBM
  result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)  
  nisa.dma_copy(src=result_sbuf, dst=result_hbm)

  return result_hbm

# Matmul-MX using on-device quantized stationary and moving tensors, demonstrating how to use
# a strided access pattern to establish the SBUF layout required by MX operations.
# Two examples are shown: the access pattern is implemented either in VectorE/ScalarE Tensor Copy or by the DMA engine.
# Unquantized input tiles from HBM are expected to be sized such that they become max-tiles for the 
# TensorE once quantized.
@nki.jit(platform_target="trn3")
def kernel_copy_strided_quantize_matmul_mx(stationary_hbm, moving_hbm, mx_dtype, use_tensor_copy: bool = True):
  
  assert mx_dtype != nl.float4_e2m1fn_x4, "FP4 not supported by Quantize-MX"
 
  MAX_TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  MAX_TILE_K = nl.tile_size.pmax  # 128
  MAX_TILE_N = nl.tile_size.gemm_moving_fmax  # 512  

  # Sanity check the shapes. We expect contraction dimension of the unquantized tile to be 4x.
  assert stationary_hbm.shape == (MAX_TILE_K*4, MAX_TILE_M)
  assert moving_hbm.shape == (MAX_TILE_K*4, MAX_TILE_N)

  # The key details of this example are shown in copy_data_strided() where data is copied into SBUF
  # using strided access patterns to achieve the required MX layout.
  # Returned shape is [P//4, F*4] where [P,F] is the input shape.
  stationary_sbuf_strided, moving_sbuf_strided = copy_data_strided(stationary_hbm, moving_hbm, use_tensor_copy)

  # Allocate quantized moving tiles
  stationary_mx_data_sbuf, stationary_mx_scale_sbuf = allocate_mx_tiles(stationary_sbuf_strided.shape, mx_dtype)
  moving_mx_data_sbuf, moving_mx_scale_sbuf = allocate_mx_tiles(moving_sbuf_strided.shape, mx_dtype)

  # Quantize-MX. Scales will automatically be spread across partition-dim quadrants.
  nisa.quantize_mx(src=stationary_sbuf_strided,
                  dst=stationary_mx_data_sbuf, 
                  dst_scale=stationary_mx_scale_sbuf)

  nisa.quantize_mx(src=moving_sbuf_strided,
                  dst=moving_mx_data_sbuf, 
                  dst_scale=moving_mx_scale_sbuf)
  
  # Allocate a tile in PSUM
  result_psum = nl.ndarray((MAX_TILE_M, MAX_TILE_N), dtype=nl.bfloat16, buffer=nl.psum)

  # Matmul-MX
  nisa.nc_matmul_mx(
    dst=result_psum,
    stationary=stationary_mx_data_sbuf,
    moving=moving_mx_data_sbuf,
    stationary_scale=stationary_mx_scale_sbuf,
    moving_scale=moving_mx_scale_sbuf
  )

  # Copy the PSUM result back to SBUF
  result_sbuf = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.sbuf)
  nisa.tensor_copy(src=result_psum, dst=result_sbuf, dtype=nl.bfloat16)  

  # Store to HBM
  result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.shared_hbm)  
  nisa.dma_copy(src=result_sbuf, dst=result_hbm)

  return result_hbm