################################################################
# NKI Kernels to demonstrate MX usage
################################################################

import nki
import nki.isa as nisa
import nki.language as nl
from mx_kernel_utils import load_scales_scattered, allocate_mx_tiles, copy_data_strided

# [start-kernel_offline_quantized_mx_matmul]
# Matmul-MX using offline-quantized input tiles in HBM, assumed to be maximum tile sizes for the TensorE.
# MX layout requirements for data tiles are ignored. (i.e. it's assumed the data tiles are 
# already correctly laid out).
# *_mx_data inputs mimic _x4 packed types via uint. This kernel will simply view it as _x4.
# *_mx_scale inputs are uint8, with scales packed contiguous (this kernel will spread them across partition-dim).
# mx_dtype = one of nl.float8_e5m2_x4, nl.float8_e4m3fn_x4, nl.float4_e2m1fn_x4.
# Returns bfloat16 matmul result.
@nki.jit
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
  nisa.dma_copy(dst=stationary_mx_data_sbuf_x4, src=stationary_mx_data_hbm_x4)
  stationary_mx_scale_sbuf = load_scales_scattered(stationary_mx_data_sbuf_x4, stationary_mx_scale)

  # Load moving
  moving_mx_data_sbuf_x4 = nl.ndarray(moving_mx_data_hbm_x4.shape, dtype=mx_dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=moving_mx_data_sbuf_x4, src=moving_mx_data_hbm_x4)
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
  nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=nl.bfloat16)  

  # Store to HBM
  result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.hbm)  
  nisa.dma_copy(dst=result_hbm, src=result_sbuf)
  
  return result_hbm
# [end-kernel_offline_quantized_mx_matmul]

# [start-kernel_on_device_quantize_matmul_mx]
# Matmul-MX using a offline-quantized stationary input tile from HBM and on-device quantized moving tile.
# Input to Quantize-MX must be bf16/fp16.
# MX layout requirements for data tiles are ignored. (i.e. it's assumed the data tiles are 
# already correctly laid out, including moving_data_bf16).
# *_mx_data inputs are float32 where each element contains 4 x quantized elements elements.
#   *_mx_data will be viewed as mx_dtype.
# *_mx_scale inputs are uint8, with scales packed contiguous (this kernel will spread them across partition-dim).
# mx_dtype = one of nl.float8_e5m2_x4, nl.float8_e4m3fn_x4, nl.float4_e2m1fn_x4.
# It's assumed TensorE max tile sizes are used.
@nki.jit
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
  nisa.dma_copy(dst=stationary_mx_data_sbuf_x4, src=stationary_mx_data_hbm_x4)
  stationary_mx_scale_sbuf = load_scales_scattered(stationary_mx_data_sbuf_x4, stationary_mx_scale)
  
  # Load moving BF16
  moving_bf16_sbuf = nl.ndarray(moving_data_bf16.shape, dtype=moving_data_bf16.dtype, buffer=nl.sbuf)
  nisa.dma_copy(dst=moving_bf16_sbuf, src=moving_data_bf16)

  # Allocate quantized moving tiles
  moving_mx_data_sbuf_x4, moving_mx_scale_sbuf = allocate_mx_tiles(moving_data_bf16.shape, moving_mx_dtype)  

  # Quantize-MX. Scales will automatically be spread across partition-dim quadrants.
  nisa.quantize_mx(dst=moving_mx_data_sbuf_x4,
                  src=moving_bf16_sbuf,
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
  nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=nl.bfloat16)  

  # Store to HBM
  result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.hbm)  
  nisa.dma_copy(dst=result_hbm, src=result_sbuf)

  return result_hbm
# [end-kernel_on_device_quantize_matmul_mx]

# Matmul-MX using on-device quantized stationary and moving tensors, demonstrating how to use
# a strided access pattern to establish the SBUF layout required by MX operations.
# Two examples are shown: the access pattern is implemented either in VectorE/ScalarE Tensor Copy or by the DMA engine.
# Unquantized input tiles from HBM are expected to be sized such that they become max-tiles for the 
# TensorE once quantized.
@nki.jit
def kernel_copy_strided_quantize_matmul_mx(stationary_hbm, moving_hbm, mx_dtype, use_tensor_copy: bool = True):
  
  assert mx_dtype != nl.float4_e2m1fn_x4, "FP4 not supported by Quantize-MX"
 
  MAX_TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  MAX_TILE_K = nl.tile_size.pmax  # 128
  MAX_TILE_N = nl.tile_size.gemm_moving_fmax  # 512  

  # Ensure input tensors are in HBM.
  assert stationary_hbm.buffer == moving_hbm.buffer == nl.hbm

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
  nisa.quantize_mx(dst=stationary_mx_data_sbuf,
                  src=stationary_sbuf_strided,
                  dst_scale=stationary_mx_scale_sbuf)

  nisa.quantize_mx(dst=moving_mx_data_sbuf,
                  src=moving_sbuf_strided,
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
  nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=nl.bfloat16)  

  # Store to HBM
  result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.hbm)  
  nisa.dma_copy(dst=result_hbm, src=result_sbuf)

  return result_hbm

#[start-kernel_copy_strided_quantize_matmul_mx_packed_scale]
# Matmul-MX using on-device quantized stationary and moving tensors, demonstrating how to use
# pack scale values from multiple quantize_mx calls into a single tensor in SBUF.
# 
# Unquantized input tiles from HBM are expected to be sized such that they become max-tiles for the 
# TensorE once quantized.
@nki.jit
def kernel_copy_strided_quantize_matmul_mx_packed_scale(stationary_hbm, moving_hbm, mx_dtype, use_tensor_copy: bool = True):
  
  assert mx_dtype != nl.float4_e2m1fn_x4, "FP4 not supported by Quantize-MX"
 
  MAX_TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
  MAX_TILE_K = nl.tile_size.pmax  # 128
  MAX_TILE_N = nl.tile_size.gemm_moving_fmax  # 512  

  # Ensure input tensors are in HBM.
  assert stationary_hbm.buffer == moving_hbm.buffer == nl.hbm

  # Sanity check the shapes. We expect contraction dimension of the unquantized tile to be 4x.
  assert stationary_hbm.shape == (MAX_TILE_K*4, MAX_TILE_M)
  assert moving_hbm.shape == (MAX_TILE_K*4, MAX_TILE_N)

  # Use strided access patterns to achieve required MX layout.
  # Returned shape is [P//4, F*4] where [P,F] is the input shape.
  stationary_sbuf_strided, moving_sbuf_strided = copy_data_strided(stationary_hbm, moving_hbm, use_tensor_copy)

  # Allocate quantized stationary/moving tiles.
  # Unlike the example kernel_copy_strided_quantize_matmul_mx, we do not allocate scale tiles here.
  stationary_mx_data_sbuf, _  = allocate_mx_tiles(stationary_sbuf_strided.shape, mx_dtype, alloc_scale=False)
  moving_mx_data_sbuf, _ = allocate_mx_tiles(moving_sbuf_strided.shape, mx_dtype, alloc_scale=False)

  # Allocate a single tile into which we will pack scale values from BOTH quantize_mx calls.
  #
  # quantize_mx requires that the input tile's free dimension contains exactly 4x as many 
  # elements as the scale tile. We will use this tile for both quantize_mx calls, so its 
  # free dimension needs to be able to hold the larger of the two input tiles, hence MAX_TILE_N.
  packed_mx_scale_sbuf = nl.ndarray((MAX_TILE_K, MAX_TILE_N), dtype=nl.uint8, buffer=nl.sbuf)

  # Each scaling group consists of 32 elements, with 8 partitions x 4 elements per partition.
  # Therefore, for each 32-partition SBUF quadrant, we get only 32 // 8 = 4 partitions' worth of scale factors.
  # This leaves 28 partitions unused. quantize_mx lets us use some of this space by storing other tensors'
  # scale factors at an offset.

  # In this example, we use tensor slicing to store:
  # - stationary's scale values at offset 0 in each quadrant (i.e., partitions 0:4, 32:36, 64:68, 96:100)
  # - moving's scale values at offset 4 in each quadrant (i.e., partitions 4:8, 36:40, 68:72, 100:104)

  # moving's scale values will be written to partitions 0:4 in each quadrant.
  # Additionally, we restrict the free dimension size to match stationary's shape.
  stationary_mx_scale_sbuf = packed_mx_scale_sbuf[0:, :MAX_TILE_M]

  # moving's scale values will be written to partitions 4:8 in each quadrant.
  # We don't restrict the size of the free dimension; it already matches moving's shape.
  moving_mx_scale_sbuf = packed_mx_scale_sbuf[4:, :]

  # Quantize-MX. Scales will automatically be spread across partition-dim quadrants.
  nisa.quantize_mx(dst=stationary_mx_data_sbuf,
                  src=stationary_sbuf_strided,
                  dst_scale=stationary_mx_scale_sbuf)

  nisa.quantize_mx(dst=moving_mx_data_sbuf,
                  src=moving_sbuf_strided,
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
  nisa.tensor_copy(dst=result_sbuf, src=result_psum, dtype=nl.bfloat16)  

  # Store to HBM
  result_hbm = nl.ndarray(result_psum.shape, dtype=nl.bfloat16, buffer=nl.hbm)  
  nisa.dma_copy(dst=result_hbm, src=result_sbuf)

  return result_hbm
#[end-kernel_copy_strided_quantize_matmul_mx_packed_scale]