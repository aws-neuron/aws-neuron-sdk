################################################################
# NKI Kernel helper utilities for using MX
################################################################

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

# data_hbm = MX data tile, dtype=*_x4, in HBM. dim[0] must be multiple of 32.
# scale_hbm = MX scale tile, dtype=*_x4, in HBM, contiguous.
# Returns SBUF tile with scales spread across P-dim quadrants as follows:
# HBM Scale:      →     Physical SBUF Layout:
# [0:4,   :]      →     Quadrant 0: partitions [0:4,   :]
# [4:8,   :]      →     Quadrant 1: partitions [32:36, :]
# [8:12,  :]      →     Quadrant 2: partitions [64:68, :]
# [12:16, :]      →     Quadrant 3: partitions [96:100, :]
def load_scales_scattered(data_hbm, scale_hbm):
  # As per nc_matmul_mx's SBUF input layout rules, we need to spread the scales across the partition-dimension.

  # P dimension must be multiple of 32 and not exceed 128
  data_p, _ = data_hbm.shape
  assert data_p % 32 == 0, f"Data tile P={data_p} must be divisible by 32 for MX. Apply padding."
  assert data_p <= 128, f"Data tile P={data_p} must be <= 128."
  
  scale_p, scale_f = scale_hbm.shape
  # This should automatically be true, but just sanity check.
  assert (scale_p == data_p//8), f"Scale tile P={scale_p} must be Data tile P//8 (data_p={data_p}), for MX." 

  # We only need to scatter the scales if more than one SBUF quadrant is used.
  if (data_p > 32): # Could also check (scale_p > 4)
    # Allocate expanded scale tile. Notice here we match the P-dim of the data tile.
    scale_sbuf = nl.ndarray((data_p, scale_f), dtype=scale_hbm.dtype, buffer=nl.sbuf)
    nisa.memset(dst=scale_sbuf,value=0)
 
    # Take each group of 4 scale rows from HBM and write them to the respective SBUF quadrant, where SBUF quadrants
    # are 32-rows.
    for q in range (scale_p // 4):
      # .ap(pattern) tuple of [step_size, count], right-most is the inner (fastest changing) dimension of the access pattern (AP)
      # The src AP reads scale_f elements, jumps to the next row, 4 times total. 
      # Outer for-loop sets the src AP start offset to be the first of a set of 4 rows.
      # The dst AP also writes scale_f elements, jumps to the next row, 4 times total.
      # But the start-offset is the first of a set of 32 rows in dst.
      nisa.dma_copy(
        src=scale_hbm.ap(pattern=[[scale_f, 4], [1, scale_f]],offset=(4*q)*scale_f),
        dst=scale_sbuf.ap(pattern=[[scale_f, 4], [1, scale_f]],offset=(32*q)*scale_f)        
      )

  else:
    # Allocate scale tile. Notice here we use scale_p directly since scales will fit into one quadrant.
    scale_sbuf = nl.ndarray((scale_p, scale_f), dtype=scale_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(src=scale_hbm, dst=scale_sbuf) # Straight copy

  return scale_sbuf

# Expected input tile shapes: stationary_hbm [4, P_st, F_st], moving_hbm [4, P_mv, F_mv]
# Output SBUF shapes: stationary_sbuf [P_st, 4, F_st], moving_sbuf [P_mv, 4, F_mv]
#
# HBM Layout [4, P, F]:           SBUF Layout [P, 4, F]:
# =====================           ======================
# ┌───────────┐                   ┌─────────┬─────────┬─────────┬─────────┐
# │           │                   │         │         │         │         │
# │ Tile0     │                   │  Tile0  │  Tile1  │  Tile2  │  Tile3  │
# │ [P,F]     │                   │  [P,F]  │  [P,F]  │  [P,F]  │  [P,F]  │
# │           │                   │         │         │         │         │
# ├───────────┤                   └─────────┴─────────┴─────────┴─────────┘
# │           │
# │ Tile1     │
# │ [P,F]     │
# │           │
# ├───────────┤
# │           │
# │ Tile2     │
# │ [P,F]     │
# │           │
# ├───────────┤
# │           │
# │ Tile3     │
# │ [P,F]     │
# │           │
# └───────────┘
def load_tensor_helper(stationary_hbm, moving_hbm):
  P_st = stationary_hbm.shape[1]
  F_st = stationary_hbm.shape[2]
  P_mv = moving_hbm.shape[1]
  F_mv = moving_hbm.shape[2]
  
  stationary_sbuf = nl.ndarray((P_st, 4, F_st), dtype=stationary_hbm.dtype, buffer=nl.sbuf)
  moving_sbuf = nl.ndarray((P_mv, 4, F_mv), dtype=moving_hbm.dtype, buffer=nl.sbuf)
  
  # .ap(pattern) tuple of [step_size, count], right-most is the inner (fastest changing) dimension of the access pattern (AP).
  # dst (SBUF) does not have an AP specified which means it is linearly accessed.
  # The src AP reads F elements, then jumps to the next Tile, 4 times. This supplies the data to fill one row of SBUF.
  #   Then we jump to the next row of HBM and repeat.

  nisa.dma_copy(src=stationary_hbm.ap(pattern=[[F_st, P_st], [P_st*F_st, 4], [1, F_st]], offset=0), dst=stationary_sbuf)
  nisa.dma_copy(src=moving_hbm.ap(pattern=[[F_mv, P_mv], [P_mv*F_mv, 4], [1, F_mv]], offset=0), dst=moving_sbuf)

  return stationary_sbuf, moving_sbuf

# shape_unquantized represents the 2D unquantized SBUF shape with interleaved
# layout established (i.e. the shape immediately before calling Quantize-MX).
def allocate_mx_tiles(shape_unquantized, mx_dtype):
  assert len(shape_unquantized) == 2, f"shape_unquantized must have exactly 2 dimensions, got {len(shape_unquantized)}"
  
  P, F = shape_unquantized
  
  # Allocate data tile
  # Quantize-MX shrinks the free-dim by 4x because it packs 4 elements into 1.
  mx_data_sbuf = nl.ndarray((P, F//4), dtype=mx_dtype, buffer=nl.sbuf)
  
  # Allocate scale tile
  # Nominally the scale tile is sized (P//8, F//4) given that the scaling
  # group shape is [8P, 4F]. But when P > 32, the scales must be placed in the
  # partition-dim quadrant from which the corresponding scaling group originated 
  # hence we must allocate the full P.
  if P <= 32: # Can store all scales in first p-dim quadrant.
    mx_scale_sbuf = nl.ndarray((P//8, F//4), dtype=nl.uint8, buffer=nl.sbuf)
  else: # Must oversize and spread across quadrants.
    mx_scale_sbuf = nl.ndarray((P, F//4), dtype=nl.uint8, buffer=nl.sbuf)
  
  return mx_data_sbuf, mx_scale_sbuf

# Read unquantized tensors from HBM and establish interleaved layout in SBUF.
# use_tensor_copy=true: Straight read from HBM->SBUF, then use SBUF-to-SBUF TensorCopy to stride the data.
#   Intended to demonstrate how to stride the tile using VectorE/ScalarE if tile already present on SBUF.
# use_tensor_copy=false: Stride the data while reading HBM->SBUF.
#   Intended to demonstrate how to stride the tile if coming from HBM, using only the DMA engine.
# The output shapes are [P//4, F*4] where the [P,F] is the shape of the corresponding unquantized input tensor.
def copy_data_strided(stationary_hbm, moving_hbm, use_tensor_copy: bool = True):  
    
  # The HBM tensors have nominal shape [P,F]. Reshape into [4, P//4, F]. 
  # In other words, we divide the contraction axis into 4 "P" tiles since we'll eventually
  # need to read data from each tile and pack them together on SBUF.
  
  # These dimensions reflect the shape of each "P" tile.
  P_st = stationary_hbm.shape[0] // 4
  F_st = stationary_hbm.shape[1]
  P_mv = moving_hbm.shape[0] // 4
  F_mv = moving_hbm.shape[1]
  
  stationary_hbm_reshape = stationary_hbm.reshape((4, P_st, F_st))
  moving_hbm_reshape = moving_hbm.reshape((4, P_mv, F_mv))

  # Allocate SBUF tensors to store the strided result.
  # The shape is [P//4, F, 4] where the [P,F] is the shape of the unquantized input tensor.
  # In other words, we view the free-dim as having F_st/F_mv groups of 4 elements.
  # Taking 3D views of both the HBM and SBUF tensors allows for cleaner indexing.
  stationary_sbuf_strided = nl.ndarray((P_st, F_st, 4), dtype=stationary_hbm.dtype, buffer=nl.sbuf)
  moving_sbuf_strided = nl.ndarray((P_mv, F_mv, 4), dtype=moving_hbm.dtype, buffer=nl.sbuf)    

  # Perform a TensorCopy to achieve the required layout.
  if (use_tensor_copy):

    # First load from HBM -> SBUF. Take "P" tiles from HBM and write them
    # contiguously (adjacent to each other) into the SBUF free-dim. 
    # This load is not the focus of this example so its details are encapsulated in load_tensor_helper().
    # The SBUF shapes will be stationary_sbuf [P_st, 4, F_st], moving_sbuf [P_mv, 4, F_mv]
    stationary_sbuf, moving_sbuf = load_tensor_helper(stationary_hbm_reshape, moving_hbm_reshape)

    # Perform SBUF-to-SBUF TensorCopy to shuffle the data into the required MX layout.
    # Here are some tips on how to read this access pattern (AP).
    # .ap(pattern) = tuple of [step_size, count], right-most is the inner (fastest changing) dimension of the access pattern (AP).
    # The dst (*_strided) has no AP specified, meaning it is linearly written to.
    # To understand the src AP it's useful to refer to the SBUF Layout diagram in load_tensor_helper().
    # We read 1 element, then step F elements to the next tile, 4 times total. In other words, we gather a group
    # of 4 elements (one from each tile).
    # Then step 1 element and repeat the above F times to read an entire row of SBUF.
    # Then step to the next row of SBUF and repeat the above for all P rows of SBUF.
    # Note, this example is shown as a strided-read but it could be re-written as a strided-write, though it will be slower.
    # Secondly, the source tile can be in PSUM (i.e. the result of a prior matmul).
  
    nisa.tensor_copy(src=stationary_sbuf.ap(pattern=[[4*F_st, P_st], [1, F_st], [F_st, 4]], offset=0), dst=stationary_sbuf_strided)
    nisa.tensor_copy(src=moving_sbuf.ap(pattern=[[4*F_mv, P_mv], [1, F_mv], [F_mv, 4]], offset=0), dst=moving_sbuf_strided)

  # Perform a strided DMA to achieve the required layout.
  else:

    # Similar to TensorCopy, the we linearly write to stationary_sbuf_strided.
    # When reading from *_hbm_reshape, we read one element from each tile.
    # Then step 1 element and repeat the above F times, thereby reading one full row of HBM.
    # Then step to the next row of HBM and repeat the above P times.

    nisa.dma_copy(src=stationary_hbm_reshape.ap(pattern=[[F_st, P_st], [1, F_st], [P_st*F_st, 4]], offset=0),
                  dst=stationary_sbuf_strided)
    nisa.dma_copy(src=moving_hbm_reshape.ap(pattern=[[F_mv, P_mv], [1, F_mv], [P_mv*F_mv, 4]], offset=0),
                  dst=moving_sbuf_strided)

  # Return as 2D.
  return stationary_sbuf_strided.reshape((P_st, F_st*4)), moving_sbuf_strided.reshape((P_mv, F_mv*4))