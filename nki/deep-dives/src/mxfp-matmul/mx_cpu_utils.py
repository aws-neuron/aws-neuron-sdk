################################################################
# CPU Utilities to generate MX kernel input and golden data
################################################################

import numpy as np
import ml_dtypes as mld

# Ensure dtype is in the list of MX FP8/FP4 dtypes we support
def validate_quantized_dtype(dtype):
  if dtype not in {mld.float8_e5m2, mld.float8_e4m3fn, mld.float4_e2m1fn}:
    raise ValueError(f"Unsupported quantized dtype: {dtype}")
  return dtype == mld.float4_e2m1fn

# Get exponent for float32 in IEEE 754 standard
def get_float32_exp(float_data):
  man_nbits, exp_nbits = 23, 8
  return (float_data.astype(np.float32).view(np.uint32) >> man_nbits) & ((1 << exp_nbits) - 1)

# max normal
# float8_e5m2: S 11110 11 = ± 2^15 × 1.75 = ± 57,344
# float8_e4m3fn: S 1111 110 = ± 2^8 × 1.75 = ± 448
# float4_e2m1fn: S 11 1 = ± 2^2 × 1.5 = ± 6
def get_mx_fp_max(mx_dtype):
  """Get maximum representable value for MX dtype"""
  validate_quantized_dtype(mx_dtype)
  if mx_dtype == mld.float8_e5m2:
    return 57344.0  # 2^15 * 1.75
  elif mx_dtype == mld.float8_e4m3fn:
    return 448.0    # 2^8 * 1.75
  elif mx_dtype == mld.float4_e2m1fn:
    return 6.0      # 2^2 * 1.5
  else:
    raise ValueError(f"Unsupported mx_dtype: {mx_dtype}")

def get_mx_max_exp(mx_dtype):
  """Get maximum exponent for MX dtype"""
  validate_quantized_dtype(mx_dtype)
  if mx_dtype == mld.float8_e5m2:
    return 15
  elif mx_dtype == mld.float8_e4m3fn:
    return 8
  elif mx_dtype == mld.float4_e2m1fn:
    return 2
  else:
    raise ValueError(f"Unsupported mx_dtype: {mx_dtype}")

def get_p_contiguous_scale(hw_scale, data_p_size, p_offset=0):
  if data_p_size <= 32:
    return hw_scale[p_offset : p_offset + data_p_size]

  scale = np.zeros((data_p_size // 8,) + tuple(hw_scale.shape[1:]), hw_scale.dtype)
  for i in range(data_p_size // 8):
    scale[i] = hw_scale[i // 4 * 32 + i % 4 + p_offset]

  return scale

# inputs/outputs are numpy, with shape [P,F]
# returns:
#   mx_data_golden x4 mimicked packing. If fp8, then uint32 containing 4 x fp8 elements. If fp4, then uint8 containing 2 x fp4 elements.
#   mx_scale_golden as uint8 with shape [P//8, F//4] (scales are packed contiguously)
def quantize_mx_golden(in_tensor, out_quantized_dtype, ocp_saturation = True, reverse_dst_fdim_group = 0, custom_mx_max_exp=None):
  max_exp = custom_mx_max_exp(out_quantized_dtype) if custom_mx_max_exp else get_mx_max_exp(out_quantized_dtype)
  max_val = get_mx_fp_max(out_quantized_dtype)
  float32_exp_bias = 127

  P, F = in_tensor.shape
  SP, SF = P // 8, F // 4

  in_tensor_ = np.copy(in_tensor)

  RG = reverse_dst_fdim_group
  # reverse free dimension by a group of RG elements (keep the order within each group)
  if RG > 0:
    assert F % RG == 0
    in_tensor_ = in_tensor_.reshape(P, F // RG, RG)[:, ::-1, :].reshape(P, F)

  exp = get_float32_exp(in_tensor_)

  # Reshape exponent tensor to group by 8x4 blocks for max computation
  exp_reshaped = exp.reshape(SP, 8, SF, 4)

  # Compute max exponent for each 8x4 block using vectorized operations
  # Take max over the 8x4 dimensions (axes 1 and 3)
  mx_scale_golden = np.max(exp_reshaped, axis=(1, 3)).astype(np.uint8) - max_exp

  # Convert scale exponents to scale factors
  scale_exp = mx_scale_golden.astype(np.int32) - float32_exp_bias
  scale_factors = 2.0**scale_exp  # Shape: [SP, SF]

  # Expand scale factors to match input tensor shape using vectorized operations
  # Each scale factor applies to an 8x4 block
  scale_expanded_p = np.repeat(scale_factors, 8, axis=0)  # Shape: [P, SF]
  scale = np.repeat(scale_expanded_p, 4, axis=1)  # Shape: [P, F]

  # Quantize: divide by scale
  mx_data_golden = in_tensor_ / scale
  if ocp_saturation:
    mx_data_golden = np.clip(mx_data_golden, -max_val, max_val)
  
  # Cast to out_quantized_dtype then mimic x4 packing
  mx_data_golden = mx_data_golden.astype(out_quantized_dtype)
  mx_data_golden_x4 = pack_mx_data_into_x4(mx_data_golden)

  return mx_data_golden_x4, mx_scale_golden

# *_x4 inputs must mimic x4 packing via uint
#   if quantized_dtype=fp8, then must be uint32 containing 4 x quantized_dtype elements
#   if quantized_dtype=fp4, then must be uint8 containing 2 x quantized_dtype elements
# *_scale inputs are numpy uint8.
# use_contiguous_scale: True=scales are packed together contiguously, False=scales are spread across p-dim quadrants.
# Return numpy result.
def nc_matmul_mx_golden(stationary_x4, moving_x4, stationary_scale, moving_scale, stationary_quantized_dtype, moving_quantized_dtype,
                        use_contiguous_scale=True, stationary_scale_p_offset=0, moving_scale_p_offset=0):
  
  validate_quantized_dtype(stationary_quantized_dtype)
  validate_quantized_dtype(moving_quantized_dtype)

  # Unpack and upcast to fp32
  moving = unpack_mx_data_from_x4(moving_x4, moving_quantized_dtype).astype(np.float32)
  moving_scale = moving_scale.astype(np.float32)
  stationary = unpack_mx_data_from_x4(stationary_x4, stationary_quantized_dtype).astype(np.float32)
  stationary_scale = stationary_scale.astype(np.float32)

  # Process moving tensor
  new_shape = moving.shape[:-1] + (moving.shape[-1] // 4, 4)
  moving = moving.reshape(new_shape)
  MP, MF0, MF1 = moving.shape
  assert MF1 == 4
  # moving_scale = moving_scale.cpu().numpy().astype(np.float32)
  if not use_contiguous_scale:
    # if scale follows hw layout, make it contiguous at partition dimension
    moving_scale = get_p_contiguous_scale(moving_scale, MP, moving_scale_p_offset)

  MSP, MSF0 = moving_scale.shape

  # The scale tensor may have more columns than needed (e.g., when stationary and moving scales are packed together).
  moving_scale_relevant = moving_scale[:, :MF0]

  # Convert scale exponents to scale factors
  moving_scale_factors = 2.0 ** (moving_scale_relevant - 127)  # Shape: [MSP, MF0]

  # Expand scale factors to match moving tensor shape
  # Each scale factor applies to an 8x1x4 block
  moving_scale_expanded = np.repeat(moving_scale_factors[:, :, np.newaxis], 4, axis=2)  # Shape: [MSP, MF0, 4]
  moving_scale_expanded = np.repeat(moving_scale_expanded[:, np.newaxis, :, :], 8, axis=1)  # Shape: [MSP, 8, MF0, 4]
  moving_scale_expanded = moving_scale_expanded.reshape(MSP * 8, MF0, 4)  # Shape: [MP, MF0, 4]

  # Apply scaling
  moving *= moving_scale_expanded

  # Process stationary tensor
  new_shape = stationary.shape[:-1] + (stationary.shape[-1] // 4, 4)
  stationary = stationary.reshape(new_shape)
  SP, SF0, SF1 = stationary.shape
  assert SF1 == 4
  stationary = stationary.astype(np.float32)

  if not use_contiguous_scale:
    # if scale follows hw layout, make it contiguous at partition dimension
    stationary_scale = get_p_contiguous_scale(stationary_scale, SP, stationary_scale_p_offset)

  SSP, SSF0 = stationary_scale.shape

  # The scale tensor may have more columns than needed (e.g., when stationary and moving scales are packed together).
  stationary_scale_relevant = stationary_scale[:, :SF0]

  # Convert scale exponents to scale factors
  stationary_scale_factors = 2.0 ** (stationary_scale_relevant - 127)  # Shape: [SSP, SF0]

  # Expand scale factors to match stationary tensor shape
  # Each scale factor applies to an 8x1x4 block
  stationary_scale_expanded = np.repeat(stationary_scale_factors[:, :, np.newaxis], 4, axis=2)  # Shape: [SSP, SF0, 4]
  stationary_scale_expanded = np.repeat(stationary_scale_expanded[:, np.newaxis, :, :], 8, axis=1)  # Shape: [SSP, 8, SF0, 4]
  stationary_scale_expanded = stationary_scale_expanded.reshape(SSP * 8, SF0, 4)  # Shape: [SP, SF0, 4]

  # Apply scaling
  stationary *= stationary_scale_expanded

  # This einsum mimics the hardware's Matmul-MX operation. In contrast to a standard 2D x 2D matmul, 
  # this performs an additional multiply-accumulate on the 4 elements inside one _x4 element, which is what
  # the hardware does.
  golden = np.einsum("kiq,kjq->ij", stationary, moving)
  return golden

def dequantize_mx_golden(mx_data_x4, quantized_dtype, mx_scale):
  """
  Dequantize MX data back to float32, reversing quantize_mx_golden.

  This is the exact reverse of quantize_mx_golden:
  - quantize: out_data = in_data / scale, then clip, then cast to MX format
  - dequantize: cast to float32, then out_data = in_data * scale
  where scale = 2^(mx_scale - float32_exp_bias)

  Args:
      mx_data_x4: np.ndarray mimicking x4 packing via uint. [P, F//4] if fp8, [P, F//2] if fp4
      mx_scale: np.ndarray [SP, SF] in uint8 - scale tensor where SP=P//8, SF=F//4 if fp8 or F//2 if fp4 

  Returns:
      np.ndarray [P, F] in float32 - dequantized data (same shape as original input to quantize)
  """
  
  is_fp4 = validate_quantized_dtype(quantized_dtype)

  float32_exp_bias = 127

  P, F_packed = mx_data_x4.shape
  SP, SF = mx_scale.shape

  assert SP == P // 8, f"Scale tensor P dimension mismatch: expected {P//8}, got {SP}"
  expected_SF = F_packed // 2 if is_fp4 else F_packed
  assert SF == expected_SF, f"Scale tensor F dimension mismatch: expected {expected_SF}, got {SF}"

  # Unpack
  mx_data_unpacked = unpack_mx_data_from_x4(mx_data_x4, quantized_dtype)
  # Convert quantized_dtype to float32
  data_float = mx_data_unpacked.astype(np.float32)
  P_expanded, F_expanded = data_float.shape

  # The F dimension is expanded, so check it's as expected
  expected_F_expanded = F_packed * 2 if is_fp4 else F_packed * 4
  assert F_expanded == expected_F_expanded, f"Unexpected expansion: expected {expected_F_expanded}, got {F_expanded}"

  # Convert scale exponents to scale factors
  scale_exp = mx_scale.astype(np.int32) - float32_exp_bias
  scale_exp = np.clip(scale_exp, -127, 127)
  scale_factors = 2.0**scale_exp

  # Use numpy's repeat and tile to expand scale factors to match data shape
  # Each scale factor needs to be applied to an 8x4 block
  # First expand along P dimension: repeat each row 8 times
  scale_expanded_p = np.repeat(scale_factors, 8, axis=0)  # Shape: [P_expanded, SF]

  # Then expand along F dimension: repeat each column 4 times
  scale_expanded = np.repeat(scale_expanded_p, 4, axis=1)   # Shape: [P_expanded, F_expanded]

  # Dequantize: multiply by scale (reverse of quantize division)
  dequantized_data = data_float * scale_expanded

  return dequantized_data

def generate_stabilized_mx_data(quantized_dtype, shape, val_range=1.0):
  """
  Generate stabilized floating-point data and its equivalent MX quantized representation.

  This function returns standard floating-point numbers along with their equivalent
  MX quantized data and scale tensors that are stabilized in the sense that the
  floating-point data and MX data can convert to each other exactly without losing precision.

  Args:
      quantized_dtype: MX quantization dtype (ml_dtypes.float8_e5m2, ml_dtypes.float8_e4m3fn, ml_dtypes.float4_e2m1fn)
      shape: 2D shape for the unquantized output tensor, each 8x4 block is a scaling group; e.g.,
             fp_data[8*row : 8*(row+1), 4*col : 4*(col+1)] is a scaling group
      val_range: fp_data output will be in (-val_range, val_range), (default: 1.0)

  Returns numpy tensors:
      tuple: (fp_data, quantized_mx_data, quantized_mx_data_x4, quantized_mx_scale)
          - fp_data: floating-point data
          - quantized_mx_data: MX quantized data that can be de-quantized to fp_data.
          - quantized_mx_data_x4: quantized_mx_data packed to mimic NKI MXFP_x4 datatypes.
              if quantized_dtype=fp8, then dtype=uint32 packed with 4 x quantized_dtype elements
              if quantized_dtype=fp4, then dtype=uint8 packed with 2 x quantized_dtype elements.
                uint16 is not used because it behaves inconsistently in torch when moving data host <-> device.
          - quantized_mx_scale: MX scale tensor, uint8
  """
  validate_quantized_dtype(quantized_dtype)

  _q_height, _q_width = 8, 4
  assert (shape[0] % _q_height == 0), f'shape[0] must be a multiple of {_q_height}, but got {shape[0]}'
  assert (shape[1] % _q_width == 0), f'shape[1] must be a multiple of {_q_width}, but got {shape[1]}'

  if val_range == 0:
    zeros = np.zeros(shape)
    return zeros, *quantize_mx_golden(zeros, quantized_dtype)

  # Get MX dtype parameters
  max_val = get_mx_fp_max(quantized_dtype)
  max_exp = get_mx_max_exp(quantized_dtype)

  # Generate initial random mxfp data within the mxfp dtype's range.
  rand_data = (np.random.random(shape) * 2 - 1) * max_val

  # For each scaling block, randomly select one element to have max exponent.
  # This prevents change in mx_scale after quantize(dequantize(rand_mx_data, rand_mx_scale)), causing precision loss.
  for i in range(0, shape[0], _q_height):
    for j in range(0, shape[1], _q_width):
      # Random position within the tile
      tile_i = np.random.randint(0, _q_height - 1)
      tile_j = np.random.randint(0, _q_width - 1)

      # Set this element to have maximum exponent
      # Value = ±1.xxx × 2^max_exp (where 1.xxx is the mantissa)
      sign = np.random.choice([-1, 1])
      # Within the range of [1.0, 1.5) (could be upto 1.75 for mxfp8).
      mantissa = 1.0 + np.random.random() * 0.5
      rand_data[i + tile_i, j + tile_j] = sign * mantissa * (2 ** max_exp)

  # Cast to quantized_dtype
  rand_data_quantized = rand_data.astype(quantized_dtype)
  # pack into uint to mimic x4
  rand_data_quantized_x4 = pack_mx_data_into_x4(rand_data_quantized)

  # Calculate mx_scale bounds based on val_range
  # max_val already takes max_exp into account
  float32_exp_bias = 127
  mx_scale_upper_bound = min(255, int(np.log2(val_range / max_val) + float32_exp_bias))
  mx_scale_lower_bound = max(0, mx_scale_upper_bound - 10)

  # Generate random scale
  scale_shape = (shape[0] // _q_height, shape[1] // _q_width)
  rand_quantized_scale_np = np.random.randint(mx_scale_lower_bound, mx_scale_upper_bound + 1,
                                            size=scale_shape, dtype=np.uint8)

  # Dequantize to get final fp data
  dequantized_fp_data_np = dequantize_mx_golden(rand_data_quantized_x4, quantized_dtype, rand_quantized_scale_np)

  return dequantized_fp_data_np, rand_data_quantized, rand_data_quantized_x4, rand_quantized_scale_np

def pack_mx_data_into_x4(mx_data):
  """
  Pack MX data based on dtype:
  - FP4: Pack 2 adjacent values into uint8 (4 bits each)
  - FP8: Pack 4 adjacent values into uint32 (8 bits each)
  """
  import ml_dtypes as mld
  
  if mx_data.dtype == mld.float4_e2m1fn:
    # FP4 path: pack 2 values into uint8. Each FP4 element consumes 8 bits. Take the relevant 4-bits from two elements
    # and pack into uint8.
    mx_as_bytes = mx_data.view(np.uint8)
    H, W = mx_data.shape
    assert W % 2 == 0, "Width must be divisible by 2 for FP4 packing"
    
    bytes_grouped = mx_as_bytes.reshape(H, W // 2, 2)
    return ((bytes_grouped[:, :, 0] & 0xF).astype(np.uint8) << 0) | \
            ((bytes_grouped[:, :, 1] & 0xF).astype(np.uint8) << 4)
  
  elif mx_data.dtype in [mld.float8_e5m2, mld.float8_e4m3fn]:
    # FP8 path: view automatically gives (H, W//4) shape
    # Just view it as uint32.
    return mx_data.view(np.uint32)
  
  else:
    raise ValueError(f"Unsupported dtype: {mx_data.dtype}")

def unpack_mx_data_from_x4(packed_data, target_dtype):
  """
  Unpack MX data based on target dtype:
  - FP4: Unpack uint8 into 2 adjacent values (4 bits each)
  - FP8: Unpack uint32 into 4 adjacent values (8 bits each)
  """
  import ml_dtypes as mld
  
  if target_dtype == mld.float4_e2m1fn:
    # FP4 path: unpack uint8 into 2 values
    assert packed_data.dtype == np.uint8, f"Expected uint8 for FP4, got {packed_data.dtype}"
    H, W_packed = packed_data.shape
    
    # Extract 4-bit values from uint8
    unpacked = np.zeros((H, W_packed, 2), dtype=np.uint8)
    unpacked[:, :, 0] = packed_data & 0xF
    unpacked[:, :, 1] = (packed_data >> 4) & 0xF
    
    # Each FP4 (target_dtype) actually consumes 8-bits.
    return unpacked.reshape(H, W_packed * 2).view(target_dtype)
  
  elif target_dtype in [mld.float8_e5m2, mld.float8_e4m3fn]:
    # FP8 path: view automatically gives (P, F*4) shape
    assert packed_data.dtype == np.uint32, f"Expected uint32 for FP8, got {packed_data.dtype}"
    return packed_data.view(target_dtype)
      
  else:
    raise ValueError(f"Unsupported dtype: {target_dtype}")

