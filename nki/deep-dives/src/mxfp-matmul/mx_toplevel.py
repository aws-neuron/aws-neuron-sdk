import torch
import os
import nki.language as nl
import numpy as np
import torch_xla
import shutil
import ml_dtypes as mld
from mx_cpu_utils import generate_stabilized_mx_data, nc_matmul_mx_golden, quantize_mx_golden
from mx_kernels import kernel_offline_quantized_mx_matmul, kernel_on_device_quantize_matmul_mx, kernel_copy_strided_quantize_matmul_mx

# Global compiler flags
NEURON_CC_BASE_FLAGS = " --target trn3 --pipeline compile SaveTemps --internal-compiler-debug-mode=all --internal-backend-options='--print-format=json,condensed' "

device = None
cpu = None

# NKI kernels use these _x4 custom dtypes to represent MXFP* data.
quantized_dtype_to_x4_map = {
  mld.float8_e5m2: nl.float8_e5m2_x4,
  mld.float8_e4m3fn: nl.float8_e4m3fn_x4,
  mld.float4_e2m1fn: nl.float4_e2m1fn_x4,
}

def setup_compiler_workdir(test_name):
  """Setup unique compiler output directory for each test"""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  workdir = f"{current_dir}/artifacts_{test_name}"
  
  # Remove existing directory if it exists
  if os.path.exists(workdir):
    shutil.rmtree(workdir)
  os.makedirs(workdir, exist_ok=True)
  
  # Set full environment variable
  os.environ["NEURON_CC_FLAGS"] = f"{NEURON_CC_BASE_FLAGS} --compile_workdir {workdir}"

def compare_and_print_results(res, golden, rtol=5e-2, atol=5e-2):
  print("\n\nResult shape:", res.shape)
  
  # Ensure both are numpy float32
  res_float = res.astype(np.float32) if res.dtype != np.float32 else res
  golden_float = golden.astype(np.float32) if golden.dtype != np.float32 else golden
  
  match = np.allclose(res_float, golden_float, rtol=rtol, atol=atol)
  print("\nnp.allclose pass?", match)
  
  if not match:
    # Print mismatch info
    diff = np.abs(res_float - golden_float)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
  
  # Print first and last row, first 3 and last 3 columns
  print(f"\nDevice Output:\n[{res_float[0,:3]} ... {res_float[0,-3:]}]\n...\n[{res_float[-1,:3]} ... {res_float[-1,-3:]}]")
  print(f"\nGolden:\n[{golden_float[0,:3]} ... {golden_float[0,-3:]}]\n...\n[{golden_float[-1,:3]} ... {golden_float[-1,-3:]}]")

def print_test_header(test_name):
  border_length = max(60, len(test_name) + 8)  # Ensure minimum width + padding
  print(f"\n\n{'='*border_length}")
  print(f"    {test_name}")
  print(f"{'='*border_length}\n")

# This test will quantize to MXFP8 on the host.
# Then execute Matmul-MX on the device using these offline-quantized tiles.
def run_offline_quantized_matmul_mx_test(quantized_dtype):
  
  # Choose max tile-sizes for TensorE.
  M, K, N = 128, 128, 512

  print_test_header(f"OFFLINE_QUANTIZED_MX_MATMUL - stationary <{quantized_dtype.__name__}> @ moving <{quantized_dtype.__name__}>")

  setup_compiler_workdir(f"offline_quantized_mx_matmul")

  # Generate stationary MX tile. Note the scales will be packed contiguously here. The kernel will later load the scales into SBUF
  # in the required scattered fashion.
  st_unquantized_shape = (K, M*4)
  _, _, st_mx_data_x4, st_mx_scale = generate_stabilized_mx_data(quantized_dtype, st_unquantized_shape)

  # Generate moving MX tile
  mv_unquantized_shape = (K, N*4)
  _, _, mv_mx_data_x4, mv_mx_scale = generate_stabilized_mx_data(quantized_dtype, mv_unquantized_shape)

  # Call the Kernel. Perform matmul-mx: stationary_mx @ moving_mx
  output_kernel = kernel_offline_quantized_mx_matmul(
    torch.from_numpy(st_mx_data_x4).to(device), 
    torch.from_numpy(st_mx_scale).to(device), 
    torch.from_numpy(mv_mx_data_x4).to(device), 
    torch.from_numpy(mv_mx_scale).to(device), 
    quantized_dtype_to_x4_map[quantized_dtype]
  )

  output_kernel_np = output_kernel.cpu().float().numpy()

  # Generate the golden
  golden = nc_matmul_mx_golden(st_mx_data_x4, mv_mx_data_x4, st_mx_scale, mv_mx_scale, quantized_dtype, quantized_dtype)

  compare_and_print_results(output_kernel_np, golden)

# This test will quantize the stationary tile to MXFP8 on the host, and moving tile on device.
# Then execute Matmul-MX on the device,
def run_on_device_quantize_matmul_mx_test(quantized_dtype_stationary, quantized_dtype_moving):
  
  # Choose max tile-sizes for TensorE.
  M, K, N = 128, 128, 512
 
  print_test_header(f"ON_DEVICE_QUANTIZE_MATMUL_MX - stationary <{quantized_dtype_stationary.__name__}> @ moving <{quantized_dtype_moving.__name__}>")

  setup_compiler_workdir(f"on_device_quantize_matmul_m")

  # Generate stationary MX tile. Note the scales will be packed contiguously here. The kernel will later load the scales into SBUF
  # in the required scattered fashion.
  st_unquantized_shape = (K, M*4)
  _, _, st_mx_data_x4, st_mx_scale = generate_stabilized_mx_data(quantized_dtype_stationary, st_unquantized_shape)

  # Generate moving tile
  mv_unquantized_shape = (K, N*4)
  # Notice we don't just generate random fp data using, say, np.random.
  # Instead we use generate_stabilized_mx_data()'s fp_data output to get stabilized unquantized data that can be
  # quantized and dequantized without loss of precision.
  mv_data, _, _, _ = generate_stabilized_mx_data(quantized_dtype_moving, mv_unquantized_shape)

  # Call the Kernel. Quantize mv_data, then perform Matmul-MX.
  output_kernel = kernel_on_device_quantize_matmul_mx(
    torch.from_numpy(st_mx_data_x4).to(device), 
    torch.from_numpy(st_mx_scale).to(device), 
    torch.from_numpy(mv_data).bfloat16().to(device), # Convert to bf16,
    quantized_dtype_to_x4_map[quantized_dtype_stationary], # stationary mx
    quantized_dtype_to_x4_map[quantized_dtype_moving], # moving qmx output
  )

  output_kernel_np = output_kernel.cpu().float().numpy()

  # Generate the golden
  # Quantize moving tensor as an intermediate step.
  moving_mx_data, moving_mx_scale = quantize_mx_golden(mv_data, quantized_dtype_moving)
  # Matmul-MX
  golden = nc_matmul_mx_golden(st_mx_data_x4, moving_mx_data, st_mx_scale, moving_mx_scale, quantized_dtype_stationary, quantized_dtype_moving)

  compare_and_print_results(output_kernel_np, golden)

# This example starts with two HBM tensors, establishes the required SBUF layout using
# either TensorCopy on the NeuronCore or via DMA, quantizes both tensors, then does Matmul-MX
def run_copy_strided_test(quantized_dtype, use_tensor_copy: bool = True):
  # Choose max tile-sizes for TensorE. But here we're specifying unquantized shapes.
  # Since Matmul-MX allows for 4x larger contraction dimension, we choose K=512.
  K, M, N = 512, 128, 512

  print_test_header(f"COPY_STRIDED_{'TENSOR_COPY' if use_tensor_copy else 'DMA'} - <{quantized_dtype.__name__}> @ <{quantized_dtype.__name__}>")

  setup_compiler_workdir(f"copy_strided_test_tensor_copy_{use_tensor_copy}")

  # Generate the stationary and moving tensors in bf16.
  # Using generate_stabilized_mx_data() to generate FP data that is within the MX data-type range.
  # Contraction dimension is the first dimensions, as is required by TensorE.
  st_shape = (K, M)
  st_data, _, _, _ = generate_stabilized_mx_data(quantized_dtype, st_shape)
  
  mv_shape = (K, N)
  mv_data, _, _, _ = generate_stabilized_mx_data(quantized_dtype, mv_shape)

  # Call the kernel
  output_kernel = kernel_copy_strided_quantize_matmul_mx(
    torch.from_numpy(st_data).bfloat16().to(device),
    torch.from_numpy(mv_data).bfloat16().to(device),
    quantized_dtype_to_x4_map[quantized_dtype],
    use_tensor_copy
  )

  output_kernel_np = output_kernel.cpu().float().numpy()

  # To generate a golden we simply perform matmul using the input fp tensors.
  # Notice we're not using the matmul_mx_golden/quantize_mx_golden utilities -- they mimic the hardware
  # and therefore assume the input tensors have the interleaved layout.
  golden = st_data.T @ mv_data
  
  compare_and_print_results(output_kernel_np, golden)

if __name__ == "__main__":

  device = torch_xla.device()
  cpu = torch.device('cpu')
  
  # Matmul-MX with MX tensors prepared on host
  run_offline_quantized_matmul_mx_test(mld.float8_e5m2) # FP8 @ FP8
  run_offline_quantized_matmul_mx_test(mld.float4_e2m1fn) # FP4 @ FP4

  # Matmul-MX with moving tensor quantized on device.
  run_on_device_quantize_matmul_mx_test(mld.float4_e2m1fn, mld.float8_e5m2) # Mixed FP4 @ FP8
  run_on_device_quantize_matmul_mx_test(mld.float8_e5m2, mld.float8_e5m2) # FP8 @ FP8

  # Use TensorCopy to stride the data
  run_copy_strided_test(mld.float8_e5m2, True) # FP8 @ FP8

  # Use DMA to stride the data
  run_copy_strided_test(mld.float8_e5m2, False) # FP8 @ FP8