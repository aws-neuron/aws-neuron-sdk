"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

LayerNorm NKI kernel implementation.

"""
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np
import math
import os
import argparse


def nki_layernorm_kernel_v1(input_tensor, epsilon, gamma_vector, beta_vector, output_tensor):
  """Computes LayerNorm.
    Used nki.language APIs only.
  """
  # Ensure that the shapes of tensors match
  assert input_tensor.shape == output_tensor.shape
  assert input_tensor.shape[1] == gamma_vector.shape[0] == beta_vector.shape[0]

  # Generate tile indices for loading/storing data
  i_p_io = nl.arange(nl.tile_size.pmax)[:, None]
  i_f_io = nl.arange(input_tensor.shape[1])[None, :]
  i_p_param = nl.arange(1)[:, None]

  # Number of rows in the input tensor
  num_rows = input_tensor.shape[0]

  # Load gamma and beta, which will be reused across rows/tiles of input_tensor
  gamma_sb = nl.load(gamma_vector.reshape((1, gamma_vector.shape[0]))[i_p_param, i_f_io])
  beta_sb = nl.load(beta_vector.reshape((1, beta_vector.shape[0]))[i_p_param, i_f_io])

  # Broadcast the gamma and beta to match the dimensions of the tiles
  gamma_sb_bcast = gamma_sb.broadcast_to((nl.tile_size.pmax, gamma_vector.shape[0]))
  beta_sb_bcast = beta_sb.broadcast_to((nl.tile_size.pmax, beta_vector.shape[0]))

  # Tile partition dimension of the input tensor by nl.tile_size.pmax
  for i in nl.affine_range(math.ceil(input_tensor.shape[0]/nl.tile_size.pmax)):
    # Load input tile
    input_sb = nl.load(input_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io],
                       mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

    # Compute mean and variance
    mean = nl.mean(input_sb, axis=1)
    # Trick to calculate var with mean: mean(x^2) - mean(x)^2
    var = nl.mean(nl.square(input_sb), axis=1) - mean * mean

    # Normalize the input by shifting with the mean 
    # and scaling with rsqrt of variance and epsilon
    shift_scale_tensor = (input_sb - mean) * nl.rsqrt(var + epsilon)
    
    # Scale the normalized tile using gamma and add beta
    output_sb = shift_scale_tensor * gamma_sb_bcast + beta_sb_bcast

    nl.store(output_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io], value=output_sb,
             mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

def nki_layernorm_kernel_v2(input_tensor, epsilon, gamma_vector, beta_vector, output_tensor):
  """Computes LayerNorm.
    Used nki.isa APIs to calculate mean/variance and perform shift/scale.
  """
  # Ensure that the shapes of tensors match
  assert input_tensor.shape == output_tensor.shape
  assert input_tensor.shape[1] == gamma_vector.shape[0] == beta_vector.shape[0]

  # Generate tile indices for loading/storing data
  i_p_io = nl.arange(nl.tile_size.pmax)[:, None]
  i_f_io = nl.arange(input_tensor.shape[1])[None, :]
  i_p_param = nl.arange(1)[:, None]

  # Number of rows in the input tensor
  num_rows = input_tensor.shape[0]

  # Load gamma and beta, which will be reused across rows/tiles of input_tensor
  gamma_sb = nl.load(gamma_vector.reshape((1, gamma_vector.shape[0]))[i_p_param, i_f_io])
  beta_sb = nl.load(beta_vector.reshape((1, beta_vector.shape[0]))[i_p_param, i_f_io])

  # Broadcast the gamma and beta to match the dimensions of the tiles
  gamma_sb_bcast = gamma_sb.broadcast_to((nl.tile_size.pmax, gamma_vector.shape[0]))
  beta_sb_bcast = beta_sb.broadcast_to((nl.tile_size.pmax, beta_vector.shape[0]))

  # Tile partition dimension of the input tensor by nl.tile_size.pmax
  for i in nl.affine_range(math.ceil(input_tensor.shape[0]/nl.tile_size.pmax)):
    # Load input tile
    input_sb = nl.load(input_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io],
                       mask=(i * nl.tile_size.pmax + i_p_io < num_rows))

    # Tile free dimension of the input tensor by nl.tile_size.bn_stats_fmax, 
    # as bn_stats has a free dimension size limit
    i_f_bn = nl.arange(nl.tile_size.bn_stats_fmax)[None, :]
    i_f_stats = nl.arange(6)[None, :]
    num_bn_stats = math.ceil(input_tensor.shape[1]/nl.tile_size.bn_stats_fmax)
    stats_results = nl.ndarray((nl.tile_size.pmax, 6*num_bn_stats), dtype=np.float32)
    for j in nl.affine_range(num_bn_stats):
      stats_results[i_p_io, j * 6 + i_f_stats] = nisa.bn_stats(
              input_sb[i_p_io, j * nl.tile_size.bn_stats_fmax + i_f_bn],
              mask=(j * nl.tile_size.bn_stats_fmax + i_f_bn < input_tensor.shape[1]),
              dtype=np.float32)
      
    # Aggregate bn_stats results to compute mean and var
    i_f_aggr = nl.arange(6*num_bn_stats)[None, :]
    mean_var = nisa.bn_aggr(stats_results[i_p_io, i_f_aggr])
    mean = mean_var[i_p_io, 0]
    var = mean_var[i_p_io, 1]

    # Get reciprocal of sqrt(var + epsilon)
    scale_var = nl.rsqrt(var + epsilon)

    # Putting the shift and scale together in one line to trigger two alu_op tensor_vector instruction
    # shift_scale_tensor = (input_sb - mean_var[i_p_stats, i_f_mean]) * scale_var
    shift_scale_tensor = nisa.tensor_scalar(data=input_sb, op0=np.subtract,
                                            operand0=mean,
                                            op1=np.multiply,
                                            operand1=scale_var)
    
    # Scale the normalized tile using gamma and add beta
    output_sb = shift_scale_tensor * gamma_sb_bcast + beta_sb_bcast

    nl.store(output_tensor[i * nl.tile_size.pmax + i_p_io, i_f_io], value=output_sb,
             mask=(i * nl.tile_size.pmax + i_p_io < num_rows))


def parse_args():
    parser = argparse.ArgumentParser(
    """Run LayerNorm pytorch implementation.
    """)
    parser.add_argument("--nrows",
                        default=4*1024,
                        type=int,
                        help="""The number of input rows""")
    parser.add_argument("--ncols",
                        default=8*1024,
                        type=int,
                        help="""The number of input columns""")
    parser.add_argument("--mode",
                        choices=["accuracy", "perf"],
                        default="accuracy",
                        help="""Do accuracy test or perf test.
                                Accuracy test compares LayerNorm kernel against PyTorch implementation.
                                Perf test will generate a NEFF for the PyTorch implementation in local directory
                                for a manual run of neuron-profile.
                             """)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    func_dict = {"v1": nki_layernorm_kernel_v1,
                 "v2": nki_layernorm_kernel_v2,
                 }

    # Generate toy example
    num_rows = args.nrows
    num_cols = args.ncols
    input_tensor = np.random.rand(num_rows, num_cols).astype(np.float32)
    gamma_vector = np.random.rand(num_cols).astype(np.float32)
    beta_vector = np.random.rand(num_cols).astype(np.float32)
    epsilon = 1e-5
            
    if args.mode == "accuracy":
        # version 1
        print(f">>>> Running version 1")
        nki_out_v1 = np.empty((num_rows, num_cols), dtype=np.float32)
        nki.baremetal(nki_layernorm_kernel_v1)\
                    (input_tensor, epsilon, gamma_vector, beta_vector, nki_out_v1)
        # version 2
        print(f">>>> Running version 2")
        nki_out_v2 = np.empty((num_rows, num_cols), dtype=np.float32)
        nki.baremetal(nki_layernorm_kernel_v2)\
                    (input_tensor, epsilon, gamma_vector, beta_vector, nki_out_v2)
        # compare
        np_all = np.all(nki_out_v1 == nki_out_v1)
        print(f">>>> LayerNorm V1 and V2 matches?", np_all)
        assert np_all
                
    else:
      # perf mode
      for version in ["v1", "v2"]:
          print(f">>>> Running version {version}.")
          func = func_dict[version]
          nki_out_test = np.empty((num_rows, num_cols), dtype=np.float32)
          nki.benchmark(func,
                        save_neff_name='file.neff',
                        save_trace_name='profile.ntff')\
                        (input_tensor, epsilon, gamma_vector, beta_vector, nki_out_test)
          os.rename("file.neff", f"{version}_{num_rows}_{num_cols}.neff")
          os.rename("profile.ntff", f"{version}_{num_rows}_{num_cols}.ntff")
