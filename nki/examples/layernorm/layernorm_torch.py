"""
Copyright (C) 2024, Amazon.com. All Rights Reserved

LayerNorm NKI kernel implementation.

"""
# NKI_EXAMPLE_47_BEGIN
import torch
from torch_xla.core import xla_model as xm
import argparse
import os

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

# Reference torch implementation
def layernorm_layer(input_tensor, epsilon, gamma_vector, beta_vector):
    # Compute the mean and variance of the input tensor along the last dimension
    mean = input_tensor.mean(dim=-1, keepdim=True)
    variance = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
    # Subtract the mean from the input and divide by the square root of the variance plus epsilon
    normalized_input = (input_tensor - mean) / torch.sqrt(variance + epsilon)
    # Apply the affine transformation
    normalized_input = normalized_input * gamma_vector + beta_vector
    return normalized_input

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
    parser.add_argument("--version",
            default="v1",
            choices=["v1", "v2"],
            help="Test versions")
    args = parser.parse_args()
    return args


from neuronxcc.nki.docs.examples.layernorm.layernorm_nki_kernel import nki_layernorm_kernel_v1, \
  nki_layernorm_kernel_v2

if __name__ == "__main__":
    args = parse_args()
    func_dict = {"v1": nki_layernorm_kernel_v1,
                 "v2": nki_layernorm_kernel_v2,
                 }

    device = xm.xla_device()
    num_rows = args.nrows
    num_cols = args.ncols

    # Generate toy example
    input_tensor = torch.rand((num_rows, num_cols), dtype=torch.float32)
    gamma_vector = torch.rand((num_cols), dtype=torch.float32)
    beta_vector = torch.rand((num_cols), dtype=torch.float32)
    epsilon = 1e-5

    # Compute torch layernorm layer in cpu
    output_torch = layernorm_layer(input_tensor, epsilon, gamma_vector, beta_vector)

    # Copy tensors to NeuronDevice
    input_tensor = input_tensor.to(device=device)
    gamma_vector = gamma_vector.to(device=device)
    beta_vector = beta_vector.to(device=device)

    print(f">>>> Running version {args.version}.")
    func = func_dict[args.version]

    # add nki_jit decorator

    # Compute NKI layernorm kernel in NeuronDevice
    xm.mark_step()
    output_nki = func(input_tensor, epsilon, gamma_vector, beta_vector)
    xm.mark_step()
    output_nki = output_nki.to(device='cpu')

    # Accuracy check : Compare the output tensors
    allclose = torch.allclose(output_torch, output_nki, atol=1e-3, rtol=1e-2)
    if allclose:
        print("NKI and Torch match")
    else:
        print("NKI and Torch differ")
        # NKI_EXAMPLE_47_END

    assert allclose