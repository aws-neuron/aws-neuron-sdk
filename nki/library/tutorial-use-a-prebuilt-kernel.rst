.. meta::
    :description: Learn how to accelerate compute workloads on AWS Trainium by using pre-built kernels from the NKI Library  with PyTorch.
    :date-modified: 12/02/2025

.. _nkl_tutorial_prebuilt_kernel:

Tutorial: Accelerate MLP with a Pre-built NKI Library Kernel
=============================================================

This tutorial demonstrates how to leverage pre-built kernels from the NKI Library  to accelerate machine learning workloads on AWS Trainium. You'll learn how to integrate an optimized multi-layer perceptron (MLP) kernel into your PyTorch model, compare its outputs with a reference implementation, and achieve significant performance improvements while maintaining numerical accuracy. By the end of this tutorial, you'll be able to replace standard PyTorch layers with high-performance NKI Library kernels in your own models.

To accelerate a compute workload on Trainium with a kernel from NKI Library, you will need the following:

* A reference implementation in PyTorch
* A matching kernel in NKI Library with input parameters in the supported range

Creating a Reference Implementation
-------------------------------------

Here is an example of a reference implementation of the MLP layer in a typical transformer:

.. code-block:: python

    class MLPReference(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int, dtype=torch.bfloat16):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            gate_output = torch.nn.functional.silu(self.gate_proj(hidden))
            up_output = self.up_proj(hidden)
            return self.down_proj(gate_output * up_output)

This will serve as a baseline for numerical correctness and optionally for performance. To get the CPU torch reference output, execute the forward pass. The reference output will be used later to confirm that the kernel has been integrated properly.

.. code-block:: python

    model = MLPReference(hidden_size, intermediate_size, dtype=torch.bfloat16)
    model.eval()
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16) * 2
    with torch.no_grad():
        reference_output = model(input_tensor)

Using the NKI Library MLP Kernel
---------------------------------

After it has been confirmed that the reference implementation is working and reasonable (non-zero numbers, non-NaN, etc.), we can try using the MLP kernel from NKI Library:

.. code-block:: python

    from nkilib.core.mlp.mlp import fused_mlp_isa_kernel

Use the API documentation to fill out the arguments and to ensure the input parameters are within the supported space for the kernel. Keep in mind that in the newest release of NKI, the SPMD launch grid (for LNC sharding) can be passed simply as an integer, and the output directly stored as an assignment like the following example. Move the output to CPU so that this can be compared.

.. code-block:: python

    with torch.no_grad():
        nki_output = fused_mlp_isa_kernel[LNC_DEGREE](
            hidden=nki_input,
            gate_w=gate_w_xla,
            up_w=up_w_xla,
            down_w=down_w_xla,
            attn_output=None,
            norm_type=NormType.NO_NORM,
            dtype=nl.bfloat16,
            act_fn=ActFnType.SiLU,
        )
    nki_output_cpu = nki_output.cpu()

Comparing Outputs
------------------

Finally, confirm that the kernel output matches the CPU output.

.. code-block:: python

    print(f"\nReference output:\n{reference_output}")
    print(f"\nNKI output:\n{nki_output_cpu}")
    print(f"\nDifference:\n{torch.abs(reference_output - nki_output_cpu)}")

    # Compare outputs.
    assert nki_output_cpu.shape == reference_output.shape, f"Shape mismatch: {nki_output_cpu.shape} vs {reference_output.shape}"
    torch.testing.assert_close(nki_output_cpu, reference_output, rtol=1e-2, atol=1e-2)

You should see something like this:

.. code-block:: text

    Compiler status PASS

    Reference output:
    tensor([[[ 0.6914, -0.3477, -0.1060,  ...,  0.0679,  0.4023,  0.2949],
             [-0.1826,  0.1572, -1.0781,  ..., -0.2422,  0.2832, -0.3457],
             [ 0.0986,  0.4785,  0.6016,  ..., -0.6133, -0.2471,  0.1484],
             ...,
             [ 0.0074, -0.4141, -0.4629,  ..., -0.2314, -0.1118, -0.0645],
             [ 0.3262,  0.6016,  0.4453,  ..., -0.1738,  0.5781,  0.2617],
             [-0.4336, -0.0167,  0.4629,  ..., -0.2715,  0.3613, -0.0204]]],
           dtype=torch.bfloat16)

    NKI output:
    tensor([[[ 0.6914, -0.3438, -0.1060,  ...,  0.0708,  0.4004,  0.2930],
             [-0.1816,  0.1592, -1.0781,  ..., -0.2412,  0.2812, -0.3496],
             [ 0.0981,  0.4785,  0.6016,  ..., -0.6133, -0.2490,  0.1484],
             ...,
             [ 0.0100, -0.4141, -0.4648,  ..., -0.2334, -0.1118, -0.0635],
             [ 0.3262,  0.6055,  0.4473,  ..., -0.1689,  0.5781,  0.2617],
             [-0.4375, -0.0175,  0.4629,  ..., -0.2715,  0.3613, -0.0215]]],
           dtype=torch.bfloat16)

    Difference:
    tensor([[[0.0000, 0.0039, 0.0000,  ..., 0.0029, 0.0020, 0.0020],
             [0.0010, 0.0020, 0.0000,  ..., 0.0010, 0.0020, 0.0039],
             [0.0005, 0.0000, 0.0000,  ..., 0.0000, 0.0020, 0.0000],
             ...,
             [0.0026, 0.0000, 0.0020,  ..., 0.0020, 0.0000, 0.0010],
             [0.0000, 0.0039, 0.0020,  ..., 0.0049, 0.0000, 0.0000],
             [0.0039, 0.0007, 0.0000,  ..., 0.0000, 0.0000, 0.0011]]],
           dtype=torch.bfloat16)
    PASSED

Now this can be used in place of the MLP layer in your torch model definition.

Complete Example
-----------------

The full script is available below. Make sure to set the environment variable ``NEURON_PLATFORM_TARGET_OVERRIDE`` before running the script (for example: ``NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python mlp_comparison.py``).

.. code-block:: python

    # MLP CTE new frontend kernel torch demo script.
    # Set `NEURON_PLATFORM_TARGET_OVERRIDE` before executing this script (e.g., trn2)
    # This is required for the new NKI frontend.
    import torch
    import torch.nn as nn
    import torch_xla.core.xla_model as xm
    import nki.language as nl
    import pytest
    from nkilib.core.mlp.mlp import fused_mlp_isa_kernel
    from nkilib.core.utils.common_types import ActFnType, NormType

    # Set LNC degree. For trn2, set to 2, otherwise 1.
    LNC_DEGREE = 2

    # Step 1: Create a reference torch implementation of your workload
    # Be careful of reference data types. They should reflect what you intend on requesting from the kernel later.
    class MLPReference(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int, dtype=torch.bfloat16):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            gate_output = torch.nn.functional.silu(self.gate_proj(hidden))
            up_output = self.up_proj(hidden)
            return self.down_proj(gate_output * up_output)


    # Sweep on different input sizes.
    @pytest.mark.parametrize(
        "batch_size,seq_len,hidden_size,intermediate_size",
        [
            (1, 512, 512, 768),
            (1, 256, 256, 512), 
            (1, 4, 8192, 132), # generation
        ],
    )
    def test_mlp_kernel_accuracy(batch_size, seq_len, hidden_size, intermediate_size):
        """Test NKI MLP kernel against reference implementation."""
        device = xm.xla_device()

        # Step 2a. Create reference model and sample input
        torch.manual_seed(42)
        model = MLPReference(hidden_size, intermediate_size, dtype=torch.bfloat16)
        model.eval()
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16) * 2

        # Step 2b. Gather reference output
        # Use no_grad for forward passes.
        with torch.no_grad():
            reference_output = model(input_tensor)

        # Step 3. If using nn.linear, transpose from [out, in] (torch weight storage layout) to [in, out] (kernel layout)
        nki_input = input_tensor.to(device=device, dtype=torch.bfloat16)
        gate_w_xla = model.gate_proj.weight.T.contiguous().to(device=device, dtype=torch.bfloat16)
        up_w_xla = model.up_proj.weight.T.contiguous().to(device=device, dtype=torch.bfloat16)
        down_w_xla = model.down_proj.weight.T.contiguous().to(device=device, dtype=torch.bfloat16)

        # Step 4. Find and execute NKI kernel
        with torch.no_grad():

            nki_output = fused_mlp_isa_kernel[LNC_DEGREE](
                hidden=nki_input,
                gate_w=gate_w_xla,
                up_w=up_w_xla,
                down_w=down_w_xla,
                attn_output=None,
                norm_type=NormType.NO_NORM,
                dtype=nl.bfloat16,
                act_fn=ActFnType.SiLU,
            )

        nki_output_cpu = nki_output.cpu()
        if len(nki_output_cpu.shape) < 3:
            nki_output_cpu = nki_output_cpu.unsqueeze(0) # Workaround: TKG squeezing left side

        print(f"\nReference output:\n{reference_output}")
        print(f"\nNKI output:\n{nki_output_cpu}")
        print(f"\nDifference:\n{torch.abs(reference_output - nki_output_cpu)}")

        # Step 5. Compare outputs.
        assert nki_output_cpu.shape == reference_output.shape, f"Shape mismatch: {nki_output_cpu.shape} vs {reference_output.shape}"
        torch.testing.assert_close(nki_output_cpu, reference_output, rtol=1e-2, atol=1e-2)


    if __name__ == "__main__":
        pytest.main([__file__, "-v", "-s", "-x"])
