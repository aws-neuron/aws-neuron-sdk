# NKI_EXAMPLE_0_BEGIN NKI_EXAMPLE_1_BEGIN
from neuronxcc import nki
import neuronxcc.nki.language as nl
# NKI_EXAMPLE_1_END


# NKI_EXAMPLE_2_BEGIN
@nki.jit
def nki_tensor_add_kernel(a_input, b_input):
    # NKI_EXAMPLE_2_END

    """NKI kernel to compute element-wise addition of two input tensors
    """

    # NKI_EXAMPLE_3_BEGIN
    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape

    # Check size of the first dimension does not exceed on-chip memory tile size limit,
    # so that we don't need to tile the input to keep this example simple
    assert a_input.shape[0] <= nl.tile_size.pmax
    # NKI_EXAMPLE_3_END

    # Load the inputs from device memory to on-chip memory
    # NKI_EXAMPLE_4_BEGIN
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)
    # NKI_EXAMPLE_4_END

    # Specify the computation (in our case: a + b)
    # NKI_EXAMPLE_5_BEGIN
    c_tile = nl.add(a_tile, b_tile)
    # NKI_EXAMPLE_5_END

    # NKI_EXAMPLE_6_BEGIN
    # Create a HBM tensor as the kernel output
    c_output = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)

    # Return kernel output as function output
    return c_output
# NKI_EXAMPLE_0_END NKI_EXAMPLE_6_END


if __name__ == "__main__":
    # NKI_EXAMPLE_8_BEGIN
    import numpy as np

    a = np.ones((4, 3), dtype=np.float16)
    b = np.ones((4, 3), dtype=np.float16)

    # NKI_EXAMPLE_12_BEGIN
    # Run NKI kernel on a NeuronDevice
    c = nki_tensor_add_kernel(a, b)
    # NKI_EXAMPLE_12_END

    print(c)
    # NKI_EXAMPLE_8_END
