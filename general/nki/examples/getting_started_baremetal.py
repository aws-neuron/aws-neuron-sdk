import neuronxcc.nki.language as nl
from neuronxcc.nki import baremetal

@baremetal
def nki_tensor_add_kernel(a_input, b_input, c_output):
    """NKI kernel to compute element-wise addition of two input tensors
    """

    # Check all input/output tensor shapes are the same for element-wise operation
    assert a_input.shape == b_input.shape == c_output.shape

    # Check size of the first dimension does not exceed on-chip memory tile size limit,
    # so that we don't need to tile the input to keep this example simple
    assert a_input.shape[0] <= nl.tile_size.pmax

    # Load the inputs from device memory to on-chip memory
    a_tile = nl.load(a_input)
    b_tile = nl.load(b_input)

    # Specify the computation (in our case: a + b)
    c_tile = nl.add(a_tile, b_tile)

    # Store the result to c_output from on-chip memory to device memory
    nl.store(c_output, value=c_tile)


if __name__ == "__main__":
    import numpy as np

    a = np.ones((4, 3), dtype=np.float16)
    b = np.ones((4, 3), dtype=np.float16)
    c = np.zeros((4, 3), dtype=np.float16)

    # Run NKI kernel on a NeuronDevice
    nki_tensor_add_kernel(a, b, c)

    print(c)