.. _error-code-evrf004:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EVRF004.

NCC_EVRF004
===========

**Error message**: Complex data types are not supported on the Neuron device.

You cannot use complex data types (such as ``complex64``, ``complex128``, and others) on the Neuron device directly. 

One fix is to offload complex operations to CPU, like so:

.. code-block:: python

    x = torch.tensor([1+2j, 3+4j], dtype=torch.complex64).to('cpu')

.. note::

   Since data transfer between CPU and device is expensive, this is best used when complex operations are rare.

You can also address this error by manually emulating complex tensors using real and imaginary parts:

.. code-block:: python

    real = x.real
    imag = x.imag
    ...
    # (a + bi) * (c + di)
    real_out = a_real * b_real - a_imag * b_imag
    imag_out = a_real * b_imag + a_imag * b_real
