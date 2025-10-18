.. _nki-tutorial-spmd-tensor-addition:

Single program, multiple data tensor addition
=============================================

In this tutorial we write a simple tensor addition kernel using NKI in PyTorch and JAX. In
doing so, we learn about:

-  The NKI syntax and the :ref:`SPMD programming model <nki-pm-spmd>`.
-  Best practices for validating and benchmarking your custom kernel
   against a reference native PyTorch or JAX implementation.

.. note::
   This tutorial is written using the SPMD programming model in NKI.
   However, as discussed in :ref:`NKI programming guide <nki-pm-spmd>`,
   adopting the SPMD programming model has **no**
   impact on performance of NKI kernel, and therefore is considered **optional** in current NKI release.


PyTorch
-------

Compute kernel
^^^^^^^^^^^^^^

We start by defining the compute kernel that has large tensor inputs,
but operates on a subset of the tensor at a tile size of ``[128, 512]``.
The partition dimension tile size is chosen according to the tile size
restrictions (:doc:`nki.language.tile_size.pmax <../api/generated/nki.language.tile_size>`),
while the free dimension tile size is chosen arbitrarily (``512``).

.. nki_example:: ../examples/tensor_addition/spmd_tensor_addition_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_27

In this example:

1. We define the NKI kernel in ``nki_tensor_add_kernel_``, decorate it with the
   `nki.jit` decorator to call the nki compiler to compile the kernel.
2. Inside, we first allocate tensor ``c_output`` as the result of the kernel
3. Next, we define offsets into the tensors, based on the ID of
   the worker executing the code (``nl.program_id``), and generate tile
   indices using these offsets with ``nl.arange``. We use advanced indexing here to showcase
   how it works. Basic indexing with slicing can also work.
   See :ref:`NKI Programming Model <nki-tensor-indexing>` for more information on different tensor indexing modes.
4. We use ``nl.program_id`` to enable SPMD execution (single-program,
   multiple-data, see :ref:`SPMD: Launching Multiple Instances of a Kernel <nki-pm-spmd>`),
   where each worker only operates on a (sub-tensor) tile of the
   input/output tensors. By accessing its own ``program_id``, each
   worker can calculate the offsets it needs to access the correct
   tiles.
5. The first axis of the tensor (mapped to the partition-dimension) is
   tiled into blocks of 128, based on hardware restrictions (see :ref:`Tile
   Size Considerations <nki-pm-tile>`).
   The second axis (mapped to the free-dimension) is tiled into blocks of 512 (no tile-size constraint, 
   since the addition operation is performed on the Vector engine, the only restriction is on-chip memory capacity).
6. We then load sub-tensors data from tensors ``a_input`` and
   ``b_input`` using ``nl.load``, to place the tiles ``a_tile`` and
   ``b_tile`` in the on-chip memory (SBUF)
7. We sum them to compute ``c_tile``, and store it back to DRAM in the
   relevant portion of the ``c_output`` tensor, using ``nl.store``.
   Since both inputs and output are the same shape, we can use the same
   set of indices to access all three tensors.
8. At the end, we use ``return`` statement to transfer the ownership of
   tensor ``c_output`` to the caller of the kernel.

SPMD execution
^^^^^^^^^^^^^^

We declare a helper function, to launch the compute-kernel with appropriate
grid/block sizes, to perform the computation over the whole input tensors.

.. nki_example:: ../examples/tensor_addition/spmd_tensor_addition_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_28

We are using a two-dimensional grid, where the first dimension of the
tensor is tiled in the X dimension of the grid, while the second
dimension is tiled in the Y dimension of the grid. In this scenario we
assume that tensor sizes are a multiple of maximum tile sizes allowed,
so we do not need to handle partial tiles.

.. _nki-tutorial-spmd-tensor-add-launching-pytorch:

Launching kernel and testing correctness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To execute the kernel, we prepare tensors ``a`` and ``b``, and call the
``nki_tensor_add`` helper function. We also verify the correctness of the NKI kernel against, torch by
comparing the outputs of both, using ``torch.allclose``:

.. nki_example:: ../examples/tensor_addition/spmd_tensor_addition_torch.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_29


Output:

::

   2023-12-29 15:18:00.000558:  14283  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
   2023-12-29 15:18:00.000559:  14283  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/neuroncc_compile_workdir/49f554a2-2c55-4a88-8054-cc9f20824a46/model.MODULE_5007921933048625946+d41d8cd9.hlo.pb', '--output', '/tmp/neuroncc_compile_workdir/49f554a2-2c55-4a88-8054-cc9f20824a46/model.MODULE_5007921933048625946+d41d8cd9.neff', '--verbose=35']
   .
   Compiler status PASS
   output_nki=tensor([[0.9297, 0.8359, 1.1719,  ..., 0.4648, 0.2188, 0.9336],
           [0.3906, 1.3125, 0.8789,  ..., 1.6562, 1.7734, 0.9531],
           [0.6445, 1.1406, 1.3281,  ..., 0.9531, 0.8711, 0.9336],
           ...,
           [0.4023, 0.6406, 1.5312,  ..., 0.7617, 0.7734, 0.3359],
           [0.8125, 0.7422, 1.2109,  ..., 0.8516, 1.2031, 0.5430],
           [1.3281, 1.2812, 1.3984,  ..., 1.2344, 0.8711, 0.5664]],
          device='xla:1', dtype=torch.bfloat16)
   2023-12-29 15:18:02.000219:  14463  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
   2023-12-29 15:18:02.000220:  14463  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/neuroncc_compile_workdir/2e135b73-1c3b-45e4-a6f0-2c4b105c20e5/model.MODULE_10032327759287407517+d41d8cd9.hlo.pb', '--output', '/tmp/neuroncc_compile_workdir/2e135b73-1c3b-45e4-a6f0-2c4b105c20e5/model.MODULE_10032327759287407517+d41d8cd9.neff', '--verbose=35']
   .
   Compiler status PASS
   output_torch=tensor([[0.9297, 0.8359, 1.1719,  ..., 0.4648, 0.2188, 0.9336],
           [0.3906, 1.3125, 0.8789,  ..., 1.6562, 1.7734, 0.9531],
           [0.6445, 1.1406, 1.3281,  ..., 0.9531, 0.8711, 0.9336],
           ...,
           [0.4023, 0.6406, 1.5312,  ..., 0.7617, 0.7734, 0.3359],
           [0.8125, 0.7422, 1.2109,  ..., 0.8516, 1.2031, 0.5430],
           [1.3281, 1.2812, 1.3984,  ..., 1.2344, 0.8711, 0.5664]],
          device='xla:1', dtype=torch.bfloat16)
   2023-12-29 15:18:03.000797:  14647  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
   2023-12-29 15:18:03.000798:  14647  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/neuroncc_compile_workdir/74f8b6ae-76d9-4dd8-af7f-e5e1c40a27a3/model.MODULE_5906037506311912405+d41d8cd9.hlo.pb', '--output', '/tmp/neuroncc_compile_workdir/74f8b6ae-76d9-4dd8-af7f-e5e1c40a27a3/model.MODULE_5906037506311912405+d41d8cd9.neff', '--verbose=35']
   .
   Compiler status PASS
   NKI and Torch match



Note that the tensor values you see will differ from what's printed
above, since this example uses torch.rand to initialize the inputs.


JAX
---

Compute kernel
^^^^^^^^^^^^^^

We can reuse the same NKI compute kernel defined for PyTorch above.

.. nki_example:: ../examples/tensor_addition/spmd_tensor_addition_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_27

SPMD execution
^^^^^^^^^^^^^^

Now we can also declare a helper function, to launch the compute-kernel with
appropriate grid/block sizes, to perform the computation:

.. nki_example:: ../examples/tensor_addition/spmd_tensor_addition_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_28

We are using a two-dimensional grid, where the first dimension of the
tensor is tiled in the X dimension of the grid, while the second
dimension is tiled in the Y dimension of the grid. In this scenario we
assume that tensor sizes are a multiple of maximum tile sizes allowed,
so we do not need to handle partial tiles.

.. _nki-tutorial-spmd-tensor-add-launching-jax:

Launching kernel and testing correctness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To execute the kernel, we prepare arrays ``a`` and ``b``, and call the
``nki_tensor_add`` helper function. We also verify the correctness of the NKI kernel against, JAX by
comparing the outputs of both, using ``jax.numpy.allclose``:

.. nki_example:: ../examples/tensor_addition/spmd_tensor_addition_jax.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_30

Output:

::

   .
   Compiler status PASS
   .
   Compiler status PASS
   .
   Compiler status PASS
   output_nki=[[0.992188 1.27344 1.65625 ... 0.90625 1.34375 1.77344]
    [0 0.90625 1.34375 ... 0.390625 0.703125 0.914062]
    [0.5 0.390625 0.703125 ... 1.22656 1.15625 1.01562]
    ...
    [1.98438 1.98438 1.98438 ... 1.33594 1.64062 1.35938]
    [0.992188 1.33594 1.64062 ... 1.16406 1.67188 1.20312]
    [1.49219 1.16406 1.67188 ... 1.375 1 1.6875]]
   .
   Compiler status PASS
   output_jax=[[0.992188 1.27344 1.65625 ... 0.90625 1.34375 1.77344]
    [0 0.90625 1.34375 ... 0.390625 0.703125 0.914062]
    [0.5 0.390625 0.703125 ... 1.22656 1.15625 1.01562]
    ...
    [1.98438 1.98438 1.98438 ... 1.33594 1.64062 1.35938]
    [0.992188 1.33594 1.64062 ... 1.16406 1.67188 1.20312]
    [1.49219 1.16406 1.67188 ... 1.375 1 1.6875]]
   .
   Compiler status PASS
   NKI and JAX match



Note that the array values you see will differ from what's printed
above, since this example uses jax.random.uniform to initialize the inputs.

Download All Source Code
--------------------------

Click the links to download source code of the kernels and the testing code
discussed in this tutorial.

* NKI baremetal implementation: :download:`spmd_tensor_addition_nki_kernels.py <../examples/tensor_addition/spmd_tensor_addition_nki_kernels.py>`
* PyTorch implementation: :download:`spmd_tensor_addition_torch.py <../examples/tensor_addition/spmd_tensor_addition_torch.py>`
    * You must also download :download:`spmd_tensor_addition_nki_kernels.py <../examples/tensor_addition/spmd_tensor_addition_nki_kernels.py>`
      into the same folder to run this PyTorch script.
* JAX implementation: :download:`spmd_tensor_addition_jax.py <../examples/tensor_addition/spmd_tensor_addition_jax.py>`
    * You must also download :download:`spmd_tensor_addition_nki_kernels.py <../examples/tensor_addition/spmd_tensor_addition_nki_kernels.py>`
      into the same folder to run this PyTorch script.

You can also view the source code in the GitHub repository `nki_samples <https://github.com/aws-neuron/nki-samples/tree/main/src/nki_samples/tutorials/tensor_addition/>`_

Example usage of the scripts:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run NKI baremetal implementation:

.. code-block::

   python3 spmd_tensor_addition_nki_kernels.py

Run PyTorch implementation:

.. code-block::

   python3 spmd_tensor_addition_torch.py

Run JAX implementation:

.. code-block::

   python3 spmd_tensor_addition_jax.py
