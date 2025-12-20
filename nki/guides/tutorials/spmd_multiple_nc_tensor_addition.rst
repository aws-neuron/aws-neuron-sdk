.. _nki_spmd_multiple_nc_tensor_addition:

SPMD Tensor Addition using Multiple Neuron Cores
=========================================================================

In this tutorial we reuse the :ref:`simple tensor addition kernel <nki-tutorial-spmd-tensor-addition>`,
but directly control how our kernels and tensors are distributed across multiple neuron cores.

Doing so, we expand our knowledge about:

-  The NKI syntax and the :ref:`Logical Neuron Cores (LNC) <nki-about-lnc>`.
-  :doc:`nki.language.spmd_dim() <../api/generated/nki.language.spmd_dim>` and :doc:`nki.language.nc()  <../api/generated/nki.language.nc>`

PyTorch
-------

Reusing existing compute kernel in helper function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We start by reusing the ``nki_tensor_add_kernel_`` compute kernel that has large tensor inputs,
but operates on a subset of the tensor at a tile size of ``[128, 512]``. 
The partition dimension tile size is chosen according to the tile size
restrictions (:doc:`nki.language.tile_size.pmax <../api/generated/nki.language.tile_size>`),
while the free dimension tile size is chosen arbitrarily (``512``).

.. nki_example:: ../../examples/tensor_addition/spmd_multiple_nc_tensor_addition_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_48

In this example:

1. We reuse the NKI kernel in ``nki_tensor_add_kernel_`` which is decorated with the
   `nki.jit` decorator to call the nki compiler to compile the kernel.
2. Recall this kernel defines offsets into the tensors based on the ID of
   the worker executing the code (``nl.program_id``), and generates tile
   indices using these offsets with ``nl.arange``. 
3. Using SPMD execution as discussed in :ref:`Logical Neuron Cores (LNC) <nki-about-lnc>`,
   note that each worker only operates on a (sub-tensor) tile of the
   input/output tensors. By accessing its own ``program_id``, each
   worker can calculate the offsets it needs to access the correct
   tiles.
4. When multiple Neuron Cores are specified in the SPMD launch grid, these tensors are further
   sharded across available cores. On Trainium 2, we have 2 local cores that have shared HBM.
5. As before, the first axis of the tensor (mapped to the partition-dimension) is
   tiled into blocks of 128, based on hardware restrictions (see :ref:`Tile
   Size Considerations <nki-about-tiling>`).
   The second axis (mapped to the free-dimension) is tiled into blocks of 512 (no tile-size constraint, 
   since the addition operation is performed on the Vector engine, the only restriction is on-chip memory capacity).
6. ``nl.store`` for kernels running on both cores will write to an ``c_output`` in
   shared HBM, dramatically increasing the throughput of the computation.

SPMD execution
^^^^^^^^^^^^^^

1. We want to shard the workload across 2 cores, so for every ``nl.nc(2)`` we determine our initial ``axis=0`` to be
   ``128`` from the expected slice size in the kernel ``*`` the number of cores ``= 256``.
2. Thus we alter our previous sample and change ``grid_x`` to ``a_input.shape[0] // (128 * 2)`` to account for this.
3. Launch the kernel with launch grid ``[nl.spmd_dim(grid_x, nl.nc(2)), grid_y]``

As before, we are using a two-dimensional grid where the first dimension of the
tensor is tiled in the X dimension of the grid while the second
dimension is tiled in the Y dimension of the grid. We similarly
assume that tensor sizes are a multiple of maximum tile sizes allowed,
so we do not need to handle partial tiles. 

However, this time we also directly specify how each instance of our kernel will be distributed
across multiple local Neuron Cores such that:

.. code-block::

   # Physical NC [0]: kernel[n, m] where n is 0 or even
   # Physical NC [1]: kernel[n, m] where n is odd

.. _nki-tutorial-spmd-multiple-nc-tensor-add-launching-pytorch:

Launching kernel and testing correctness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To execute the kernel, we prepare tensors ``a`` and ``b``, and call the
``nki_tensor_add_nc2`` helper function. We also verify the correctness of the NKI kernel against, torch by
comparing the outputs of both, using ``torch.allclose``:

.. nki_example:: ../../examples/tensor_addition/spmd_multiple_nc_tensor_addition_torch.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_49


Output:

::

   2023-12-29 15:18:00.000558:  14283  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
   2023-12-29 15:18:00.000559:  14283  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/neuroncc_compile_workdir/49f554a2-2c55-4a88-8054-cc9f20824a46/model.MODULE_5007921933048625946+d41d8cd9.hlo.pb', '--output', '/tmp/neuroncc_compile_workdir/49f554a2-2c55-4a88-8054-cc9f20824a46/model.MODULE_5007921933048625946+d41d8cd9.neff', '--verbose=35']
   .
   Compiler status PASS
   output_nki=tensor([[1.459  1.488  1.607  ... 1.217  0.7354 1.457 ]
         [1.793  0.7373 0.8877 ... 1.813  0.8936 1.39  ]
         [0.7285 0.9473 1.531  ... 1.04   1.302  0.8413]
         ...
         [0.7705 1.195  1.047  ... 1.307  0.588  0.7725]
         [1.21   1.719  1.209  ... 1.171  0.583  0.5034]
         [1.307  1.521  0.9526 ... 0.5825 1.518  0.673 ]],
          device='xla:1', dtype=torch.bfloat16)
   2023-12-29 15:18:02.000219:  14463  INFO ||NEURON_CACHE||: Compile cache path: /var/tmp/neuron-compile-cache
   2023-12-29 15:18:02.000220:  14463  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: ['neuronx-cc', '--target=trn1', 'compile', '--framework', 'XLA', '/tmp/neuroncc_compile_workdir/2e135b73-1c3b-45e4-a6f0-2c4b105c20e5/model.MODULE_10032327759287407517+d41d8cd9.hlo.pb', '--output', '/tmp/neuroncc_compile_workdir/2e135b73-1c3b-45e4-a6f0-2c4b105c20e5/model.MODULE_10032327759287407517+d41d8cd9.neff', '--verbose=35']
   .
   Compiler status PASS
   output_torch=tensor([[1.459  1.488  1.607  ... 1.217  0.7354 1.457 ]
         [1.793  0.7373 0.8877 ... 1.813  0.8936 1.39  ]
         [0.7285 0.9473 1.531  ... 1.04   1.302  0.8413]
         ...
         [0.7705 1.195  1.047  ... 1.307  0.588  0.7725]
         [1.21   1.719  1.209  ... 1.171  0.583  0.5034]
         [1.307  1.521  0.9526 ... 0.5825 1.518  0.673 ]],
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

Helper function and SPMD execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can reuse the same NKI compute kernel defined for PyTorch above and declare a helper function
to launch the compute-kernel with appropriate grid/block sizes, to perform the computation:

.. nki_example:: ../../examples/tensor_addition/spmd_multiple_nc_tensor_addition_nki_kernels.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_48

As before, we are using a two-dimensional grid where the first dimension of the
tensor is tiled in the X dimension of the grid, while the second
dimension is tiled in the Y dimension of the grid. We similarly
assume that tensor sizes are a multiple of maximum tile sizes allowed,
so we do not need to handle partial tiles. 

However, this time we also directly specify how each instance of our kernel will be distributed
across multiple local Neuron Cores such that:

.. code-block::

   # Physical NC [0]: kernel[n, m] where n is 0 or even
   # Physical NC [1]: kernel[n, m] where n is odd

.. _nki-tutorial-spmd-multiple-nc-tensor-add-launching-jax:

Launching kernel and testing correctness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To execute the kernel, we prepare arrays ``a`` and ``b``, and call the
``nki_tensor_add_nc2`` helper function. We also verify the correctness of the NKI kernel against, JAX by
comparing the outputs of both, using ``jax.numpy.allclose``:

.. nki_example:: ../../examples/tensor_addition/spmd_multiple_nc_tensor_addition_jax.py
   :language: python
   :linenos:
   :marker: NKI_EXAMPLE_50

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

Download all source code
--------------------------

Click the links to download source code of the kernels and the testing code
discussed in this tutorial.

* NKI baremetal implementation: :download:`spmd_multiple_nc_tensor_addition_nki_kernels.py <../../examples/tensor_addition/spmd_multiple_nc_tensor_addition_nki_kernels.py>`
    * You must also download :download:`spmd_tensor_addition_nki_kernels.py <../../examples/tensor_addition/spmd_tensor_addition_nki_kernels.py>`
      into the same folder to run this script.
* PyTorch implementation: :download:`spmd_multiple_nc_tensor_addition_torch.py <../../examples/tensor_addition/spmd_multiple_nc_tensor_addition_torch.py>`
    * You must also download :download:`spmd_multiple_nc_tensor_addition_nki_kernels.py <../../examples/tensor_addition/spmd_multiple_nc_tensor_addition_nki_kernels.py>` and
      :download:`spmd_tensor_addition_nki_kernels.py <../../examples/tensor_addition/spmd_tensor_addition_nki_kernels.py>`
      into the same folder to run this PyTorch script.
* JAX implementation: :download:`spmd_multiple_nc_tensor_addition_jax.py <../../examples/tensor_addition/spmd_multiple_nc_tensor_addition_jax.py>`
    * You must also download :download:`spmd_multiple_nc_tensor_addition_nki_kernels.py <../../examples/tensor_addition/spmd_multiple_nc_tensor_addition_nki_kernels.py>` and
      :download:`spmd_tensor_addition_nki_kernels.py <../../examples/tensor_addition/spmd_tensor_addition_nki_kernels.py>`
      into the same folder to run this PyTorch script.

You can also view the source code in the GitHub repository `nki_samples <https://github.com/aws-neuron/nki-samples/tree/main/src/nki_samples/tutorials/tensor_addition/>`_

Example usage of the scripts:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run NKI baremetal implementation:

.. code-block::

   python3 spmd_multiple_nc_tensor_addition_nki_kernels.py

Run PyTorch implementation:

.. code-block::

   python3 spmd_multiple_nc_tensor_addition_torch.py

Run JAX implementation:

.. code-block::

   python3 spmd_multiple_nc_tensor_addition_jax.py
