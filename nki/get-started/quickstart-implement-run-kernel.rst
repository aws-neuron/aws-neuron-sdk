.. meta::
    :description: Learn how to implement and run your first NKI kernel on AWS Neuron accelerators
    :date-modified: 10/19/2026

.. _quickstart-run-nki-kernel:

Quickstart: Implement and run your first kernel
================================================

The Neuron Kernel Interface (NKI) lets you write low-level kernels that use the ISA of Trainium2 and Trainium3 ML accelerators. Your kernels can be used in PyTorch and JAX models to speed up critical parts of your model. This topic guides you through your first time writing a NKI kernel. It will help you understand the process when using AWS Neuron and NKI. 

When you have completed it, you will have a simple kernel that adds two input tensors and returns the result and a test program in PyTorch or JAX.

* This quickstart is for: Customers new to NKI
* Time to complete: ~10 minutes

Prerequisites
--------------

Before you begin, you will need a Trn2 or Trn3 EC2 instance.

* Your EC2 instance should have the Neuron SDK and NKI library installed on them. If you used the Deep Learning AMI (DLAMI), these will be available by activating a PyTorch or JAX environment with Python's venv.
* You will need a text editor or IDE for editing code.
* A basic familiarity with Python and either PyTorch or JAX will be helpful, though not strictly required.


Before you start
-----------------

Make sure you are logged in to your EC2 instance and have activated either a PyTorch or JAX environment. See :doc:`Set up your environment for NKI development <setup-env>` for details.

Step 1: Import the nki library
-------------------------------

In this step you create the ``add_kernel.py`` file and add imports for the ``nki``, ``nki.language``, and ``nki.isa`` libraries.

.. code-block:: python

    import nki
    import nki.language as nl
    import nki.isa as nisa

Open your favorite editor or IDE and create the ``add_kernel.py`` code file, and then add the imports for the NKI libraries.

Step 2: Create the nki_tensor_add_kernel
-----------------------------------------

In this step, you define the ``nki_tensor_add_kernel`` function. 

.. code-block:: python

    @nki.jit
    def nki_tensor_add_kernel(a_input, b_input):
        """
        NKI kernel to compute element-wise addition of two input tensors.
        """

Add the ``nki_tensor_add_kernel`` function definition above. Make sure you annotate it with the ``@nki.jit`` decorator as in the example above.

Step 3: Check input size and shapes
------------------------------------

In this step, you add a couple of assertions to check that ``a_input`` and ``b_input`` are the same size/datatype and that these will fit within the on-chip tile size.

Add the following assertions to your ``nki_tensor_add_kernel`` function in ``add_kernel.py``.

.. code-block:: python

        # check both input tensor shapes/dtypes are the same for element-wise operation.
        assert a_input.shape == b_input.shape
        assert a_input.dtype == b_input.dtype

        # Check the first dimension's size to ensure it does not exceed on-chip
        # memory tile size, since this simple kernel does not tile inputs.
        assert a_input.shape[0] <= nl.tile_size.pmax

The first assertion checks that ``a_input`` and ``b_input`` have the same shape. The second assertion checks that the inputs will fit in within the tile size of the on-chip memory. If an input is larger than the on-chip tile size, you must tile the input. To keep this example simple we will avoid discussing tiling further in this quick start.

Step 4: Read input into the on-chip memory
-------------------------------------------

In this step, you will add code to read the inputs from HBM into on-chip memory.

The ``nki_tensor_add_kernel`` function will receive inputs from the HBM memory and must move them into on-chip memory to operate over their values. You first create space in the on-chip memory and then copy the value into on-chip memory for each input. See :doc:`Memory Hierarchy </nki/get-started/about/memory-hierarchy-overview>` for more details on the memory hierarchy.

.. code-block:: python

    # Allocate space for the input tensors in SBUF and copy the inputs from HBM
    # to SBUF with DMA copy.
    a_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_tile, src=a_input)

    b_tile = nl.ndarray(shape=b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_tile, src=b_input)

The ``nl.ndarray`` function allows you to allocate tensors in SBUF. Here you allocate ``a_tile`` and ``b_tile`` and use the ``nisa.dma_copy`` :doc:`instruction </nki/api/generated/nki.isa.dma_copy>` to copy tensors between HBM and SBUF memories. You first supply the destination for the copy, ``a_tile`` and ``b_tile``. Then you provide the source for the copy, ``a_input`` and ``b_input``, as seen in this example.

Step 5: Add the two tensors
----------------------------

In this step, you add code to allocate a destination tensor in SBUF and put the results of adding these two tensor in the new tensor.

.. code-block:: python

    # Allocate space for the result and use tensor_tensor to perform
    # element-wise addition. Note: the first argument of 'tensor_tensor'
    # is the destination tensor.
    c_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

As in step 4, you allocate a space for the ``c_tile`` in SBUF, using ``nl.ndarray``. Since the shape of the output will be the same shape as the inputs, you can use the ``a_input`` data type and shape for the allocation. You use the ``nisa.tensor_tensor`` :doc:`instruction </nki/api/generated/nki.isa.tensor_tensor>` to perform element-wise calculation on two tensors. The first argument of ``tensor_tensor`` is the destination tensor, ``c_tile``, and the sources, ``a_tile`` and ``b_tile``, follow it. You must also provide an op which tells ``tensor_tensor`` which operation to perform on the inputs. In this case, you use ``op=nl.add`` to specify addition.

Step 6: Copy the result to HBM
-------------------------------

In this step, you will allocate space for the output tensor in HBM and copy the result from SBUF to the new tensor. This is the inverse of what you did with the input, where you copied the inputs from HBM into SBUF.

.. code-block:: python

    # Create a tensor in HBM and copy the result into HBM.
    c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=c_output, src=c_tile)

You use ``nl.ndarray`` with ``buffer=nl.shared_hbm`` to create tensors in HBM, similar to how you allocated space in SBUF with ``buffer=nl.sbuf``. You then copy the result in ``c_tile`` into ``c_output``. Remember that ``c_output`` is the destination and ``c_tile`` is the source for the ``dma_copy`` instruction. The copy is needed because outputs, like inputs, need to be in HBM.

Step 7: Return the output
--------------------------

In this step, you will return the result.

.. code-block:: python

    # Return kernel output as function output.
    return c_output

You should now have an ``add_kernel.py`` file that looks as follows.

.. code-block:: python

    import nki
    import nki.language as nl
    import nki.isa as nisa

    @nki.jit
    def nki_tensor_add_kernel(a_input, b_input):
        """
        NKI kernel to compute element-wise addition of two input tensors.
        """

        # check both input tensor shapes/dtypes are the same for element-wise operation.
        assert a_input.shape == b_input.shape
        assert a_input.dtype == b_input.dtype

        # Check the first dimension's size to ensure it does not exceed on-chip
        # memory tile size, since this simple kernel does not tile inputs.
        assert a_input.shape[0] <= nl.tile_size.pmax

        # Allocate space for the input tensors in SBUF and copy the inputs from HBM
        # to SBUF with DMA copy.
        a_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a_input)

        b_tile = nl.ndarray(shape=b_input.shape, dtype=b_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b_tile, src=b_input)

        # Allocate space for the result and use tensor_tensor to perform
        # element-wise addition. Note: the first argument of 'tensor_tensor'
        # is the destination tensor.
        c_tile = nl.ndarray(shape=a_input.shape, dtype=a_input.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

        # Create a tensor in HBM and copy the result into HBM.
        c_output = nl.ndarray(dtype=a_input.dtype, shape=a_input.shape, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=c_output, src=c_tile)

        # Return kernel output as function output.
        return c_output


Step 8: Create a PyTorch or JAX test program
---------------------------------------------

In this step, you create a test program as a Python script using either PyTorch or JAX.

.. tabs::

   .. tab:: PyTorch

      You can create a file called ``test_program.py`` with the following content.

      .. code-block:: python

          import torch
          from add_kernel import nki_tensor_add_kernel

          # Generate the input tensors on the Neuron device.
          a = torch.ones((4, 3), dtype=torch.float16, device="neuron")
          b = torch.ones((4, 3), dtype=torch.float16, device="neuron")

          # Invoke the kernel to add the results.
          c = nki_tensor_add_kernel(a, b)

          # Print the result tensor.
          print(c)

      You create input tensors on the Neuron device using PyTorch. You call the ``nki_tensor_add_kernel`` function to invoke the kernel. The ``print`` function prints the result to the console.

   .. tab:: JAX

      You can create a file called ``test_program.py`` with the following content.

      .. code-block:: python

          import jax.numpy as jnp
          from add_kernel import nki_tensor_add_kernel

          # Generate the input tensors.
          a = jnp.ones((4, 3), dtype=jnp.float16)
          b = jnp.ones((4, 3), dtype=jnp.float16)

          # Invoke the kernel to add the results.
          c = nki_tensor_add_kernel(a, b)

          # Print the result tensor.
          print(c)

      You create input tensors using the ``jax.numpy`` library. You call the ``nki_tensor_add_kernel function`` to invoke the kernel. The ``print`` function prints the result to the console.

All complete! Now, let's confirm everything works.

Confirmation
-------------

You can confirm the success of the kernel by running the driver you created in step 8.

First explicitly set the kernel LNC degree to 1:

.. code-block:: bash

    export NEURON_LOGICAL_NC_CONFIG=1

Run the driver:

.. code-block:: bash

    NEURON_PLATFORM_TARGET_OVERRIDE=trn3 python test_program.py

The ``NEURON_PLATFORM_TARGET_OVERRIDE`` environment variable sets the target architecture for compilation. In this example it is set to ``trn3`` which creates a binary suitable for running on Trn3 machines. For Trn2, specify ``trn2``.

Whether you used PyTorch or JAX for the driver, you should see the following result.

.. code-block:: text

    [[2. 2. 2.]
     [2. 2. 2.]
     [2. 2. 2.]
     [2. 2. 2.]]

You will also see some additional output depending on whether you used PyTorch or JAX.

.. tabs::

   .. tab:: PyTorch

      .. code-block:: text

            tensor([[2., 2., 2.],
                    [2., 2., 2.],
                    [2., 2., 2.],
                    [2., 2., 2.]], device='neuron:0', dtype=torch.float16)

   .. tab:: JAX

      .. code-block:: text

            WARNING:2026-04-13 01:56:40,630:jax._src.xla_bridge:901: Platform 'neuron' is experimental and not all JAX functionality may be correctly supported!
            2026-Apr-13 01:56:47.0115 838811:838863 [3] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):219 CCOM WARN NET/OFI Failed to initialize rdma protocol
            2026-Apr-13 01:56:47.0117 838811:838863 [3] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):354 CCOM WARN NET/OFI aws-ofi-nccl initialization failed
            2026-Apr-13 01:56:47.0120 838811:838863 [3] ncclResult_t nccl_net_ofi_init_no_atexit_fini_v6(ncclDebugLogger_t):183 CCOM WARN NET/OFI Initializing plugin failed
            2026-Apr-13 01:56:47.0122 838811:838863 [3] net_plugin.cc:97 CCOM WARN OFI plugin initNet() failed is EFA enabled?
            [[2. 2. 2.]
             [2. 2. 2.]
             [2. 2. 2.]
             [2. 2. 2.]]

      .. note::

         The "Platform 'neuron' is experimental" warning and CCOM warnings are harmless and can be safely ignored.

Congratulations! You have now your first NKI kernel written and running. If you encountered any issues, see the Common issues section below.

Common issues
--------------

Uh oh! Did you encounter an error or other issue while working through this quickstart? Here are some commonly encountered issues and how to address them.

* ``nki``, ``jax``, ``torch``, etc. library not found: You may need to activate the PyTorch or JAX environment.
* No neuron device available: You may not have the ``neuron`` kernel module loaded. Make sure the ``neuron`` module is loaded with ``sudo modprobe neuron``.
* ``NCC_EVRF066`` error such as ``NKI kernel 'nki_tensor_add_kernel' is configured with LNC=1, but neuronx-cc is using LNC=2``:

  - Set ``NEURON_LOGICAL_NC_CONFIG`` explicitly so the kernel and ``neuronx-cc`` use the same LNC degree, for example ``export NEURON_LOGICAL_NC_CONFIG=1``.

Clean up
---------

When you are finished with this example, you can deactivate your ``venv`` with ``deactivate`` and remove both ``add_kernel.py`` and ``test_program.py``.

Next steps
-----------

Now that you've completed this quickstart, take your work and dive into other topics that build off of it.

* :doc:`NKI Language Guide </nki/get-started/nki-language-guide>`
* :doc:`NKI Tutorials </nki/guides/tutorials/index>`

Further reading
----------------

* :doc:`NKI API Reference Manual </nki/api/index>`
* :doc:`NKI Developer Guides </nki/guides/index>`
