.. meta::
    :description: Learn how to implement and run your first NKI kernel on AWS Neuron accelerators
    :date-modified: 11/18/2025

.. _quickstart-run-nki-kernel:

Quickstart: Implement and run your first kernel
================================================

The Neuron Kernel Interface (NKI) lets you write low-level kernels that use the ISA of Trainium1/Inferentia2/Trainium2/Trainium3 ML accelerators. Your kernels can be used in PyTorch and JAX models to speed up critical parts of your model. This topic guides you through your first time writing a NKI kernel. It will help you understand the process when using AWS Neuron and NKI. 

When you have completed it, you will have a simple kernel that adds two input tensors and returns the result and a test program in PyTorch or JAX.

* This quickstart is for: Customers new to NKI
* Time to complete: ~10 minutes

Prerequisites
--------------

Before you begin, you will need an Inf2, Trn1, Trn2, or Trn3 EC2 instance.

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

    @nki.jit(platform_target="trn1")
    def nki_tensor_add_kernel(a_input, b_input):
        """
        NKI kernel to compute element-wise addition of two input tensors.
        """

Add the ``nki_tensor_add_kernel`` function definition above. Make sure you annotate it with the ``@nki.jit`` decorator as in the example above.

Step 3: Check input size and shapes
------------------------------------

In this step, you add a couple of assertions to check that ``a_input`` and ``b_input`` are the same size and that these will fit within the on-chip tile size.

Add the following assertions to your ``nki_tensor_add_kernel`` function in ``add_kernel.py``.

.. code-block:: python

    # Check both input tensor shapes are the same for element-wise operation.
    assert a_input.shape == b_input.shape

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
    # to SBUF with DMA copy. Note: 'sbuf' is a keyword in NKI.
    a_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)
    nisa.dma_copy(dst=a_tile, src=a_input)

    b_tile = sbuf.view(dtype=b_input.dtype, shape=b_input.shape)
    nisa.dma_copy(dst=b_tile, src=b_input)

The ``sbuf.view`` function allows you to allocate tensors in SBUF. The ``sbuf`` keyword is available in any NKI kernel. Here you allocate ``a_tile`` and ``b_tile`` and use the ``nisa.dma_copy`` :doc:`instruction </nki/api/generated/nki.isa.dma_copy>` to copy tensors between HBM and SBUF memories. You first supply the destination for the copy, ``a_tile`` and ``b_tile``. Then you provide the source for the copy, ``a_input`` and ``b_input``, as seen in this example.

Step 5: Add the two tensors
----------------------------

In this step, you add code to allocate a destination tensor in SBUF and put the results of adding these two tensor in the new tensor.

.. code-block:: python

    # Allocate space for the result and use tensor_tensor to perform
    # element-wise addition. Note: the first argument of 'tensor_tensor'
    # is the destination tensor.
    c_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)
    nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

As in step 4, you allocate a space for the ``c_tile`` in SBUF, using ``sbuf.view``. Since the shape of the output will be the same shape as the inputs, you can use the ``a_input`` data type and shape for the allocation. You use the ``nisa.tensor_tensor`` :doc:`instruction </nki/api/generated/nki.isa.tensor_tensor>` to perform element-wise calculation on two tensors. The first argument of ``tensor_tensor`` is the destination tensor, ``c_tile``, and the sources, ``a_tile`` and ``b_tile``, follow it. You must also provide an op which tells ``tensor_tensor`` which operation to perform on the inputs. In this case, you use ``op=nl.add`` to specify addition.

Step 6: Copy the result to HBM
-------------------------------

In this step, you will allocate space for the output tensor in HBM and copy the result from SBUF to the new tensor. This is the inverse of what you did with the input, where you copied the inputs from HBM into SBUF.

.. code-block:: python

    # Create a tensor in HBM and copy the result into HBM. Note: Simlar to
    # 'sbuf', 'hbm' is a keyword in NKI.
    c_output = hbm.view(dtype=a_input.dtype, shape=a_input.shape)
    nisa.dma_copy(dst=c_output, src=c_tile)

You use the hbm keyword to create tensors in HBM, similar to how you allocated space in SBUF with the sbuf keyword. You then copy the result in ``c_tile`` into ``c_output``. Remember that ``c_output`` is the destination and ``c_tile`` is the source for the ``dma_copy`` instruction. The copy is needed because outputs, like inputs, need to be in HBM.

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

    @nki.jit(platform_target="trn1")
    def nki_tensor_add_kernel(a_input, b_input):
        """
        NKI kernel to compute element-wise addition of two input tensors.
        """

        # Check both input tensor shapes are the same for element-wise operation.
        assert a_input.shape == b_input.shape

        # Check the first dimension's size to ensure it does not exceed on-chip
        # memory tile size, since this simple kernel does not tile inputs.
        assert a_input.shape[0] <= nl.tile_size.pmax

        # Allocate space for the input tensors in SBUF and copy the inputs from HBM
        # to SBUF with DMA copy. Note: 'sbuf' is a keyword in NKI.
        a_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)
        nisa.dma_copy(dst=a_tile, src=a_input)

        b_tile = sbuf.view(dtype=b_input.dtype, shape=b_input.shape)
        nisa.dma_copy(dst=b_tile, src=b_input)

        # Allocate space for the result and use tensor_tensor to perform
        # element-wise addition. Note: the first argument of 'tensor_tensor'
        # is the destination tensor.
        c_tile = sbuf.view(dtype=a_input.dtype, shape=a_input.shape)
        nisa.tensor_tensor(dst=c_tile, data1=a_tile, data2=b_tile, op=nl.add)

        # Create a tensor in HBM and copy the result into HBM. Note: Simlar to
        # 'sbuf', 'hbm' is a keyword in NKI.
        c_output = hbm.view(dtype=a_input.dtype, shape=a_input.shape)
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
          from torch_xla.core import xla_model as xm
          from add_kernel import nki_tensor_add_kernel

          # Setup the XLA device and generate input tensors.
          device = xm.xla_device()

          a = torch.ones((4, 3), dtype=torch.float16).to(device=device)
          b = torch.ones((4, 3), dtype=torch.float16).to(device=device)

          # Invoke the kernel to add the results.
          c = nki_tensor_add_kernel(a, b)

          # Print creates an implicit XLA barrier/mark-step (triggers XLA compilation)
          print(c)

      You use the ``xla_device`` function to look up device information. You use the device to move tensors created in PyTorch onto the Neuron device. You call the ``nki_tensor_add_kernel(a, b)`` function to invoke the kernel. The ``print`` function tells PyTorch to trace the model, causing the kernel to be compiled and run on the Neuron device.

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

.. code-block:: bash

    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python test_program.py

Note that the ``NEURON_PLATFORM_OVERRIDE`` environment variable sets the target architecture. This can also be set in by using ``@nki.jit`` with the ``platform_target`` argument. In this example it is set to ``trn2``  which creates a binary suitable for running on Trn2 machines. For Trn1 / Inf2, specify ``trn1``; and for Trn3 specify ``trn3``.

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

          driver.py:6: DeprecationWarning: Use torch_xla.device instead
            device = xm.xla_device()
          2025-11-07 07:29:41.834546: W neuron/pjrt-api/neuronpjrt.cc:1972] Use PJRT C-API 0.73 as client did not specify a PJRT C-API version
          2025-Nov-07 07:29:46.0669 78638:78679 [1] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):213 CCOM WARN NET/OFI Failed to initialize sendrecv protocol
          2025-Nov-07 07:29:46.0679 78638:78679 [1] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):354 CCOM WARN NET/OFI aws-ofi-nccl initialization failed
          2025-Nov-07 07:29:46.0689 78638:78679 [1] ncclResult_t nccl_net_ofi_init_no_atexit_fini_v6(ncclDebugLogger_t):183 CCOM WARN NET/OFI Initializing plugin failed
          2025-Nov-07 07:29:46.0699 78638:78679 [1] net_plugin.cc:97 CCOM WARN OFI plugin initNet() failed is EFA enabled?
          The KLR format is located at: final_klir_filepath='/tmp/nki_tensor_add_kernelapbsy67c.klir'
          =========== warnings from kernel tracing add_kernel.nki_tensor_add_kernel ===========
          /home/ubuntu/pytorch-klir/lib/python3.10/site-packages/nki/isa/neuron_isa.py:527:86:keyword-only arguments are not supported in NKI
          /home/ubuntu/pytorch-klir/lib/python3.10/site-packages/nki/isa/neuron_isa.py:527:86:keyword-only arguments are not supported in NKI
          /home/ubuntu/pytorch-klir/lib/python3.10/site-packages/nki/language/math_ops.py:59:29:keyword-only arguments are not supported in NKI
          /home/ubuntu/pytorch-klir/lib/python3.10/site-packages/nki/language/math_ops.py:59:29:keyword-only arguments are not supported in NKI
          /home/ubuntu/pytorch-klir/lib/python3.10/site-packages/nki/isa/neuron_isa.py:1748:65:keyword-only arguments are not supported in NKI
          /home/ubuntu/pytorch-klir/lib/python3.10/site-packages/nki/isa/neuron_isa.py:1748:65:keyword-only arguments are not supported in NKI
          2025-11-07 07:29:46.000731:  78638  INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: neuronx-cc compile --framework=XLA /tmp/ubuntu/neuroncc_compile_workdir/31e1821a-b569-4bdd-b46d-7baef6bb345f/model.MODULE_6747419296008355553+e30acd3a.hlo_module.pb --output /tmp/ubuntu/neuroncc_compile_workdir/31e1821a-b569-4bdd-b46d-7baef6bb345f/model.MODULE_6747419296008355553+e30acd3a.neff --target=trn1 --verbose=35
          .Completed run_backend_driver.

          Compiler status PASS
          tensor([[2., 2., 2.],
                  [2., 2., 2.],
                  [2., 2., 2.],
                  [2., 2., 2.]], device='xla:0', dtype=torch.float16)
          nrtucode: internal error: 54 object(s) leaked, improper teardown

   .. tab:: JAX

      .. code-block:: text

          Compiler status PASS
          The KLR format is located at: final_klir_filepath='/tmp/nki_tensor_add_kernelq3uk7mz0.klir'
          =========== warnings from kernel tracing add_kernel.nki_tensor_add_kernel ===========
          /home/ubuntu/jax-klir/lib/python3.10/site-packages/nki/isa/neuron_isa.py:527:86:keyword-only arguments are not supported in NKI
          /home/ubuntu/jax-klir/lib/python3.10/site-packages/nki/isa/neuron_isa.py:527:86:keyword-only arguments are not supported in NKI
          /home/ubuntu/jax-klir/lib/python3.10/site-packages/nki/language/math_ops.py:59:29:keyword-only arguments are not supported in NKI
          /home/ubuntu/jax-klir/lib/python3.10/site-packages/nki/language/math_ops.py:59:29:keyword-only arguments are not supported in NKI
          /home/ubuntu/jax-klir/lib/python3.10/site-packages/nki/isa/neuron_isa.py:1748:65:keyword-only arguments are not supported in NKI
          /home/ubuntu/jax-klir/lib/python3.10/site-packages/nki/isa/neuron_isa.py:1748:65:keyword-only arguments are not supported in NKI
          .Completed run_backend_driver.

          Compiler status PASS
          [[2. 2. 2.]
           [2. 2. 2.]
           [2. 2. 2.]
           [2. 2. 2.]]

Congratulations! You have now your first NKI kernel written and running. If you encountered any issues, see the Common issues section below.

Common issues
--------------

Uh oh! Did you encounter an error or other issue while working through this quickstart? Here are some commonly encountered issues and how to address them.

* ``nki``, ``jax``, ``torch``, etc. library not found: You may need to activate the PyTorch or JAX environment.
* No neuron device available: You may not have the ``neuron`` kernel module loaded. Make sure the ``neuron`` module is loaded with ``sudo modprobe neuron``.

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
* :doc:`NKI Developer Guides </nki/guides/how-to-guides/index>`
