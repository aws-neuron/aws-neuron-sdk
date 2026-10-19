.. meta::
   :description: Overview of using Logical NeuronCores with NKI
   :date_updated: 09/25/2026

.. _nki-about-lnc:

Using Logical NeuronCores (LNC) with NKI
========================================

Logical NeuronCore (LNC)
------------------------

The Neuron SDK supports grouping multiple physical NeuronCores (PNCs) into a
single logical NeuronCore (LNC). The number of PNCs grouped into an LNC is configured
both when compiling the NEFF and when loading and executing the NEFF via the
Neuron Runtime. For the corresponding compiler and runtime settings, see
:doc:`Logical NeuronCore configuration
</about-neuron/arch/neuron-features/logical-neuroncore-config>`. The Neuron SDK
currently supports an LNC degree (the number of physical NeuronCores grouped
into an LNC) of either 1 or 2.

A NEFF compiled for LNC=2 contains PNC-specific instructions for each of the
two PNCs. At runtime, the two PNCs have access to shared HBM memory (see the
:ref:`Trainium2 Architecture Guide for NKI <trainium2_arch>` for more details
on NeuronCores).

This topic explains how to use NKI with Logical NeuronCore (LNC)
configurations. The following sections describe:

1. How to configure a kernel's LNC degree
2. How to write NKI kernels with LNC-specific APIs

Configuring the LNC degree of kernels
-------------------------------------

There are several ways to explicitly specify the LNC degree for a NKI kernel.
It can be specified using the ``NEURON_LOGICAL_NC_CONFIG`` environment variable
or via bracket syntax at the kernel call site.

.. note::

   When a NKI kernel is compiled as part of a compute graph, its LNC degree
   must match the global graph compilation ``--lnc`` degree used by ``neuronx-cc``.

Environment variable
~~~~~~~~~~~~~~~~~~~~

Suppose you have a kernel called ``lnc_test()``. You can specify
its LNC degree by setting ``NEURON_LOGICAL_NC_CONFIG`` to 1 or 2.

.. code-block:: bash

   # Configure kernels to use LNC=1
   export NEURON_LOGICAL_NC_CONFIG=1

   # Launch lnc_test with LNC=1
   lnc_test(input)

Launch grid/bracket syntax
~~~~~~~~~~~~~~~~~~~~~~~~~~

For backwards-compatibility, the LNC degree may also be specified with bracket
syntax such as ``lnc_test[<degree>](...)``.

.. code-block:: python

   # Launch lnc_test with LNC=1
   lnc_test[1](input)

   # Launch lnc_test with LNC=2
   lnc_test[2](input)

If both the environment variable and bracket syntax are provided, bracket syntax takes precedence.

.. note::

   If neither ``NEURON_LOGICAL_NC_CONFIG`` nor bracket syntax specifies an
   LNC degree, the kernel implicitly defaults to LNC=1.

.. code-block:: python

   # Launch lnc_test with LNC=1 (implicit default)
   lnc_test(input)

Writing NKI kernels with LNC-specific APIs
------------------------------------------

A NKI kernel defines the instruction stream for each physical NeuronCore within an LNC.
NKI provides two main LNC-specific APIs:

* :func:`nl.num_programs() <nki.language.num_programs>` returns the number of
  program instances, identifying the LNC degree
* :func:`nl.program_id() <nki.language.program_id>` returns the current
  program's index, identifying the corresponding PNC within the LNC

For LNC=1, there is one program with index 0. For LNC=2, there are two
programs with indices 0 and 1. A kernel can assert its required LNC
degree:

.. code-block:: python

   @nki.jit
   def lnc2_kernel(input):
       assert nl.num_programs() == 2
       ...

With LNC=2, branching on ``nl.program_id()`` allows the PNC-specific
instruction streams to differ. The branch is evaluated separately for each
PNC at compile time, so it is not present in the final device instruction
streams.

In the following example, program 0 adds 1 to its input tile, while program 1
copies its input tile unchanged:

.. code-block:: python

   @nki.jit
   def different_instruction_streams(input):
       assert nl.num_programs() == 2
       assert input.shape[0] == 2

       pid = nl.program_id()
       in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
       out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
       output = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)

       nisa.dma_copy(dst=in_tile, src=input[pid])
       if pid == 0:
           nisa.tensor_scalar(
               dst=out_tile,
               data=in_tile,
               op0=nl.add,
               operand0=1.0,
           )
       else:
           nisa.tensor_copy(dst=out_tile, src=in_tile)
       nisa.dma_copy(dst=output[pid], src=out_tile)
       return output

The program ID can also select a slice of a tensor in shared HBM for each
program. For example, suppose you have a tensor with shape ``2x128x128`` and
want to compute the reciprocal of all of its elements:

.. code-block:: python

   @nki.jit
   def reciprocal_kernel(input):
       assert nl.num_programs() == 2

       # Check that the first dimension is 2 for this example
       assert input.shape[0] == 2

       # create temporary storage on SBUF for computation
       in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
       out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)

       # create output tensor in shared HBM
       output = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)

       shard_id = nl.program_id()
       nisa.dma_copy(in_tile, input[shard_id])
       nisa.reciprocal(out_tile, in_tile)
       nisa.dma_copy(output[shard_id], out_tile)
       return output

Final Notes
-----------

There are two things to be mindful of when using LNC=2.
First, the inputs and outputs of the kernel should be stored in the shared HBM
that all of the cores can access. Second, device-side control flow that
diverges across PNCs is not currently supported; such kernels may deadlock or
produce incorrect results.

.. warning::

   The following LNC=2 kernel is invalid. PNC 0 executes four loop iterations,
   while PNC 1 executes three.

   .. code-block:: python

      @nki.jit
      def divergent_control_flow():
          assert nl.num_programs() == 2
          pid = nl.program_id()

          result = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
          output = nl.ndarray(
              (2, 1, 1),
              dtype=nl.int32,
              buffer=nl.shared_hbm,
          )

          nisa.memset(dst=result, value=0)

          def loop_body(_):
              nisa.memset(dst=result, value=1)

          # Illegal: each PNC executes a different number of loop iterations.
          if pid == 0:
              nl.fori_loop(0, 4, loop_body)
          else:
              nl.fori_loop(0, 3, loop_body)

          nisa.dma_copy(dst=output[pid], src=result)
          return output
