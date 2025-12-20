.. meta::
   :description: Overview of Neuron Logical Cores
   :date_updated: 12/12/2025

.. _nki-about-lnc:

Using Logical Neuron Cores (LNC)
================================

This topic covers how to use multiple neuron cores by launching your NKI kernel
on multiple cores at the same time. This overview will cover how to launch
kernels, and the basic methods for writing a kernel to run on multiple cores.

Logical Neuron Cores (LNC)
--------------------------

The Neuron SDK supports running NKI kernels on multiple logical cores. When
launching a kernel, you can opt to run the kernel on 1 or 2 logical cores. If
you choose to run on 2 logical cores, at runtime, your kernel will be run on
two physical cores (if available) that have shared HBM memory (see Trainium3
Architrecture <trainium3_arch> for more details on NeuronCores). These two
version can operate on different parts of the input data, increasing overall
performance of your kernel.

NKI gives you a few mechanisms to for using Logical Neuron Cores (LNC). We will
look briefly at each of these, specifically we will describe:

1. How to launch a kernel on multiple cores
2. How to tell if a kernel is running on multiple cores
3. How to tell which core a kernel is running on

Launching a kernel on multiple cores
-----------

To launch a NKI kernel on multiple cores, you specify the number of cores to
use, in square brackets, when calling the kernel. For example, suppose we have
a kernel called `lnc_test`, and we want to launch this kernel on two cores.

.. code-block::

   # Launch lnc_test on 2 cores
   lnc_test[2](input)

The bracket syntax must contain only one number, the number of cores to use.
If no brackets are given the number of cores defaults to 1. If the number is
too large for the current architecture, then you will receive an error.

.. code-block::

   # Launch lnc_test on 1 core
   lnc_test(input)

   # Launch lnc_test on 1 core
   lnc_test[1](input)

   # Launch lnc_test on 2 cores
   lnc_test[2](input)

   # Launch lnc_test on 8 cores (ERROR on current architecture)
   lnc_test[8](input)

Programming for multiple cores
-----

When writing a NKI kernel for multiple cores, there are two important APIs that
can be used to tell how many cores are being used and which core the current
instance is running on. These APIs are called `num_programs` and `program_id`.

The `num_programs` API will return the total number of cores the current kernel
is running on. If LNC is not being used, this API will return 1. So, we can
tell if we are running on multiple cores by inspecting the result of this
variable:

.. code-block::

   @nki.jit
   def lnc_test(input):
     if nl.num_programs() > 1:
       print("Running on multiple cores")
     else:
       print("Running on one core - no LNC")

   # Launch lnc_test on 1 core
   # prints "Running on one core - no LNC"
   lnc_test(input)

   # Launch lnc_test on 2 cores
   # prints "Running on multiple cores"
   lnc_test[2](input)

The `program_id` API will return the logical core id that the current
instance is running on. In the case of LNC=2, this API will return either 0
or 1. When not using LNC, this API will return 0. This API can be used to
programmatically divide work between multiple cores.

For example, suppose we have a tensor with shape `2x128x128` and we want to
compute the reciprocal of all of the elements of this tensor. We can write a
kernel function that is LNC-aware and can make use of extra cores when
available.

.. code-block::

   def lnc_test(input):
    # Check the first dimension is 2 for this example
    assert input.shape[0] == 2

    # create temporary storage on SBUF for comptation
    in_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)
    out_tile = nl.ndarray(input.shape[1:], input.dtype, buffer=nl.sbuf)

    # create output tensor
    output = nl.ndarray(input.shape, input.dtype, buffer=nl.shared_hbm)

    if nl.num_programs() == 1:
      # Not using multiple cores, process two tiles
      for i in range(2):
        nisa.dma_copy(in_tile, input[i])
        nisa.reciprocal(out_tile, in_tile)
        nisa.dma_copy(output[i], out_tile)
    else:
      # Using multiple cores, process tiles in parallel, one per core
      i = nl.program_id(0)
      nisa.dma_copy(in_tile, input[i])
      nisa.reciprocal(out_tile, in_tile)
      nisa.dma_copy(output[i], out_tile)
    return output

The code above has two cases, one for when we are not using LNC
(`num_programs` returns 1), and one for when we are using LNC=2
(`num_programs` returns 2). In the non-LNC case, there is a for loop that
processes each input tiles one after the other. However, in the LNC=2 case,
we can use the `program_id` API to query which core we are on. This API will
return either `0` or `1`. The code uses the `program_id` to have each core
process one of the two tiles, in parallel.

Final Notes
---

Using LNC can improve the performance of NKI kernels by leveraging multiple
NeuronCores. However, there are two things to be mindful of when using LNC.
First, the inputs and outputs of the kernel should be stored in the Shared HBM
that all of the cores can access. Second, the Neuron SDK assumes that when
running a kernel on multiple cores, the program on each core is "the same".
This means that each core is executing the same basic control flow as the other
cores. Most of the time, this requirement will be automatically satisfied by
the NKI compiler. However, if you use dynamic control flow, and this
control-flow is different on the different cores, then the behavior is
undefined, and you will likely receive an error at runtime.
