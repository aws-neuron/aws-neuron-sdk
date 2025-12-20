.. meta::
   :description: Overview of Data Representations in NKI
   :date_updated: 12/02/2025

.. _nki-about-data:

==========================
Data Representation in NKI
==========================

This topic covers Data Representation and how it applies to developing with the AWS Neuron SDK.
This overview will describe how data appears to the NKI programmer, and how this data is organized on the NeuronDevice.

Representing data in NKI
------------------------

NKI represents data in NeuronCore's memory hierarchy with built-in ``tensor`` type.
A ``tensor`` is a multi-dimensional array which contains elements with
the same data type, or "dtype".

Programmers can pass ``tensor`` values in and out of NKI kernels, and declare or initialize ``tensor`` values in any memory within the NeuronDevice
(PSUM, SBUF, HBM) using APIs such as :doc:`nki.language.ndarray </nki/api/generated/nki.language.ndarray>` and :doc:`nki.language.zeros </nki/api/generated/nki.language.zeros>`.

A ``tensor`` value has a name, a shape that describes the number and size of
each of its dimensions, an element data type (or "dtype"), and a description of
the physical location of the underlying data on the NeuronCore. For example, a
matrix of 16-bit floating point numbers may have a shape of ``(128,64)``
indicating that there are 128 rows and 64 columns of numbers, and a dtype of
"bfloat16" describing the floating format.

The physical location of a ``tensor`` consists of a memory (HBM, SBUF, or
PSUM), and an offset and size for the underlying data. In the case of HBM
tensors, there is only one offset and size. However, for SBUF and PSUM tensors
there are two offsets and two sizes because those memories are two-dimensional.
The two offsets and sizes describe a rectangle in the underlying memory within
which the tensor data will live. For two-dimensional memories, the first
dimension is called the "partition dimension" and corresponds to the partitions
of the underlying memory. Using our example from above, if our 128x64 element
tensor was resident on the SBUF, then the partition offset and size could be 0
and 128 indicating that each row of the matrix corresponds to one partition of
the SBUF. The second offset and size could be, for instance, 1024 and 128,
indicating that each matrix row start 1024 bytes from the beginning of each
partition, and consumes 128 bytes (2 bytes for each 16-bit float), within each
partition.

Often, NKI programmers will not need to worry about the physical location of
tensors. When using high-level APIs such as
:doc:`nki.language.ndarray </nki/api/generated/nki.language.ndarray>`,
the physical location is assigned automatically by the NKI compiler. However,
more advanced kernels may directly control the relative physical locations of
tensors using the direct allocation APIs.

Input and output tensors from ML frameworks to NKI kernels will be tensors with
underlying memory type of ``hbm``. These tensors are placed in the HBM memory
prior to calling the NKI kernel. Intermediate tensors can be allocated using the
tensor creation APIs, for instance:

.. code-block::

   # Allocate 3D tensor on the SBUF
   x = nl.ndarray((128, 32, 512), dtype=nl.float32, buffer=nl.sbuf)

The above code creates a new 3D tensor on the SBUF memory with shape 128x32x512, and with
an element type of 32-bit floats. The physical location of this tensor will
be assigned by the NKI compiler, and the total amount of memory used will be:
``8,388,608 = 128 * 32 * 512 * 4``
