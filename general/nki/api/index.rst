.. _nki_api_reference:

NKI API Reference Manual
===============================

Summary of different NKI API sets:

* **nki** top-level module contains APIs to decorate and simulate NKI kernels as well as
  NKI object types.
* **nki.language** consists of high-level compute and data movement APIs designed for ease-of-use. ``nki.language``
  allows NKI programmers to transition from NumPy/Triton implementation to NKI quickly without the need to *fully* understand
  underlying NeuronDevice architecture. Most language APIs invoke one or more ``nki.isa`` APIs (that is, NeuronDevice
  hardware instructions) under the hood.
* **nki.isa** consists of low-level APIs that highly resemble hardware instructions in NeuronDevice ISA
  (instruction set architecture) designed to provide fine
  control over the hardware. These APIs expose all the programmable input parameters of the corresponding
  hardware instructions and also enforce the same tile-size and layout requirements as specified in NeuronDevice ISA.

* Other documents:

  * **NKI API Common Fields** documents common NKI API input parameters such as data types and masks,
    as well as common API behavior such as type promotion.
  * **NKI API Errors** captures common error types that are thrown by the NKI kernel compilation *frontend*,
    including syntax, tile-size and layout violation errors.


.. toctree::
    :maxdepth: 2

    nki
    nki.language
    nki.isa
    nki.api.shared
    nki.errors
