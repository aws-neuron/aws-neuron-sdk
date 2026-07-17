.. _nki_faq:

NKI FAQ
=========

When should I use NKI?
~~~~~~~~~~~~~~~~~~~~~~

NKI lets you write custom operators that program directly against the Neuron ISA.
There are two common reasons to use NKI:

* **Performance optimization**: When the Neuron Compiler's general-purpose optimizations
  don't fully exploit the hardware for your specific workload, NKI lets you write
  hand-tuned operators that maximize compute and memory throughput. For example,
  the NKI Library provides optimized kernels for attention, MLP, RMSNorm with
  quantization, and collective communication that outperform compiler-generated
  equivalents.

* **Novel operators and architectures**: NKI enables you to implement operators that
  are not yet supported by the Neuron Compiler, letting you self-serve new deep learning
  architectures and custom operations without waiting for compiler support.

Which AWS chips does NKI support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI supports Trainium2 and Trainium3 chips,
available in the following instance types: Trn2 and Trn3.

Which compute engines are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following AWS Trainium and Inferentia compute engines are
supported: Tensor Engine, Vector Engine, Scalar Engine, and GpSimd Engine.
For more details, see the :doc:`NeuronDevice Architecture Guide </nki/guides/architecture/index>`,
and refer to :doc:`nki.isa <api/nki.isa>` APIs to identify which engines are utilized for each instruction.

How do I launch a NKI kernel onto a logical NeuronCore with Trainium2 or Trainium3 from NKI?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A logical NeuronCore (LNC) can consist of multiple physical NeuronCores. In the current Neuron release, an LNC on Trainium2 or Trainium3 can have up to two physical NeuronCores.

For more details on NeuronCore configurations, see
`Logical NeuronCore configurations <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html#logical-neuroncore-config>`__.

In NKI, users can launch a NKI kernel onto multiple physical NeuronCores within a logical NeuronCore using ``kernel[2]`` to set LNC=2. Each core receives a different ``nl.program_id(0)`` value (0 or 1).

What ML Frameworks support NKI kernels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI is integrated with :ref:`nki_framework_custom_op_pytorch` and :ref:`nki_framework_custom_op_jax`
frameworks. For more details, see the :ref:`nki_framework_custom_op`.

Where can I find NKI sample kernels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI provides two open source repositories with kernel examples:

* `NKI Library <https://github.com/aws-neuron/nki-library>`__ — Production-ready, optimized kernels for common operations (matrix multiplication, attention, normalization, quantization, etc.) that you can use directly in your models. See the :doc:`NKI Library documentation </nki/library/index>` for API reference and design specifications.

* `NKI Samples <https://github.com/aws-neuron/nki-samples>`__ — Reference and tutorial kernels that demonstrate NKI programming patterns and concepts. These are designed for learning and experimentation rather than production use.

For step-by-step guides on writing NKI kernels, see the :doc:`NKI tutorials </nki/guides/tutorials/index>`.

What should I do if I have trouble resolving a kernel compilation error?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refer to the `NKI sample GitHub issues <https://github.com/aws-neuron/nki-samples/issues>`__ for guidance on
resolving common NKI compilation errors.

If you encounter compilation errors from Neuron Compiler that you cannot understand or
resolve, you may check out NKI sample `GitHub issues <https://github.com/aws-neuron/nki-samples/issues>`__
and open an issue if no similar issues exist.

How can I debug numerical issues in NKI kernels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We encourage NKI programmers to build kernels incrementally and verify output of small operators one at a time.
NKI also provides a CPU simulation mode that supports printing of kernel intermediate tensor values to the console.
See :doc:`nki.simulate </nki/api/generated/nki.simulate>` for a code example.


How can I optimize my NKI kernel?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To learn how to optimize your NKI kernel, see the :ref:`nki_perf_guide`.

Does NKI support entire Neuron instruction set?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neuron will iteratively add support for the Neuron
instruction set through adding more :doc:`nki.isa <api/nki.isa>` (Instruction Set
Architecture) APIs in upcoming Neuron releases.


Will NKI APIs guarantee backwards compatibility?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :doc:`NKI APIs <api/index>` follow the Neuron Software Maintenance policy for Neuron APIs.
For more information, see the
`SDK Maintenance Policy <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/sdk-policy.html>`__.