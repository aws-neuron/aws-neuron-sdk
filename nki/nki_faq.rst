.. _nki_faq:

NKI FAQ
=========

When should I use NKI?
~~~~~~~~~~~~~~~~~~~~~~

NKI enables customers to self serve, onboard novel deep learning
architectures, and implement operators currently unsupported by
traditional ML Framework operators. With NKI, customers can experiment
with models and operators and can create unique differentiation.
Additionally, in cases where the compiler's optimizations are too
generalized for a developers' particular use case, NKI enables customers
to program directly against the Neuron primitives and therefore optimize
performance of existing operators that are not being compiled
efficiently.

Which AWS chips does NKI support?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI supports all families of chips included in AWS custom-built machine
learning accelerators, Trainium and Inferentia. This includes the second generation chips and beyond,
available in the following instance types: Inf2, Trn1, Trn1n and Trn2.

Which compute engines are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following AWS Trainium and Inferentia compute engines are
supported: Tensor Engine, Vector Engine, Scalar Engine, and GpSimd Engine.
For more details, see the :doc:`NeuronDevice Architecture Guide <nki_arch_guides>`,
and refer to :doc:`nki.isa <api/nki.isa>` APIs to identify which engines are utilized for each instruction.

How do I launch a NKI kernel onto a logical NeuronCore with Trainium2 from NKI?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A logical NeuronCore (LNC) can consist of multiple physical NeuronCores. In the current Neuron release, an LNC on Trainium2 can have up to two physical NeuronCores (subject to future changes).

For more details on NeuronCore configurations, see
`Logical NeuronCore configurations <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html#logical-neuroncore-config>`__.

In NKI, users can launch a NKI kernel onto multiple physical NeuronCores within a logical NeuronCore using single program, multiple data (SPMD) grids.

For a step-by-step guide, refer to the tutorial here:
:doc:`SPMD Tensor addition with multiple NeuronCores <tutorials/spmd_multiple_nc_tensor_addition>`.

What ML Frameworks support NKI kernels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI is integrated with :ref:`nki_framework_custom_op_pytorch` and :ref:`nki_framework_custom_op_jax`
frameworks. For more details, see the :ref:`nki_framework_custom_op`.

What Neuron software does not currently support NKI?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NKI does not currently support integration with
Neuron Custom C++ Operators, Transformers NeuronX, and Neuron Collective Communication.

Where can I find NKI sample kernels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI hosts an open source sample repository
`nki-samples <https://github.com/aws-neuron/nki-samples>`__ which
includes a set of reference kernels and tutorial kernels built by the
Neuron team and external contributors. For more information, see :ref:`nki_kernels` and :doc:`NKI tutorials <tutorials>`.

What should I do if I have trouble resolving a kernel compilation error?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refer to :doc:`NKI Error Manual <api/nki.errors>` for a detailed guidance on how
to resolve some of the common NKI compilation errors.

If you encounter compilation errors from Neuron Compiler that you cannot understand or
resolve, you may check out NKI sample `GitHub issues <https://github.com/aws-neuron/nki-samples/issues>`__
and open an issue if no similar issues exist.

How can I debug numerical issues in NKI kernels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We encourage NKI programmers to build kernels incrementally and verify output of small operators one at a time.
NKI also provides a CPU simulation mode that supports printing of kernel intermediate tensor values to the console.
See :doc:`nki.simulate <api/generated/nki.simulate_kernel>` for a code example.


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