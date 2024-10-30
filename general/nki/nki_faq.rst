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
learning accelerators, Trainium and Inferentia. This includes the second
generation of NeuronCore-v2 and the following instances: Inf2, Trn1, Trn1n.

Which hardware engines are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following AWS Trainium and Inferentia hardware engines are
supported: Tensor Engine, Vector Engine, Scalar Engine, and GpSIMD
Engine. For more details, see the :ref:`Trainium/Inferentia2 Architecture Guide <trainium_inferentia2_arch>`.

What ML Frameworks support NKI kernels?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NKI is integrated with :ref:`nki_framework_custom_op_pytorch` and :ref:`nki_framework_custom_op_jax`
frameworks. For more details, see the :ref:`nki_framework_custom_op`.

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
resolve, you may check out NKI sample `Github issues <https://github.com/aws-neuron/nki-samples/issues>`__
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
For more information, please see the
`SDK Maintenance Policy <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/sdk-policy.html>`__.