.. post:: October 16, 2026
    :language: en
    :tags: announce-nki-specialization-change

.. _announce-nki-specialization-change:

Announcing changes to NKI specialization
========================================

Starting with :ref:`Neuron Release 2.33.0 <neuron-2-33-0-whatsnew>` (NKI 0.7.0), NKI
specializes kernels by tracing them with Python. During tracing, each NKI API call records
an operation in an intermediate representation (IR), which NKI then compiles. Previous
versions specialized kernels with dedicated compiler lowering passes. Compilation and
execution are unchanged. For how specialization, compilation, and execution fit together,
see the :ref:`NKI Language Guide <nki-language-guide>`.

Kernel developers can now use the full Python language during specialization, including
generators and class inheritance. NKI meta-programming follows standard Python semantics.

Kernels that use ``nl.dynamic_range`` or ``while`` loops with a virtual register must move
to :func:`nki.language.fori_loop` and :func:`nki.language.while_loop`. Kernels that use
``==`` on user-defined classes can behave differently. For details, see
:ref:`Migrating from NKI 0.6.0 to NKI 0.7.0 <nki-migrate-0-6-to-0-7>`.

To keep the previous specialization while migrating a kernel, set
``NKI_COMPILER_FRONTEND=custom``. This setting is deprecated in Neuron 2.33.0 and will be
removed along with the dedicated lowering passes in the next Neuron release.
