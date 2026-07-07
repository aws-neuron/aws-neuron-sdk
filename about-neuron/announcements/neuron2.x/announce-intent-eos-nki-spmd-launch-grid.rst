.. post:: June 28, 2026
    :language: en
    :tags: announce-intent-eos, nki

.. _announce-intent-eos-nki-spmd-launch-grid:

Announcing upcoming deprecation of the SPMD launch grid requirement for LNC2 NKI kernels
-----------------------------------------------------------------------------------------

Launching an LNC2 NKI kernel currently relies on an SPMD launch grid whose dimension matches the LNC degree. This requirement will be deprecated in a future release. The behavior does not change in NKI 0.5.0 (Neuron 2.31.0).

When this change takes effect:

#. the SPMD launch grid will no longer be required to launch an LNC2 kernel;
#. passing an SPMD dimension that differs from the LNC degree in use will raise a compile-time error; and
#. regardless of whether an SPMD dimension is specified, the kernel will always be specialized across all physical NeuronCores (PNCs) in the LNC.

Kernels that continue to pass a matching SPMD grid will keep working for backwards compatibility. For more information, see :doc:`LNC Overview </nki/get-started/about/lnc>`.
