.. |Trn1| replace:: :ref:`Trn1 <aws-trn1-arch>`
.. |Inf2| replace:: :ref:`Inf2 <aws-inf2-arch>`

.. _libneuronxla-rn:

Neuron XLA pluggable device (``libneuronxla``) release notes
================================================

.. contents:: Table of Contents
   :local:
   :depth: 1

``libneuronxla`` is a software package containing Neuron's integration into
the `PJRT <https://openxla.org/xla/pjrt_integration>`__ runtime, built using
the `PJRT C-API plugin <https://github.com/openxla/xla/blob/5564a9220af230c6c194e37b37938fb40692cfc7/xla/pjrt/c/docs/pjrt_integration_guide.md>`__
mechanism.

Release [2.0.4986.0]
--------------------
Date: 10/25/2024

Summary
~~~~~~~

This patch release removes the excessive lock wait time during neuron_parallel_compile graph extraction for large cluster training.

Release [2.0.4115.0]
----------------------
Date: 09/16/2024


Summary
~~~~~~~

This release of ``libneuronxla`` officially adds beta support for running JAX on AWS Trainium and Inferentia accelerators.


What’s new in this release
~~~~~~~~~~~~~~~~~~~~~~~~~~

Announcing beta Neuron support for JAX.

- Trainium and Inferentia as PJRT pluggable devices
- JAX 0.4.31 support (through PJRT C-API version 0.54)
