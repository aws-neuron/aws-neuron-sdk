.. meta::
    :description: Complete release notes for the JAX NeuronX component across all AWS Neuron SDK versions.
    :keywords: jax neuronx, jax, release notes, aws neuron sdk
    :date-modified: 09/18/2025

.. _jax_rn:

Component Release Notes for JAX NeuronX
========================================

The release notes for the JAX NeuronX component. Read them for the details about the changes, improvements, and bug fixes for all release versions of the AWS Neuron SDK.

.. _jax-2-26-0-rn:

JAX NeuronX [0.6.2.1.0.*] (Neuron 2.26.0 Release)
---------------------------------------------------

Date of Release: 09/18/2025

Improvements
~~~~~~~~~~~~~~~

* This release introduces support for JAX version 0.6.2.

Known Issues
~~~~~~~~~~~~

* The Threefry RNG algorithm is not completely supported. Use the rbg algorithm instead. This can be configured by setting the following config option: jax.config.update("jax_default_prng_impl", "rbg").
* For JAX versions older than 0.4.34, caching does not work out of the box.
* For JAX versions older than 0.4.34, buffer donation does not work out of the box.
* Mesh configurations which use non-connected Neuron cores may crash during execution.
* Not all dtypes supported by JAX work on Neuron.
* jax.random.randint does not produce expected distribution of randint values. Run it on CPU instead.
* Dynamic loops are not supported for jax.lax.while_loop. Only static while loops are supported.
* jax.lax.cond is not supported.
* Host callbacks are not supported.
* jax.dlpack is not supported.
* jax.experimental.sparse is not supported.
* jax.lax.sort only supports comparators with LE, GE, LT and GT operations.
* jax.lax.reduce_precision is not supported.
* Certain operations might result in slow compilations.
* Neuron only supports float8_e4m3 and float8_e5m2 for FP8 dtypes.
* Complex dtypes (jnp.complex64 and jnp.complex128) are not supported.
* Variadic reductions are not supported.
* Out-of-bounds access for scatter/gather operations can result in runtime errors.
* Dot operations on int dtypes are not supported.
* lax.DotAlgorithmPreset is not always respected.


----

.. _jax-2-25-0-rn:

JAX NeuronX [0.6.1.1.0.*] (Neuron 2.25.0 Release)
---------------------------------------------------

Date of Release: 07/31/2025

Improvements
~~~~~~~~~~~~~~~

* This release introduces support for JAX version 0.6.1.

Bug Fixes
~~~~~~~~~

* Previously, using multiple meshes within a single program wasn't supported. This is fixed to add support for sub-meshes.

Known Issues
~~~~~~~~~~~~

* Known issues are listed at jax-neuron-known-issues.


----

.. _jax-2-24-0-rn:

JAX NeuronX [0.6.0.1.0.*] (Neuron 2.24.0 Release)
---------------------------------------------------

Date of Release: 06/20/2025

Improvements
~~~~~~~~~~~~~~~

* This release supports JAX versions up to ``0.6.0``.
* Known issues are listed within :ref:`jax-neuron-known-issues`.

Known Issues
~~~~~~~~~~~~

* Known issues are listed within :ref:`jax-neuron-known-issues`.


----

.. _jax-2-23-0-rn:

JAX NeuronX [0.5.3.1.0.*] (Neuron 2.23.0 Release)
---------------------------------------------------

Date of Release: 05/20/2025

Improvements
~~~~~~~~~~~~~~~

* This release supports JAX versions up to ``0.5.3``.
* Known issues are listed within :ref:`jax-neuron-known-issues`.

Breaking Changes
~~~~~~~~~~~~~~~~

* ``jax_neuronx.nki_call`` is no longer supported. Use ``neuronxcc.nki.jit`` instead.

Known Issues
~~~~~~~~~~~~

* Known issues are listed within :ref:`jax-neuron-known-issues`.


----

.. _jax-2-22-0-rn:

JAX NeuronX [0.1.3] (Neuron 2.22.0 Release)
---------------------------------------------

Date of Release: 04/03/2025

Improvements
~~~~~~~~~~~~~~~

* This release supports JAX versions up to ``0.5.0``.
* Known issues are listed within :ref:`jax-neuron-known-issues`.

Known Issues
~~~~~~~~~~~~

* Known issues are listed within :ref:`jax-neuron-known-issues`.


----

.. _jax-2-21-0-rn:

JAX NeuronX [0.1.2] (Neuron 2.21.0 Release)
---------------------------------------------

Date of Release: 12/20/2024

Improvements
~~~~~~~~~~~~~~~

* This release supports JAX versions up to ``0.4.35``.
* Support for JAX versions up to ``0.4.35``.
* Support for JAX caching API for versions ``0.4.30+``.


----

.. _jax-2-20-0-rn:

JAX NeuronX [0.1.1] (Neuron 2.20.0 Release)
---------------------------------------------

Date of Release: 09/16/2024

Improvements
~~~~~~~~~~~~~~~

* This is the initial beta release of JAX NeuronX that contains Neuron-specific JAX features, such as the Neuron NKI JAX interface.
* Announcing the first JAX NeuronX release.
* JAX interface for Neuron NKI.

Known Issues
~~~~~~~~~~~~

* None reported for this release.