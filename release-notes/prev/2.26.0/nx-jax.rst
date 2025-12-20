.. _neuron-2-26-0-jax:

.. meta::
   :description: The official release notes for the AWS Neuron SDK JAX support component, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0: JAX support release notes
================================================

**Date of release**:  September 18, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.26.0 release notes home <neuron-2-26-0-whatsnew>`

Released versions
-----------------
* ``0.6.2.1.0.*``

Improvements
------------

* This release introduces support for JAX version ``0.6.2``.

Known issues
------------

* The ``Threefry`` RNG algorithm is not completely supported. Use the ``rbg`` algorithm instead. This can be configured by setting the following config option: ``jax.config.update("jax_default_prng_impl", "rbg")``
* For JAX versions older than ``0.4.34``, caching does not work out of the box. Use this code to enable caching support:
  
  .. code:: python
    
    import jax
    import jax_neuronx
    from jax._src import compilation_cache

    compilation_cache.set_cache_dir('./cache_directory')

* For JAX versions older than ``0.4.34``, buffer donation does not work out of the box. Add the following snippet to your script to enable it * ``jax._src.interpreters.mlir._platforms_with_donation.append('neuron')``
* Mesh configurations which use non-connected Neuron cores may crash during execution. You may observe compilation or Neuron runtime errors for such configurations. Device connectivity can be determined by using ``neuron-ls --topology``.
* Not all dtypes supported by JAX work on Neuron. Check :ref:`neuron-data-types` for supported data types.
* ``jax.random.randint`` does not produce expected distribution of randint values. Run it on CPU instead.
* Dynamic loops are not supported for ``jax.lax.while_loop``. Only static while loops are supported.
* ``jax.lax.cond`` is not supported.
* Host callbacks are not supported. As a result APIs based on callbacks from ``jax.debug`` and ``jax.experimental.checkify`` are not supported.
* ``jax.dlpack`` is not supported.
* ``jax.experimental.sparse`` is not supported.
* ``jax.lax.sort`` only supports comparators with LE, GE, LT and GT operations.
* ``jax.lax.reduce_precision`` is not supported.
* Certain operations (for example, rng weight initialization) might result in slow compilations. Try to run such operations on the CPU backend or by setting the following environment variable: ``NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1``.
* Neuron only supports ``float8_e4m3`` and ``float8_e5m2`` for FP8 dtypes.
* Complex dtypes (``jnp.complex64`` and ``jnp.complex128``) are not supported.
* Variadic reductions are not supported.
* Out-of-bounds access for scatter/gather operations can result in runtime errors.
* Dot operations on ``int`` dtypes are not supported.
* ``lax.DotAlgorithmPreset`` is not always respected. Dot operations occur in operand dtypes. This is a configurable parameter for ``jax.lax.dot`` and ``jax.lax.dot_general``.

Previous release notes
----------------------

* :ref:`neuron-2-25-0-jax`
* :ref:`jax-neuronx-rn`
