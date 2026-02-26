.. _neuron-2-26-0-nxd-core:

.. meta::
   :description: The official release notes for the AWS Neuron SDK NxD Core component, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0: NxD Core release notes
=============================================

**Date of release**:  September 18, 2025

**Version**: 0.15.22259

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.26.0 release notes home <neuron-2-26-0-whatsnew>`

NxD Core inference improvements
-------------------------------

Non-distributed inference in parallel layers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Updated parallel layers to support non-distributed inference when parallel state isn't initialized.
In non-parallel environments, RowParallelLinear and ColumnParallelLinear now function as ``nn.Linear``,
and ``ParallelEmbedding``now functions as ``nn.Embedding``. This change enables you to simplify model code that
works on device and on CPU by enabling you to use the parallel layer in both cases.

Other improvements
^^^^^^^^^^^^^^^^^^

* Added a ``compiler_flag_hook`` argument to ModelBuilder, which you can use to override compiler flags
  for different submodels and buckets.

Bug fixes
---------

Here's what we fixed in 2.26.0:

Inference
^^^^^^^^^

* Added additional instance types to the ``hardware`` enum. For example, ``inf2`` now maps to ``trn1``.
* Other minor bug fixes and improvements.

Known issues
------------

*Something doesn't work. Check here to find out if we already knew about it. We hope to fix these soon!*

Inference
^^^^^^^^^

* At high batch size (>=32), we have observed performance degradation with ``shard-on-load`` for some models such as Llama3.1-8B. Our current recommendation is to disable this feature by enabling 
  ``save_sharded_checkpoint`` in ``NeuronConfig`` when you trace and compile the model.
* ``spmd_mode = True`` does not work when provided to the ``parallel_model_trace`` API. ``parallel_model_trace`` will be deprecated in the next Neuron SDK release.

Previous release notes
----------------------

* :ref:`neuron-2-25-0-nxd-core`
* :ref:`nxd-core_rn`
