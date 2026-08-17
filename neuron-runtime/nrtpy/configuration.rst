.. meta::
   :description: nrtpy runtime configuration — set visible NeuronCores and manage the AWS Neuron Runtime singleton from Python.
   :date-modified: 2026-07-30
   :keywords: Neuron, nrtpy, configure, reset, runtime singleton, NEURON_RT_VISIBLE_CORES

.. _nrtpy-configuration-ref:

========================
nrtpy configuration
========================

This page documents the module-level functions that configure and manage the
``nrtpy`` runtime singleton.

Overview
--------

``nrtpy`` uses a single-runtime execution model: one Python process holds one
runtime singleton managing one ``libnrt`` instance. The singleton is created
lazily the first time you use an :py:class:`~nrtpy.NrtpyTensor` or
:py:class:`~nrtpy.NrtpyModel`.

- Call :py:func:`~nrtpy.configure` **before** the first runtime operation to
  set which NeuronCores are visible.
- Call :py:func:`~nrtpy.reset` to close the runtime and clear configuration
  state. After ``reset()``, you may call :py:func:`~nrtpy.configure` again
  before the next operation.

Functions
---------

.. py:function:: nrtpy.configure(visible_cores=None)

   Set runtime configuration before initialization. This only sets the
   ``NEURON_RT_VISIBLE_CORES`` environment variable — it does not initialize
   the runtime. The runtime is initialized lazily on the first ``nrtpy``
   operation after ``configure()``.

   Can only be called when the runtime is not active — either before the first
   ``nrtpy`` operation, or after :py:func:`~nrtpy.reset` (which closes the
   runtime and clears state). Calling ``configure()`` is optional; if not
   called, the runtime uses the system default (all NeuronCores visible).

   :param visible_cores: Iterable of NeuronCore IDs, for example ``[0, 1, 2]``
      or ``range(4)``. If ``None``, visible cores are left unchanged (uses the
      current ``NEURON_RT_VISIBLE_CORES`` value or system default).
   :type visible_cores: collections.abc.Iterable[int] or None
   :raises RuntimeError: If the runtime is already active. Call
      :py:func:`~nrtpy.reset` first.
   :raises TypeError: If ``visible_cores`` is an ``int`` rather than an
      iterable, or contains a non-integer value.
   :raises ValueError: If any core ID is negative.

.. py:function:: nrtpy.reset()

   Close the current runtime and clear configuration state. If the runtime has
   not been initialized yet, this is a no-op. Call :py:func:`~nrtpy.configure`
   afterward to set new visible cores before the next ``nrtpy`` operation.

   .. warning::

      All existing :py:class:`~nrtpy.NrtpyTensor` and
      :py:class:`~nrtpy.NrtpyModel` objects become invalid after this call.
      Any operations on them will raise errors.

Example
-------

.. code-block:: python

   import nrtpy
   from nrtpy import NrtpyModel, NrtpyTensor

   # Configure visible NeuronCores before first use
   nrtpy.configure(visible_cores=[0, 1])

   # The singleton is created lazily on first operation
   model = NrtpyModel.load_from_neff("model.neff")
   tensor = NrtpyTensor.from_numpy(data, name="input")

   # Switch to different cores: reset, then reconfigure
   del model, tensor
   nrtpy.reset()
   nrtpy.configure(visible_cores=[2, 3])

   # New runtime uses cores 2 and 3
   model2 = NrtpyModel.load_from_neff("model.neff", core_id=0)

Related reference
-----------------

- :ref:`nrtpy-model-ref` — load and execute models on configured cores.
- :ref:`nrtpy-tensor-ref` — allocate tensors on specific cores.
