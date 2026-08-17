.. meta::
   :description: nrtpy error handling — the exception hierarchy raised by the Pythonic AWS Neuron Runtime bindings.
   :date-modified: 2026-07-30
   :keywords: Neuron, nrtpy, NrtpyError, NrtError, exceptions, error handling

.. _nrtpy-errors-ref:

========================
nrtpy error handling
========================

This page documents the exception hierarchy raised by ``nrtpy``.

Exception hierarchy
-------------------

.. code-block:: text

   RuntimeError
   +-- NrtpyError        # nrtpy-level errors (e.g., using a freed tensor)
   +-- NrtError          # libnrt errors with preserved status code

Both exception classes inherit from Python's built-in ``RuntimeError``, so a
bare ``except RuntimeError`` catches all ``nrtpy`` errors.

NrtpyError
~~~~~~~~~~

.. py:exception:: nrtpy.NrtpyError

   Raised for errors detected at the ``nrtpy`` Python layer — for example,
   passing a freed tensor to an operation, or calling
   :py:func:`~nrtpy.configure` when the runtime is already active.

NrtError
~~~~~~~~

.. py:exception:: nrtpy.NrtError

   Raised when the underlying ``libnrt`` C API returns a non-success status
   code. The exception message includes the NRT status name and numeric code.

   Common causes:

   - Loading a NEFF file that does not exist or is corrupt.
   - Allocating a tensor larger than available device memory.
   - Executing a model with mismatched input/output tensor names.

Examples
--------

.. code-block:: python

   from nrtpy import NrtpyModel, NrtpyError, NrtError

   # Catch libnrt errors (e.g., file not found)
   try:
       model = NrtpyModel.load_from_neff("nonexistent.neff")
   except NrtError as e:
       print(f"NRT error: {e}")
       # "NRT Error NRT_FAILURE(1): Failed to load model"

   # Catch nrtpy-level errors (e.g., invalid state)
   try:
       # ... use a tensor after nrtpy.reset() ...
       pass
   except NrtpyError as e:
       print(f"nrtpy error: {e}")

   # Catch all nrtpy errors
   try:
       model(inputs={"x": some_tensor})
   except RuntimeError as e:
       print(f"Error: {e}")

Related reference
-----------------

- :ref:`nrtpy-model-ref` — model operations that can raise these exceptions.
- :ref:`nrtpy-configuration-ref` — ``configure()`` and ``reset()`` error
  conditions.
