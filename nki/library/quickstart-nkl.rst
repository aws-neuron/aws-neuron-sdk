.. meta::
   :description: Get started with the NKI Library - install the pre-built NKI kernel package and use it with the Neuron compiler (neuronx-cc) in your model development.
   :keywords: NKI Library, nkilib, NKI, AWS Neuron, neuronx-cc, prebuilt kernels, quickstart
   :date-modified: 2026-06-15

.. _nki-library-quickstart:

Get started with the NKI Library
================================

The NKI Library provides pre-built reference kernels you can use directly in
your model development with the AWS Neuron SDK and NKI. This quickstart walks
you through installing the standalone ``nki-library`` package and using its
kernels with the Neuron compiler. When you finish, the library's kernels will
be active in your environment with no code changes required.

**This quickstart is for**: ML developers using AWS Neuron who want to use the
latest NKI Library kernels or contribute kernel changes.

**Time to complete**: ~10 minutes

Bundled nkilib vs. the standalone package
-----------------------------------------

The Neuron compiler ships a bundled copy of the NKI Library inside
``neuronx-cc``, available under the ``nkilib`` Python namespace (for example,
``import nkilib``). This *bundled nkilib* has been validated against that
specific compiler version and works out of the box — most users need nothing
more.

Install the standalone ``nki-library`` package only when you want to use the
latest kernels or contribute a kernel change.

.. note::

   Unlike bundled nkilib, kernels from the standalone package are **not**
   guaranteed to be compatible with the latest release of the Neuron
   compiler. To start from a known-good commit for your compiler version,
   check out the branch in the `NKI Library repository
   <https://github.com/aws-neuron/nki-library>`_ that corresponds to your
   compiler version.

Prerequisites
-------------

* The Neuron SDK installed, including the Neuron compiler (``neuronx-cc``).
  If you haven't set this up, see the :doc:`Neuron Quick Start Guide
  </about-neuron/quick-start/index>`.
* A Python virtual environment for your project.
* Basic familiarity with NKI. If you're new, start with
  :ref:`Get started with NKI <nki-get-started>`.

Step 1: Confirm the Neuron compiler is installed
------------------------------------------------

In this step, you confirm that ``neuronx-cc`` is available in your
environment. In most cases it is already installed as part of the Neuron SDK.

Verify it is importable:

.. code-block:: bash

   python -c "import neuronxcc; print('neuronx-cc OK')"

If this fails, install the Neuron SDK first using the
:doc:`Neuron Quick Start Guide </about-neuron/quick-start/index>`.

Step 2: Install the NKI Library package
---------------------------------------

In this step, you install the standalone ``nki-library`` package into the
**same virtual environment** as the rest of your project:

.. code-block:: bash

   pip install nki-library

Installing into the same environment as ``neuronx-cc`` is what allows the
package to take effect in the next step.

Step 3: Use the kernels
------------------------

In this step, you import and use kernels as you normally would. The package
automatically replaces the bundled nkilib kernels with the content of the
installed package — **no code changes are required**.

.. code-block:: python

   import nkilib

   # Use NKI Library kernels in your model as usual.

Confirmation
------------

After installing, the next invocation of ``neuronx-cc`` uses the kernels from
the standalone package instead of the bundled copy. You can confirm the
package is installed in the active environment:

.. code-block:: bash

   pip show nki-library

Congratulations! The NKI Library kernels are now active in your environment.
If you ran into trouble, see **Common issues** below.

Controlling which package gets loaded
-------------------------------------

To *temporarily* revert to the bundled version of nkilib, set the
``NKILIB_FORCE_BUNDLED_LIBRARY`` environment variable to a truthy value:

.. code-block:: bash

   export NKILIB_FORCE_BUNDLED_LIBRARY=true

On its next execution, ``neuronx-cc`` uses the bundled version of nkilib. To
return to the kernels from the standalone package, unset the variable:

.. code-block:: bash

   unset NKILIB_FORCE_BUNDLED_LIBRARY

Uninstalling
------------

To uninstall the standalone package, run:

.. code-block:: bash

   pip uninstall nki-library

After uninstalling, the compiler falls back to the bundled nkilib.

Common issues
-------------

- **A kernel behaves unexpectedly or fails to compile after installing.** The
  standalone package isn't guaranteed to match your compiler version. Check
  out the `NKI Library repository <https://github.com/aws-neuron/nki-library>`_
  branch that corresponds to your ``neuronx-cc`` version.
- **The installed kernels don't seem to take effect.** Confirm the package is
  installed in the *same* virtual environment as ``neuronx-cc``, and that
  ``NKILIB_FORCE_BUNDLED_LIBRARY`` is not set.
- **You need to rule out the standalone package while debugging.** Set
  ``NKILIB_FORCE_BUNDLED_LIBRARY=true`` to force the bundled version, then
  unset it to switch back.

Next steps
----------

* :doc:`NKI Library supported kernel reference </nki/library/api/index>` —
  functions, parameters, and usage for each pre-built kernel.
* :ref:`Get started with NKI <nki-get-started>` — write and run your own NKI
  kernels.

Further reading
---------------

- `NKI Library GitHub repository <https://github.com/aws-neuron/nki-library>`_
- :ref:`NKI Library documentation home <nkl_home>`
