.. _neuron-profiler-migration-guide:

Migration Guide from Neuron Profiler to Neuron Explorer
========================================================

This guide provides detailed information for migrating from Neuron Profiler or Neuron Profiler 2.0 to Neuron Explorer.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Neuron Explorer is the recommended profiling tool for AWS Neuron workloads, replacing both Neuron Profiler and Neuron Profiler 2.0. This guide helps you transition your profiling workflows to Neuron Explorer.

Key Differences
---------------

The following table summarizes the key differences between Neuron Profiler/Profiler 2.0 and Neuron Explorer:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1
   :align: left

   * - Feature
     - Neuron Profiler / Profiler 2.0
     - Neuron Explorer
   * - CLI tool
     - ``neuron-profile``
     - ``neuron-explorer``
   * - Device Profiling
     - Yes
     - Yes (enhanced)
   * - System Profiling
     - Yes (Profiler 2.0 only)
     - Yes
   * - Hierarchy Viewer
     - No
     - Yes
   * - Source Code Viewer
     - Yes (Device profiles)
     - Yes (Device profiles)
   * - AI Recommendation Viewer
     - No
     - Yes (for NKI profiles)
   * - IDE Integration
     - No
     - Yes (VSCode Extension)
   * - Database Viewer
     - No
     - Yes
   * - Tensor Viewer
     - No
     - Yes
   * - Additional Installation Requirements
     - InfluxDB installation required
     - None


Update CLI Commands
--------------------

Replace ``neuron-profile`` with ``neuron-explorer`` in your scripts and workflows. The following commands are subject to change before GA:

.. list-table::
   :widths: 50 50
   :header-rows: 1
   :align: left

   * - Neuron Profiler Command
     - Neuron Explorer Command
   * - ``neuron-profile view -d ./output``
     - ``neuron-explorer view -d ./output``
   * - ``neuron-profile view -n file.neff -s profile.ntff``
     - ``neuron-explorer view -n file.neff -s profile.ntff``
   * - ``neuron-profile capture -n file.neff -s profile.ntff``
     - ``neuron-explorer capture -n file.neff -s profile.ntff``


Frequently Asked Questions
--------------------------

Do I need to install InfluxDB for Neuron Explorer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No. Unlike Neuron Profiler, Neuron Explorer requires no external installation or setup.

How do I view existing profiles captured with Neuron Profiler?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Existing NEFF and NTFF files captured with Neuron Profiler are fully compatible with Neuron Explorer. To view them:

.. code-block:: bash

   # View a single device profile
   neuron-explorer view -n file.neff -s profile.ntff

The profiles will be reprocessed using Neuron Explorer's processing pipeline, which may provide additional insights not available in the original Neuron Profiler view.

How do I capture profiles with Neuron Explorer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron Explorer provides the ``neuron-explorer capture`` command for standalone NEFF profiling, similar to ``neuron-profile capture``:

.. code-block:: bash

   # Capture a device profile
   neuron-explorer capture -n file.neff -s profile.ntff

You can also use the framework profiling APIs or environment variables to capture profiles during your actual workload execution. For NKI kernel profiling, continue using the ``nki.benchmark`` or ``nki.profile`` APIs as documented in the :ref:`NKI profiling guide <use-neuron-profile>`.

What new features does Neuron Explorer provide?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Neuron Explorer introduces several new capabilities:

- **Hierarchy Viewer**: Visualize execution from model layers down to hardware operations. See :doc:`overview-hierarchy-view`.
- **Source Code Viewer**: Navigate between source code and profile data. See :doc:`how-to-link-view-source-code`.
- **AI Recommendation Viewer**: Get AI-powered optimization suggestions for NKI profiles. See :doc:`overview-ai-recommendations`.
- **Database Viewer**: Run custom queries on profiling data. See :doc:`overview-database-viewer`.
- **Tensor Viewer**: Examine tensor information including shapes and memory usage. See :doc:`overview-tensor-viewer`.
- **VSCode Extension**: View profiles directly in your IDE with native code linking support.
- **System Trace Viewer**: Enhanced system-level profiling visualization. See :doc:`overview-system-profiles`.

How do I get help during migration?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Review the :doc:`get-started` guide for initial setup
- See :doc:`how-to-profile-workload` for detailed capture and viewing instructions
- Check submitted issues and file new issues via the `AWS Neuron GitHub issues <https://github.com/aws-neuron/aws-neuron-sdk/issues>`_
