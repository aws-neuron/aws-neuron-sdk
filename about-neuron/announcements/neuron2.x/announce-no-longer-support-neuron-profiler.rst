.. post:: May 20, 2026
    :language: en
    :tags: announce-no-longer-support-neuron-profiler

.. _announce-no-longer-support-neuron-profiler:

Neuron Explorer replaces Neuron Profiler, starting with Neuron 2.30
--------------------------------------------------------------------

SNeuron Profiler and Profiler 2.0 (UI and CLI) are no longer supported. Neuron Profiler will no longer receive bug fixes, feature updates, or technical support. It will no longer be distributed as part of the Neuron SDK. If you are currently using the Neuron Profiler, migrate to Neuron Explorer.

For migration guidance, see the :doc:`/tools/neuron-explorer/migration-faq`.

What is Neuron Explorer?
~~~~~~~~~~~~~~~~~~~~~~~~

Neuron Explorer is the next-generation suite of tools, guiding developers through their development journey on Trainium. It enables ML performance engineers to:

* **Trace execution end-to-end** — from source code down to hardware operations.
* **Analyze model behavior at every layer of the stack** — with detailed breakdowns per operation, per core, and per device.
* **Profile distributed workloads** — with native support for multi-node and multi-worker analysis at scale.

For more details, see :doc:`/tools/neuron-explorer/index`.

How does this impact current Neuron Profiler users?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

    Neuron strongly recommends migrating to Neuron Explorer **before** the Neuron 2.30.0 release.

There are two things to be aware of when migrating:

* **Existing NTFF profile files are supported**, but must be reprocessed before they can be viewed in the Neuron Explorer UI.
* **New features require new profiles.** To access the full set of Neuron Explorer capabilities, you must recapture your profiles using the updated tooling.

For detailed migration steps, see the :doc:`/tools/neuron-explorer/migration-faq` and the :ref:`Neuron Explorer FAQ <neuron-explorer-faq>`.
If you need to continue using Neuron Profiler temporarily, you must pin your environment to Neuron 2.29 or earlier. This is **not recommended**, as you will not receive any SDK updates or security fixes.

