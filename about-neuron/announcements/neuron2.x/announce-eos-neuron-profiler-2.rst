.. post:: February 26, 2026
    :language: en
    :tags: announce-eos-neuron-profiler

.. _announce-eos-neuron-profiler-2:

Neuron Explorer Replaces Neuron Profiler, Starting with Neuron 2.29
-------------------------------------------------------------------

Starting with Neuron 2.29, **Neuron Profiler and Profiler 2.0 (UI and CLI) will reach end of support** and be replaced by Neuron Explorer. If you are currently using the Neuron Profiler, migrate to Neuron Explorer before the Neuron 2.29 release.

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

    Neuron strongly recommends migrating to Neuron Explorer **before** the Neuron 2.29 release.

There are two things to be aware of when migrating:

* **Existing NTFF profile files are supported**, but must be reprocessed before they can be viewed in the Neuron Explorer UI.
* **New features require new profiles.** To access the full set of Neuron Explorer capabilities, you must recapture your profiles using the updated tooling.

For detailed migration steps, see the :doc:`/tools/neuron-explorer/migration-faq` and the :ref:`Neuron Explorer FAQ <neuron-explorer-faq>`.

What happens to Neuron Profiler after Neuron 2.29?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After Neuron 2.29, Neuron Profiler will:

* **No longer receive** bug fixes, feature updates, or technical support.
* **No longer be distributed** as part of the Neuron SDK.

If you need to continue using Neuron Profiler temporarily, you must pin your environment to Neuron 2.28 or earlier. This is **not recommended**, as you will not receive any SDK updates or security fixes.
