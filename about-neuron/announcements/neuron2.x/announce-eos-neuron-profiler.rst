.. post:: December 16, 2025
    :language: en
    :tags: announce-eos-neuron-profiler

.. _announce-eos-neuron-profiler:

End of Support for Neuron Profiler and Neuron Profiler 2.0 UI and CLI coming in a future Neuron release
--------------------------------------------------------------------------------------------------------

What's changing
^^^^^^^^^^^^^^^^
Neuron will end support for the legacy Neuron Profiler and Neuron Profiler 2.0 UI and CLI tools in a coming release (planned for v2.29.0). We launched Neuron Explorer in Neuron SDK 2.27, replacing these tools with a unified developer experience that will include device and system profiling in a single view, eager mode support, enhanced memory profiling, improved visualization capabilities, as well as support for the full developer lifecycle.

Why are we making this change
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consolidating to Neuron Explorer allows us to focus development efforts on a single, modern profiling solution while providing you with enhanced features and a better user experience.

How does this impact you
^^^^^^^^^^^^^^^^^^^^^^^^^

If you are currently using the legacy Neuron Profiler UI or CLI, please do the following before Neuron 2.29:

* Begin using Neuron Explorer (available since Neuron 2.27). See https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/get-started.html#
* Reprocess your existing NTFF files for the new UI: see https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/how-to-profile-workload.html

Note: Neuron Explorer is backwards compatible with existing Profiler NTFF files, but they must be reprocessed to view in the new UI. For new features (eager mode, memory viewer, certain NKI tools), you'll need to recapture profiles.

After Neuron 2.29.0 releases (planned):

* Legacy UI will no longer receive bug fixes, updates, or technical support
* To continue using legacy UI, you must pin to the last version that supports it (not recommended)

