.. meta::
    :description: Learn about the System Trace Viewer in Neuron Explorer for analyzing system-level execution across instances and workers with runtime and hardware events.
    :date-modified: 01/30/2026

System Trace Viewer
===================

The Neuron system profiles show a system-level granularity of execution across instances and workers in your workload. This provides visibility into Neuron Runtime API calls and ML framework function calls (PyTorch or JAX) to help identify bottlenecks in distributed workloads. The Neuron Explorer UI provides system-level widgets for an extensible and customizable workflow.

.. image:: /tools/neuron-explorer/images/neuron-explorer-system-viewer.png

System Timeline
-----------------

The System Trace Viewer provides an interactive timeline interface with time range selection, configurable event grouping, system event details on hover, and linking of hardware events to Device Trace Viewer widgets.

.. image:: /tools/neuron-explorer/images/system-timeline-widget.png


Settings
----------

The System Trace Viewer supports multiple grouping modes to organize events for different analysis perspectives.
You can switch between the following grouping modes in the settings to focus your analysis on different aspects of system performance:

.. list-table:: Grouping Options
   :widths: auto
   :header-rows: 1
   :align: left

   * - Grouping Option
     - Description
     - Example
   * - CPU vs Device Grouping (Default)
     - Groups events by event source (CPU or Neuron device events)
     - Runtime events: ``i-0b1ea78ca2865fd32/PID:1765325/TID:0/neuron_rt``, Hardware events: ``i-0b1ea78ca2865fd32/PID:1765325/Worker:0/neuron_hw``
   * - NeuronCore Grouping
     - Groups events by individual NeuronCore
     - ``i-0b1ea78ca2865fd32/NC:0``, ``i-0b1ea78ca2865fd32/NC:1``
   * - Thread Grouping
     - Groups events by thread identifier
     - ``i-0b1ea78ca2865fd32/PID:1765325/TID:0``
   * - Process Grouping
     - Groups events by process identifier
     - ``i-0b1ea78ca2865fd32/PID:1765325``
   * - Instance Grouping
     - Groups all events by instance only
     - ``i-0b1ea78ca2865fd32``

.. image:: /tools/neuron-explorer/images/system-timeline-settings.png

Event Details
--------------

Clicking on trace events in the timeline populates the Event Details widget with a list of properties for the system profile event.

.. image:: /tools/neuron-explorer/images/system-event-details.png

Device Profile Linking
------------------------

The System Trace Viewer links hardware events to the Device Trace Viewer, which renders the corresponding device traces.

Navigating from the System Trace Viewer to a Device Trace Viewer can be accomplished in two ways:

Open the Device Profile List Modal
------------------------------------

To see a list of all device profiles captured during your workload:

1. **Click the "Device Profiles List" button** in the top right action bar of the System Trace Viewer to open a modal containing a list of device profiles
2. **Select a device profile and click Submit** to open the Device Trace Viewer with the selected device profile

.. image:: /tools/neuron-explorer/images/system-timeline-device-profiles-list-modal.png

Drill-down from Hardware Events
---------------------------------

To drill-down from a hardware event to the Device Trace Viewer:

1. Find a hardware event such as ``nc_exec_running``
2. Click on the hardware event
3. Wait for the Device Trace Viewer to open

This will open a new Device Trace Viewer with the selected device profile showing detailed hardware events. To learn about device profiles, see :doc:`Device Profiles in Neuron Explorer <overview-device-profiles>`.

.. image:: /tools/neuron-explorer/images/system-timeline-hardware-event-linking.gif
