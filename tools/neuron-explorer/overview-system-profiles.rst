.. meta::
    :description: Learn about the System Profile in Neuron Explorer for analyzing system-level execution across instances and workers with runtime and hardware events.
    :date-modified: 01/30/2026

System Profile
================

The Neuron System Profile show a system-level granularity of execution across instances and workers in your workload. This provides visibility into Neuron Runtime API calls and ML framework function calls (PyTorch or JAX) to help identify bottlenecks in distributed workloads. The Neuron Explorer UI provides system-level widgets for an extensible and customizable workflow.

.. image:: /tools/images/neuron-explorer-system-viewer.png

System Trace Viewer
---------------------

The System Trace Viewer provides an interactive timeline interface with time range selection, configurable event grouping, system event details on hover, and linking of hardware events to Device Trace Viewer widgets.

You can see events in the Neuron Runtime and correlate them with hardware execution events on the Neuron Devices.

.. image:: /tools/images/system-timeline-widget.png

You can also see the device memory (HBM) allocations for each Neuron device over time. Hovering over these memory usage events shows a breakdown by usage category.

.. image:: /tools/images/system-timeline-widget-hbm-usage.png

Visualizing Host Device Transfers
-----------------------------------

The System Trace Viewer includes tracks for host-device data transfers, which are often a significant contributor to end-to-end latency and can reveal whether a workload is bottlenecked on PCIe traffic. ``host_to_device`` transfers move data from CPU RAM to device HBM (for example, loading model weights or input tensors), while ``device_to_host`` transfers move data in the reverse direction (for example, returning output tensors to the host). For each direction, two tracks are shown:

* A **transfer events** track that displays each individual transfer as a discrete event on the timeline. This is useful for correlating specific transfers with surrounding runtime and hardware events to understand when and why a transfer occurred.
* A **transfer bandwidth** track that plots the PCIe bandwidth consumed by transfers in that direction over time. This helps identify how close the workload is to the link's practical throughput limit.

.. image:: /tools/images/system-timeline-host-device-transfers.png

For distributed workloads, transfer tracks are aggregated by Instance ID and NeuronCore. For example, ``i-046decdee2ea88e5a/NC:0/host_to_device_transfers`` shows transfers from host to device on instance ``i-046decdee2ea88e5a`` that are required for model execution on NeuronCore 0 of that instance.

Other grouping options are available in the System Timeline settings. The default ``Instance + NeuronCore`` grouping renders one set of transfer tracks per NeuronCore on each instance in the workload, which is useful for pinpointing per-core behavior. The ``Instance`` grouping aggregates transfers across all NeuronCores on a given instance, producing a single set of tracks per instance, which is useful for comparing total host-device traffic between instances.


Adding Widgets
---------------
The System Profile supports both System and Device widgets, enabling multi-profile analysis, for example comparing annotated device events across different devices.

To add a widget:

1. Click the **Add Widget** button to open the Add Widget modal.
2. Select a Device or System widget.
3. Click a widget tile to load it with the selected profile. Each tile is tagged with its supported profile type (system, device, or both).

To load multiple instances of the same widget type for different profiles, repeat the steps above and select a different profile each time.

.. image:: /tools/images/system-timeline-add-widget.gif


After adding a widget, you can switch to a different profile by using the profile dropdown at the top of the widget.

.. image:: /tools/images/widget_switch_profiles.png

.. note::

   Adding duplicate widgets for the same profile is not currently supported.



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

.. image:: /tools/images/system-timeline-settings.png

Event Details
--------------

Clicking on trace events in the timeline populates the Event Details widget with a list of properties for the system trace event.

.. image:: /tools/images/system-event-details.png

Device Profile Linking
------------------------

The System Trace Viewer links hardware events to the Device Trace Viewer, which renders the corresponding device traces.

Navigating from the System Trace Viewer to a Device Trace Viewer can be accomplished in two ways:

Open the Device Profile List Modal
------------------------------------

To see a list of all device profiles captured during your workload:

1. **Click the "Device Profiles List" button** in the top right action bar of the System Trace Viewer to open a modal containing a list of device profiles
2. **Select a Device Profile and click Submit** to open the Device Trace Viewer with the selected device profile

.. image:: /tools/images/system-timeline-device-profiles-list-modal.png

Drill-down from Hardware Events
---------------------------------

To drill-down from a hardware event to the Device Trace Viewer:

1. Find a hardware event such as ``nc_exec_running``
2. Click on the hardware event
3. Wait for the Device Trace Viewer to open

This will open a new Device Trace Viewer with the selected device profile showing detailed hardware events. To learn about device profiles, see :doc:`Device Profiles in Neuron Explorer <overview-device-profiles>`.

.. image:: /tools/images/system-timeline-hardware-event-linking.gif

Dependency Chain Viewer
-----------------------

The Dependency Chain Viewer widget enables you to navigate upstream and downstream between related ``neuron_rt`` and ``neuron_hw`` events.
This helps you correlate runtime and hardware events to identify performance bottlenecks.
For example, you can select a runtime event and navigate to its related hardware events to understand where time is being spent.

Clicking on an event with dependencies populates the UI with the following elements:

- **Arrows** — curved arrows rendered in the System Trace Viewer display the flow between dependent events.
- **Upstream Events** — table of events preceding the selected event.
- **Downstream Events** — table of events following the selected event.

Using the Dependency Chain Viewer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Open the system profile in the Neuron Explorer UI.
2. Click on a system profile event that has upstream or downstream events to populate the Dependency Chain Viewer.
3. In the **Upstream Events** and **Downstream Events** tables, click on an event link to automatically scroll and focus the System Trace Viewer on that event.

.. image:: /tools/images/dependency_chain_viewer.png
