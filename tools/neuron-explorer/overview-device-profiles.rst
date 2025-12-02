.. meta::
    :description: Learn about Neuron Explorer widgets for device profiling including timeline views, event details, annotations, and performance analysis tools.
    :date-modified: 12/02/2025

Device Viewer
===============

The Neuron device profiles show a hardware instruction level granularity of execution on a NeuronCore. The profiler will collect the timestamped start and end events that occur on the device into a NTFF. As a post-processing step, the profiler will correlate these events with information in the compiled NEFF to generate a detailed report of the hardware performance. The Neuron Profiler UI provides several different tools as "widgets" for an extensible and customizable workflow.

.. image:: /tools/profiler/images/device-profile-1.png

Widgets
-------

Device Timeline
~~~~~~~~~~~~~~~

The Device Timeline widget presents a timeline view of the device execution, including activity on the DMA and compute engines, Hardware FLOPs Utilization (HFU) and device memory utilization over time, and more.

.. image:: /tools/profiler/images/device-profile-2.png

Hover
^^^^^

.. image:: /tools/profiler/images/device-profile-3.png

Hover on events in the timeline to see important identifying information at a glance, such as the time window, the hierarchy, and the hardware instruction that was executed.

For more details, clicking the event will display the full details in the Event Details widget.

Color Scheme
^^^^^^^^^^^^

.. list-table::
   :header-rows: 0
   :widths: 50 50

   * - .. image:: /tools/profiler/images/device-profile-4.png
          :width: 100%
     - .. image:: /tools/profiler/images/device-profile-5.png
          :width: 100%

Instructions are color-coded according to their associated PyTorch operator. All instructions derived from the same PyTorch operator share an identical color.

.. note::
   In future releases, we will introduce more customizable options for color-coding.

Panning
^^^^^^^

.. image:: /tools/profiler/images/device-profile-6.gif

Panning is supported in a couple of ways:

* Left-clicking the x-axis and dragging it
* Spinning scroll-wheel while holding down shift
* With the keyboard:
    * A/D keys for left/right movement
    * Left/right arrow keys for left/right movement

The amount panned depends on the current zoom level.

Event Details
~~~~~~~~~~~~~

Upon clicking an event in the Instruction widget, all details related to the event will appear in the Event Detail widget. The information shown will be a superset of the information available on hover, allowing us to dive deeper into what is happening on the hardware.

* The Event Details table will populate with field data from clicked events from the instruction widget.
* When filtering by fields through the Search widget, all matching events will be rendered as pages in the Event Details. Users can navigate through each page to analyze data per each matching event.
  
.. image:: /tools/profiler/images/device-profile-7.png

Annotations
~~~~~~~~~~~

Users can create annotations by right-clicking on the Instruction widget. These annotations can be moved by clicking and dragging the vertical line, and will snap to the closest events when applicable.

The Annotations widget will show more details on all available annotations in the profile, such as the time difference and summary metrics that occur between two markers. The option of which two markers to compare is configurable in the Diff vs column. We can also quickly zoom in to the region between two markers by selecting the checkbox on the left. This widget also supports renaming, deleting, saving, and loading markers for better readability and collaboration.

.. image:: /tools/profiler/images/device-profile-8.png

Operator Table
~~~~~~~~~~~~~~

The Operator Table aggregates the hardware level metrics into framework layers and operations, such as the MFU and amount of data being moved. We can progressively expand each row to get a further breakdown of each nested operator.

Filters can be applied and columns can be sorted for more streamlined viewing.

.. image:: /tools/profiler/images/device-profile-9.png

Overall Summary
~~~~~~~~~~~~~~~

The Overall Summary displays performance metrics across the entire profile run, with metrics broken down into different categories such as by the NeuronCore engines. These can be used for quick insights into how well the model performed.

.. image:: /tools/profiler/images/device-profile-10.png

Current Selection Summary
~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to the Overall Summary, the Current Selection Summary provides metrics for the current time window. Zooming in and out in the Instruction widget will update the summary. This can be used in conjunction with the zoom feature of the Annotations widget for easy access to a region of interest.

.. image:: /tools/profiler/images/device-profile-11.png

Code Editor
~~~~~~~~~~~

Profiles that are uploaded with source code files enable users to quickly navigate between NKI and application level source code and the corresponding hardware level instructions.

In the Instruction widget, we can click on an event to highlight the source code line in the Code Editor. A (Ctrl/Cmd) + click on the event will scroll to the corresponding source code line.

In the Code Editor widget, clicking on a line in the source code will automatically highlight all associated events in the Instruction widget. Similarly, highlighting multiple lines of the source code will also highlight all events in the timeline.

.. image:: /tools/profiler/images/device-profile-12.png

See :ref:`neuron-explorer-source-code` for instructions on how to enable source code viewing.

Layout Customization
~~~~~~~~~~~~~~~~~~~~

Understanding and optimizing performance with the profiler can be overwhelming given the amount of information being processed and displayed. It is often useful to cross-reference different information, such as the device timeline with the application source code. With the widget-based profiler view, we can customize the layout to best fit a specific workflow. Each widget can be added, removed, dragged around, and resized. Once we are happy with the layout, it can be saved through the Layout dropdown at the top right. The layouts are not tied to a specific profile, so they can be loaded and re-used for future profiles as well.

.. image:: /tools/profiler/images/device-profile-13.png