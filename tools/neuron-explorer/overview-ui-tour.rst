.. meta::
    :description: Guided overview of the Neuron Explorer UI — every viewer and tool grouped by use case with screenshots and descriptions.
    :keywords: neuron explorer, profiling, UI overview, performance analysis, debugging, memory, timeline, NKI, Trainium
    :date-modified: 2026-07-31

.. _neuron-explorer-ui-tour:

Neuron Explorer UI overview
============================

This page provides a guided tour of the Neuron Explorer UI. For each viewer and
tool, you'll find what it does and when to use it, grouped by workflow.

Neuron Explorer is available as a browser UI, a VS Code extension, and a CLI. The
viewers below cover the browser and VS Code UIs. For CLI-only workflows, see
:doc:`get-started`.

.. contents:: On this page
   :local:
   :depth: 2

UI navigation map
-----------------

The following diagram shows how the viewers and tools connect within the Neuron
Explorer UI. Use it as a navigation map.

.. image:: /tools/neuron-explorer/images/Neuron-Explorer-UI-NavMap.png
   :alt: Neuron Explorer UI navigation map showing all viewers and their relationships
   :target: ../../_images/Neuron-Explorer-UI-NavMap.png

*Click the image to enlarge.*

Navigation and workspace
-------------------------

The navigation provides multiple ways to access profiles and tools,
organized around common workflows.

Sidebar navigation
~~~~~~~~~~~~~~~~~~

The left sidebar provides quick access to profile management without leaving your
current workspace:

* **Favourites** — Profiles you have starred appear directly in the sidebar for
  one-click access. Star a profile from the overflow menu or the Profile Manager
  page.
* **Recently Viewed** — Your most recently opened profiles, ordered by last access
  time. Quickly switch between profiles you are actively working with.
* **Recently Uploaded** — Newly uploaded profiles appear here automatically. 
  Use this to find profiles you just uploaded.
* **Resources** — Links to documentation and getting started guides.

Click **Profile Manager** in the sidebar header to return to the full profile
management page for uploading, searching, and bulk operations.

Focus Mode
~~~~~~~~~~

Focus Mode hides the top and side bars to maximize the workspace
area for timeline analysis. Toggle Focus Mode from the button cluster in the
top-right corner of the navigation bar.

Use Focus Mode when you need fewer distractions or more space to do your work.

Widget Tab Visibility
~~~~~~~~~~~~~~~~~~~~~

Widget tabs can now be hidden for less visual clutter when viewing widgets.

To control how widget tab handles are displayed, open **Settings**, choose
**Workspace settings**, and set **Tab bar visibility** to **Always visible**,
**Hide single-tab bars**, or **Hide all tab bars**.

Notifications
~~~~~~~~~~~~~

Notifications tell you anything that's wrong or could be wrong with your profile.
Click the notification icon to view recent alerts, profile processing status updates,
and system messages without leaving your current workspace.

Performance overview and triage
-------------------------------

Start with these viewers when you open a profile for the first time. They give you a
high-level picture before you drill into details.

Summary Viewer
~~~~~~~~~~~~~~

Shows key performance metrics (MFU, HFU, MBU, compute/memory utilization),
performance insights with ranked recommendations, FLOP utilization breakdowns by
engine, DMA utilization, memory bandwidth breakdown, collective operation durations,
and NKI instruction coverage. Use it to quickly identify the top bottleneck and get
actionable next steps.

.. image:: /tools/images/explorer-summary-page.png
   :target: ../../_images/explorer-summary-page.png

**When to use:** First thing after loading a profile — identify the top bottleneck
and decide where to focus.

**Learn more:** :doc:`overview-summary-page`

Region Highlighter
~~~~~~~~~~~~~~~~~~

Automatically identifies and marks time regions in your profile based on collective
operations, operation hierarchy, or kernel stack frames. The Summary Viewer then
shows comparative metrics for each region so you can compare layers or kernels
without placing manual annotations.

.. image:: /tools/neuron-explorer/images/region-highlighter-fast-region-perf.png
   :target: ../../_images/region-highlighter-fast-region-perf.png

**When to use:** When you need to compare performance across model layers, kernels,
or collective boundaries without manually annotating the timeline.

**Learn more:** :doc:`overview-region-highlighter`

AI Recommendation Viewer
~~~~~~~~~~~~~~~~~~~~~~~~

Provides AI-powered bottleneck analysis and optimization recommendations for NKI
kernel profiles. Each report ranks 2–3 optimization opportunities by effort and
impact, with quantified metrics, implementation guidance, and expected speedup.
Requires Amazon Bedrock (opt-in, billed to your account).

.. image:: /tools/images/recommendation-view.png
   :target: ../../_images/recommendation-view.png

**When to use:** After profiling an NKI kernel — get specific optimization
suggestions without deep manual analysis.

**Learn more:** :doc:`overview-ai-recommendations`

Timeline analysis
-----------------

Use these viewers to understand execution flow over time at both the hardware and
system levels.

Device Trace Viewer
~~~~~~~~~~~~~~~~~~~

Displays hardware instruction-level execution on a NeuronCore as an interactive
timeline. Shows compute engine instructions, DMA operations, HFU/memory utilization
over time. Supports hover details, color-coding by PyTorch operator, panning, and
zooming. The core viewer for device-level performance analysis.

.. image:: /tools/images/device-profile-2.png
   :target: ../../_images/device-profile-2.png

**When to use:** Investigate hardware-level behavior — identify stalls, DMA/compute
overlap, and instruction-level bottlenecks.

**Learn more:** :doc:`overview-device-profiles`

System Trace Viewer
~~~~~~~~~~~~~~~~~~~

Shows system-level execution across instances and workers — Neuron Runtime API calls,
framework function calls (PyTorch/JAX), host-device transfers, and HBM memory usage
over time. Supports multiple grouping modes (CPU vs Device, NeuronCore, Thread,
Process, Instance) and links ``nc_exec_running`` events to the Device Trace Viewer.

.. image:: /tools/images/neuron-explorer-system-viewer.png
   :target: ../../_images/neuron-explorer-system-viewer.png

**When to use:** Investigate distributed workload behavior, host-device data transfer
bottlenecks, or framework overhead.

**Learn more:** :doc:`overview-system-profiles`

Hierarchy Viewer
~~~~~~~~~~~~~~~~

Displays execution organized by framework layers and HLO operations. You can
progressively drill down from model-level constructs to hardware instructions.
Right-click an operator to highlight its corresponding instructions in the Device
Trace Viewer.

.. image:: /tools/images/hierarchy-view-1.gif
   :target: ../../_images/hierarchy-view-1.gif

**When to use:** Map high-level model operations to their hardware execution — find
which layer or operator is responsible for a performance issue.

**Learn more:** :doc:`overview-hierarchy-view`

Dependency Chain Viewer
~~~~~~~~~~~~~~~~~~~~~~~

Navigates the dependency chain between system profile events. Click a system event to
see arrows showing upstream/downstream dependencies — from framework calls through
runtime to hardware execution. Part of the System Trace Viewer.

.. image:: /tools/images/dependency_chain_viewer.png
   :target: ../../_images/dependency_chain_viewer.png

**When to use:** Trace the causal chain of a slow event — understand what triggered
it and what it blocks.

**Learn more:** :doc:`overview-system-profiles` (Dependency Chain Viewer section)

Data inspection and querying
-----------------------------

Use these viewers to examine specific data — tensors, memory, or raw tables — in
detail.

Tensor Viewer
~~~~~~~~~~~~~

Lists all tensors in the NEFF file with their names, types (input/output/weight),
shapes, sizes, and SBUF loading statistics (DMA count, repeat factor, average/total
bytes). Enter a tensor name in the search bar to find all related DMA instructions in
the Device Trace Viewer.

.. image:: /tools/images/tensor-viewer-table.png
   :target: ../../_images/tensor-viewer-table.png

**When to use:** Verify tensors are loaded efficiently, find specific tensor DMA
patterns, or cross-reference tensor names with device-level operations.

**Learn more:** :doc:`overview-tensor-viewer`

Memory Viewer
~~~~~~~~~~~~~

Visualizes memory allocation patterns across SBUF partitions over time. Hover over
allocations to see timing, addresses, opcodes, and DMA queue info. Use it to identify
memory fragmentation and spill/reload opportunities.

.. image:: /tools/images/memory_viewer_overview.png
   :target: ../../_images/memory_viewer_overview.png
   :width: 80%

**When to use:** Debug memory fragmentation, analyze spill/reload patterns, or verify
memory compactness for NKI kernel optimization.

**Learn more:** :doc:`overview-memory-viewer`

Code correlation
----------------

Use these viewers to link profiling data back to your source code.

Source Code Viewer
~~~~~~~~~~~~~~~~~~

Links NKI/PyTorch source code and hardware instructions bidirectionally:

- In the Device Trace Viewer, Ctrl+click an event to open the Source Code Viewer
  at the corresponding source line.
- In the Source Code Viewer, click a line of code to highlight all linked events
  in the Device Trace Viewer.

.. image:: /tools/neuron-explorer/images/bidirectional-nav.gif
   :alt: Bidirectional navigation between Source Code Viewer and Device Trace Viewer
   :target: ../../_images/bidirectional-nav.gif

**When to use:** Trace a performance issue back to the exact source code line, or
find the hardware behavior of a specific code region.

**Learn more:** :doc:`how-to-link-view-source-code`

Measurement and comparison
--------------------------

Use these tools to measure, annotate, and compare specific regions of a profile.

Annotations
~~~~~~~~~~~

Right-click the Device Trace Viewer to place annotation markers. Annotations snap to
events and display time differences and summary metrics between any two markers. You
can save and load annotations for collaboration.

.. image:: /tools/images/device-profile-8.png
   :target: ../../_images/device-profile-8.png

**When to use:** Mark specific regions for measurement, compare before/after, or
share specific profile views with teammates.

**Learn more:** :doc:`overview-device-profiles` (Annotations section)

Operator Table
~~~~~~~~~~~~~~

Aggregates hardware metrics into framework layers and operations (MFU, data
movement). Expand rows to drill into nested operators. You can sort and filter
columns for streamlined viewing.

.. image:: /tools/images/device-profile-9.png
   :target: ../../_images/device-profile-9.png

**When to use:** Compare per-operator performance metrics across your model's layers
without manually selecting timeline regions.

**Learn more:** :doc:`overview-device-profiles` (Operator Table section)

Workspace and layout
--------------------

Customize your profiling workflow with these features.

Layout customization
~~~~~~~~~~~~~~~~~~~~

Add widgets to your workspace — drag, resize, and arrange them freely. Widgets
include the Device Trace Viewer, System Trace Viewer, Search, Event Details, and
more. Save named layouts and reload them across profiles for a consistent workflow.

.. image:: /tools/images/device-profile-13.png
   :target: ../../_images/device-profile-13.png

**When to use:** Set up your workspace once for a specific workflow (e.g., NKI
debugging, distributed profiling) and reuse it.

Multi-profile analysis
~~~~~~~~~~~~~~~~~~~~~~

Add multiple widgets side by side in a profile page to create your own analysis
dashboard. Each widget can show a different profile or data view. Use the profile
dropdown to switch profiles within any widget.

.. image:: /tools/images/widget_switch_profiles.png
   :target: ../../_images/widget_switch_profiles.png

**When to use:** Compare execution across NeuronCores or instances in a distributed
workload.

.. _neuron-explorer-dark-light-mode:

Dark and light mode
~~~~~~~~~~~~~~~~~~~

Click the theme toggle in the top-right corner to switch between dark and light mode.

.. image:: /tools/neuron-explorer/images/dark-light-mode-toggle.png
   :target: ../../_images/dark-light-mode-toggle.png

**When to use:** Personal preference, accessibility considerations, or matching your
presentation theme in screenshots.
