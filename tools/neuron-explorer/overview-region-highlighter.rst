.. meta::
    :description: Learn how Region Highlighter in Neuron Explorer automatically identifies regions in your profiles based on collective operations, hierarchy data, and kernel stack frames.
    :date-modified: 04/26/2026

.. _region-highlighter-overview:

Region Highlighter
==================

Region Highlighter automatically identifies and marks regions in your profiles based on collective operations, operation hierarchy, and kernel stack frames. 
Rather than manually scanning the timeline and placing annotations, the Region Highlighter analyzes the profile structure, in combination with a user-provided
configuration, and generates bounded time regions that highlight areas of interest for performance analysis. 
These regions integrate directly with existing Annotation tool, and the :doc:`Summary Viewer <overview-summary-page>` for comparative, per-region performance insights.

Region Highlighter is enabled by default and requires no additional configuration or external services.

.. image:: /tools/neuron-explorer/images/region-highlighter-configs.png

Applies to
----------

Region Highlighter is applicable to:

- Device profiles with collective operations, hierarchy data, or kernel stack frame information
- System profiles containing device-level profiling data

How it works
------------

Region Highlighter uses three complementary analysis methods to identify meaningful regions in a profile:

**Collective operations boundaries**
    Uses Collective Operation boundaries in the profile timeline, such as all-reduce or all-gather to generate regions in between collectives for quick comparison.

**Hierarchy data analysis**
    Analyzes the operation hierarchy structure to detect logical groupings of operations. This produces regions aligned to model layers and operation groups, giving you a structured view of where time is spent across different parts of your model.

**Kernel stack frame analysis**
    Generates regions based on kernel execution boundaries derived from stack frame information. These regions align with individual kernel invocations, making it straightforward to compare the performance characteristics of different kernels.

Each method contributes a set of annotation regions, the resulting regions are combined to give a comprehensive view of the profile's structure.
Once generated, the explorer provides a quick overview summary of the performance of each of the regions.

.. image:: /tools/neuron-explorer/images/region-highlighter-fast-region-perf.png

**NOTE**: Only one method of generating regions can be used at a time.

What are annotation regions?
----------------------------

An annotation region is a time slice in the profile demarcated by a start annotation and an end annotation. Unlike individual annotation markers, regions represent bounded time intervals suitable for comparative performance analysis.

Region Highlighter generates these regions automatically. You can view, rename, save, and load them through the Annotations tab in the Device Trace Viewer, just like manually created annotations. Each region captures a specific segment of the profile timeline so you can compare metrics across different parts of your workload.

.. image:: /tools/neuron-explorer/images/region-highlighter-regions.png

Detailed Region Comparison
--------------------------

The Region Highlighter tool also provides the ability to compare up to 3 regions side-by-side using the same views on the SummaryPage.
To access this view, click the data points in the `Annotation Region Duration Distribution` chart shown above.

.. image:: /tools/neuron-explorer/images/region-highlighter-detailed-region-perf.png

Using Region Highlighter with the Summary Viewer
------------------------------------------------

Region Highlighter regions automatically populate the Summary Viewer's region selection menu. This allows you to:

1. **Compare per-region metrics** — Select individual regions to view their performance statistics, including compute utilization, memory bandwidth, and DMA activity.
2. **Identify outliers** — Quickly spot regions with anomalous performance characteristics by switching between regions and comparing key metrics.
3. **Drill into specific layers** — Use hierarchy-based regions to analyze performance at the model layer level without manually placing annotations.

Select a region from the "Region Selection" menu in the Summary Viewer to view its performance data. See :doc:`overview-summary-page` for details on interpreting region-level insights.

.. _neuron-explorer-nki-regions:

NKI Regions
------------

NKI kernel developers can mark sections of their kernel source as NKI Regions, using ``nki.language.profiler_region_open`` and ``nki.language.profiler_region_close``. Those regions appear in the Region Highlighter panel alongside the automatically generated ones, so you can compare the performance of different phases of a kernel. For how to define them, see :ref:`Define NKI Regions in Your Kernel <nki-define-regions>`.

Each region spans the execution time of all hardware instructions compiled from the marked code, even where the compiler reordered them. A region opened inside another appears on the instructions of both, so nested regions overlap on the timeline.

NKI Regions require ``NKI_DEBUG_INFO=True`` at compile time. Without it the compiler emits no region metadata and the panel shows none. See :doc:`environment-variables`.

.. image:: /tools/neuron-explorer/images/region-highlighter-nki-regions.png

Filtering regions by invocation and iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A region label that is opened from more than one call to the kernel, or opened inside a loop, covers more than one span of time. The NKI Regions table gives each label a filter box so you can narrow it to the occurrences you care about.

**Invocation**
    Which call to the kernel the region belongs to. This column appears only when the label was reached from more than one call.

**Iteration**
    Which loop pass within that call. This column appears when the region was opened inside a loop.

Each box accepts the same syntax, also available from the info icon in the column header:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Input
     - Matches
   * - *(empty)*
     - All occurrences.
   * - ``0``
     - A single value.
   * - ``0,2,5``
     - A list of values.
   * - ``1-4``
     - An inclusive range.
   * - ``0,3-5``
     - A list and a range combined.

Data requirements
-----------------

Region Highlighter generates regions based on the data available in each profile. If a profile is missing a particular data source, the corresponding analysis method is skipped:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Data source
     - Annotation type produced
   * - Collective operations metadata
     - Regions around collective communication operations
   * - Operation hierarchy
     - Regions aligned to model layers and operation groups
   * - Kernel stack frames
     - Regions aligned to kernel execution boundaries

If none of these data sources are present, Region Highlighter will not generate any regions.

Further reading
---------------

- :doc:`overview-summary-page` — Interpret the per-region performance insights that Region Highlighter feeds into
- :doc:`overview-device-profiles` — Learn about the Device Trace Viewer where annotation regions are displayed
- :doc:`overview-ai-recommendations` — Get AI-powered optimization recommendations for NKI profiles
