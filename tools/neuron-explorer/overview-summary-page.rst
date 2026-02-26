.. meta::
    :description: Learn how to use the Neuron Explorer summary page to quickly identify performance issues, view key metrics, and get actionable optimization recommendations for your profiles.
    :date-modified: 12/02/2025

Summary Viewer
================

The Neuron Explorer summary viewer provides a streamlined view of your profile's most critical performance insights, enabling quick identification of issues and optimization opportunities without navigating through detailed data.

.. image:: /tools/profiler/images/explorer-summary-page.png

Benefits
--------

Both new and experienced users benefit from this streamlined view of profiling data.

* Identify performance issues quickly
* Understand your profile's most critical metrics at a glance
* Get actionable recommendations for optimization
* View all key information on a single screen without scrolling

How to use
-------------

1. **Open your profile** - The Summary Viewer is accessible via the Profile Manager or Neuron Explorer UI.
2. **Examine key metrics** - Review the metrics and graphs to understand your profile's performance characteristics.
3. **Review recommendations** - Start with the **Performance Insights & Recommendations** section. This section highlights the most important performance issues.
4. **Select specific time regions** - Use the "Region Selection" menu to view specific timeslices corresponding to network layers. This helps you drill down into specific sections of your profile. You can generate custom time regions using the "Add Region" button.
5. **Take action** - Apply the recommended optimizations to your model or workload.

Understanding region-level insights
-----------------------------------

When you work with profiles from entire networks or network subgraphs, different regions will have different performance characteristics. The landing page enables performance analysis on a per-layer basis and provides:

* Layer-specific recommendations
* Time-range indication of where problems occur
* More accurate insights for complex profiles

Use the 'Region Selection' menu to navigate between different layers and view their individual performance data.

What the landing page displays
------------------------------

Performance Insights and Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section shows 2-4 recommendations to help you improve performance. The profiler analyzes your data and identifies the most important issues to address. The profiler prioritizes recommendations and shows you the most critical ones first.

Example recommendations
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Condition
     - Root Cause
     - Recommended Action
   * - Tensor Engine Utilization < 70%
     - Small batch sizes, memory-bound operations
     - Increase batch size, optimize data layout
   * - High DMA Wait Time (>15% of total)
     - Data movement blocking compute
     - Prefetch data, optimize memory access patterns
   * - State Buffer Fullness > 90% or sbuf_reload_ratio > 60% or sbuf_save_ratio > 60%
     - Intermediate tensors are spilling
     - Fuse adjacent ops and/or restructure to keep data live in SBUF to cut spills
   * - mm_arithmetic_intensity < peak_flops_bandwidth_ratio
     - The kernel is memory-bound
     - Focus on raising arithmetic intensity (reduce reloads/spills, cut layout overhead, and improve DMA efficiency)
   * - mm_arithmetic_intensity >= peak_flops_bandwidth_ratio
     - The kernel is compute-bound
     - Keep at least one compute engine's active_time_percent near 90%+ and raise mfu_estimated_percent toward 100% for matmul-heavy code
   * - weight_queue_bytes >> weight_size_bytes
     - Weights are being reloaded
     - Restructure to reuse weights from SBUF and reduce duplicate HBM reads
   * - dma_transfer_average_bytes is small and dma_transfer_count is large
     - Transfers are fragmented
     - Increase dma_transfer_average_bytes (target ≥ 32768) to raise mbu_estimated_percent

Key Metrics
~~~~~~~~~~~

This section displays tables and graphs that summarize your profile's performance metrics.

Compute Performance Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **total_time** - Total duration of on-device time for the run in seconds. This doesn't include host-device data movement overhead or host runtime/framework overhead.
* **mm_arithmetic_intensity** - The ratio of regular Matrix Multiplication (MATMUL) Floating Point Operations (FLOPs) to total Dynamic Random Access Memory (DRAM) transfer size. This metric helps you determine if your workload is memory-bound or compute-bound.
* **hfu_estimated_percent** - Hardware FLOPs Utilization reflects the Tensor Engine utilization calculated from all Tensor Engine instructions.
* **mfu_estimated_percent** - Model FLOPs Utilization reflects the Tensor Engine utilization for useful compute (matrix multiplications from your model definition).

Memory Bandwidth Utilization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **total_bandwidth_available** - The total bytes possible to be transferred within the given time region for the current Neuron hardware specification.
* **mbu_estimated_percent** - Memory Bandwidth Utilization (MBU) shows the achieved (as running on the current Neuron hardware) High Bandwidth Memory (HBM) bandwidth utilization.
* **average_dma_size** - The average DMA transfer size (higher is better).
* **useful_read_percent** - The fraction of HBM reads that are useful (``hbm_read_bytes`` - ``hbm_reload_bytes``) / hbm_read_bytes). Note that "useful" is related to an inherent property of the memory itself, but a measurement of how efficiently the memory is being utilized by a specific workload or application. Low numbers may indicate inefficient memory access patterns and suboptimal layouts.

FLOPs Utilization
^^^^^^^^^^^^^^^^^

For each compute engine (tensor, vector, scalar, gpsimd), displays the how well utilized the engine is:

Tensor Engine
"""""""""""""

The Tensor engine has a detailed breakdown of how the FLOPs are being used:

* **model_flops**: The percentage of tensor flops spent performing useful matrix operations, contributing to model progress
* **transpose_flops**: The percentage of tensor flops spent performing transpose operations / data movement
* **active_flops** - Percentage of tensor flops that correspond to the active_time of the tensor engine, but where the engine was not effectively utilized.
* **throttled_flops (active and inactive)** - Percentage of FLOPs wasted due to throttling, either during active or inactive tensor engine periods,

There are a few key things to look for in this graph:

1. **model_flops relative to active_flops**. Large differences could indicate that the tensor engine is being poorly utilized with small tensor sizes, or that operations are not being pipelined effectively.
2. **model_flops relative to transpose_flops**. It is desired to have little-to-no ``transpose_flops`` consuming tensor engine utilization. Ideally the ``model_flops`` amount is much larger than the amount of transposes.
3. **active_throttled_flops**: FLOPs lost due to throttling during active periods is undesirable. It is worth identifying the root cause for the throttling if there is indication of this happening.

Other Engines (Scalar, Vector, GpSimd)
"""""""""""""""""""""""""""""""""""""""

These engines do not yet have detailed FLOP utilization breakdowns, they only show the active period of operation for the engine.

* **active_flops** - Percentage of FLOPs when the engine processes at least one instruction (excluding semaphore waits).

Memory Bandwidth Breakdown
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shows how the available HBM memory bandwidth was used:

* Total HBM read / write operations
* State Buffer (SBUF) spill/reload operations

Collective Operations Duration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Displays the duration of each collective operation in the profile, grouped by operation type and size. This is useful for identifying outliers in Collective runtime, which can be used to investigate specific sections of the profile more deeply. It is possible to filter out datasets by clicking on the datasets in the legend of the graph.
