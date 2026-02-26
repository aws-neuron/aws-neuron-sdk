.. meta::
    :description: Learn about the Tensor Viewer in Neuron Explorer for viewing tensor information including names, sizes, shapes, and memory usage details.
    :date-modified: 01/27/2026

.. _tensor-viewer-overview:

Tensor Viewer
=================

The Tensor Viewer contains the following information about all tensors in the NEFF file:

* **variable_name** - The tensor name.
* **type** - How the system uses the tensor. Examples include input tensor, output tensor, or weight tensor.
* **format** - How the tensor arranges in memory. For example, "NHWC" shows a specific dimension arrangement. Letters include N (batch size), H (height), W (width), C (channel).
* **shape** - The tensor's multi-dimensional shape.
* **size** - The tensor's total size in bytes.
* **node** - NEFF node.
* **pcore_idx** - Index of the physical NeuronCore within a Logical NeuronCore (LNC). A Logical NeuronCore groups physical NeuronCores. For LNC2, this field shows either 0 or 1.
* **load_to_sbuf_avg_size_bytes** - The average size in bytes of each DMA transfer when the system loads this tensor into the State Buffer.
* **load_to_sbuf_total_size_bytes** - The total size in bytes of all DMA transfers when the system loads this tensor into the State Buffer.
* **load_to_sbuf_dma_count** - The total number of DMAs that loaded this tensor into the State Buffer.
* **load_to_sbuf_repeat_factor** - How many times the system loaded this tensor into the State Buffer. A value of 1 means one load, 2 means two loads, and so on.

.. image:: /tools/profiler/images/tensor-viewer-table.png

You can use this data to match with framework-level instructions or for kernel development. You can also use it to search for instructions in the Device Timeline Viewer. 
The SBUF loading information in the table can help you verify tensors are loaded efficiently.

Searching
---------

You can use the Tensor Viewer with the Device Timeline Viewer and Search tool to match tensor information in the table with instructions that run on the device. 
Enter the variable_name from the table, into the DMA search field to see all DMA instructions that relate to that tensor.
The example below shows a complete search for the tensor token_position_to_id:

.. image:: /tools/profiler/images/tensor-viewer-search-example.png
