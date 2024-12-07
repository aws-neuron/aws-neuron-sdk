.. _inferentia2-arch:

Inferentia2 Architecture
------------------------

At the heart of each Inf2 instance are up to twelve Inferentia2 chips (each with two :ref:`NeuronCore-v2 <neuroncores-v2-arch>` cores). Inferentia2 is the second
generation AWS purpose-built Machine Learning inference accelerator. The Inferentia2 chip architecture is depicted below: 

.. image:: /images/inferentia2.png


Each Inferentia2 chip consists of:

+----------------------------------+----------------------------------+
| Compute                          | Two :ref:`NeuronCore-v2          |
|                                  | <neuroncores-v2-arch>`           |
|                                  | cores, delivering 380 INT8 TOPS, |
|                                  | 190 FP16/BF16/cFP8/TF32 TFLOPS,  |
|                                  | and 47.5 FP32 TFLOPS.            |
+----------------------------------+----------------------------------+
| Chip Memory                      | 32GiB of high-bandwidth chip     |                                  
|                                  | memor (HBM) (for storing model   |                                  
|                                  | state), with 820 GiB/sec of      |                                  
|                                  | bandwidth.                       |
+----------------------------------+----------------------------------+
| Data Movement                    | 1 TB/sec of DMA bandwidth, with  |
|                                  | inline memory                    |
|                                  | compression/decompression.       |
+----------------------------------+----------------------------------+
| NeuronLink                       | NeuronLink-v2 for                |                                  
|                                  | chip-to-chip interconnect        |                                  
|                                  | enables high-performance         |                                  
|                                  | collective compute for           |                                  
|                                  | co-optimization of latency and   |                                  
|                                  | throughput.                      |
+----------------------------------+----------------------------------+
| Programmability                  | Inferentia2 supports dynamic     |
|                                  | shapes and control flow, via ISA |
|                                  | extensions of NeuronCore-v2 and  |
|                                  | :ref:`custom-operators           |
|                                  | <feature-custom-c++-operators>`  |
|                                  | via the deeply embedded GPSIMD   |
|                                  | engines.                         |
+----------------------------------+----------------------------------+

For a more detailed description of all the hardware engines, see :ref:`NeuronCore-v2 <neuroncores-v2-arch>`.