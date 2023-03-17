.. _inferentia2-arch:

Inferentia2 Architecture
------------------------

At the heart of the Inf2 instance are up to 12 Inferentia2 devices (each Inferentia2 include 2 :ref:`NeuronCore-v2 <neuroncores-v2-arch>`). Inferentia2 is the second
generation purpose built Machine Learning inference accelerator from
AWS. The Inferentia2 device architecture is depicted below: 

.. image:: /images/inferentia2.jpg


Each Inferentia2 device consists of:


-  Compute:
    * 2x :ref:`NeuronCore-v2 <neuroncores-v2-arch>` cores, delivering 380 INT8 TOPS, 190 FP16/BF16/cFP8/TF32 TFLOPS, and 47.5 FP32 TFLOPS.

-  Device Memory:
    * 32GB of HBM2E of device memory (for storing model state), with 820 GB/sec of bandwidth.


-  Data movement:
    * 1 TB/sec of DMA bandwidth, with inline memory compression/decompression.

-  NeuronLink:
    * NeuronLink-v2 for device-to-device interconnect enables high performance collective compute for co-optimization of latency and throughput.

-  Programmability:
    * Inferentia2 supports dynamic shapes and control flow, via ISA extensions of NeuronCore-v2 and custom-operators via the deeply embedded GPSIMD engines.

