.. _trainium2-arch:

######################
Trainium2 Architecture
######################

Trainium2 is the third generation, purpose-built Machine Learning chip from AWS. It powers Amazon EC2 trn2-16.48xlarge instances and 
the u-trn2x64 UltraServer. Every Trainium2 device contains eight NeuronCore-V3. Beginning with Trainium2, AWS Neuron adds support for Logical 
NeuronCore Configuration (LNC), which lets you combine the compute and memory resources of multiple physical NeuronCores into a 
single logical NeuronCore. The following diagram shows the architecture overview of a Trainium2 device.

.. image:: /images/architecture/Trainium2/trainium2.png
    :align: center
    :width: 400
===========================
Trainium2 device components
===========================

Each Trainium2 device consists of the following components:

+----------------------------------+-----------------------------------------------------+
| Compute                          | Eight NeuronCore-v3 that collectively deliver:      |
|                                  |                                                     |
|                                  | * 1,287 FP8 TFLOPS                                  | 
|                                  | * 655 BF16/FP16/TF32 TFLOPS                         |
|                                  | * 2,551 FP8/FP16/BF16/TF32 sparse TFLOPS            |
|                                  | * 181 FP32 TFLOPS                                   |
|                                  |                                                     |
+----------------------------------+-----------------------------------------------------+
| Device Memory                    | 96 GiB of device memory with 2.9 TB/sec of          |
|                                  | bandwidth.                                          |             
+----------------------------------+-----------------------------------------------------+
| Data Movement                    | 3.5 TB/sec of DMA bandwidth, with inline            |
|                                  | memory compression and decompression.               |
+----------------------------------+-----------------------------------------------------+
| NeuronLink                       | NeuronLink-v3 for device-to-device interconnect     |
|                                  | provides 1.28 TB/sec bandwidth per device. It allows|
|                                  | for efficient scale-out training and inference, as  |
|                                  | well as memory pooling between Trainium2 devices.   |
+----------------------------------+-----------------------------------------------------+
| Programmability                  | Trainium2 supports dynamic shapes and control flow  |
|                                  | via NeuronCore-v3 ISA extensions. Trainium2 also    |
|                                  | allows for user-programmable                        |
|                                  | :ref:`rounding mode <neuron-rounding-modes>`        |
|                                  | (Round Nearest Even or Stochastic Rounding), and    |
|                                  | custom operators via deeply embedded GPSIMD engines.|
+----------------------------------+-----------------------------------------------------+
| Collective communication         | 20 CC-Cores orchestrate collective communication    |
|                                  | among Trainium2 devices within and across instances.|
+----------------------------------+-----------------------------------------------------+     

==================================
Trainium2 performance improvements
==================================

The following set of tables offer a comparison between Trainium and Trainium2 devices. 
 
Compute
"""""""

.. list-table::
    :widths: auto
    :header-rows: 1 
    :stub-columns: 1    
    :align: left
      
    *   - 
        - Trainium
        - Trainium2
        - Improvement factor
    
    *   - FP8 (TFLOPS)
        - 191
        - 1287
        - 6.7x
    *   - BF16/FP16/TF32 (TFLOPS)
        - 191
        - 655
        - 3.4x
    *   - FP32 (TFLOPS)
        - 48
        - 181
        - 3.7x
    *   - FP8/FP16/BF16/TF32 Sparse (TFLOPS)
        - Not applicable
        - 2551 
        - Not applicable

Memory
""""""

.. list-table::
    :widths: auto
    :header-rows: 1 
    :stub-columns: 1    
    :align: left
      
    *   - 
        - Trainium
        - Trainium2
        - Improvement factor
    
    *   - HBM Capacity (GiB)
        - 32
        - 96
        - 3x
    *   - HBM Bandwidth (TB/sec)
        - 0.8
        - 2.9
        - 3.6x
    *   - SBUF Capacity (MiB)
        - 48
        - 224
        - 4.7x
    *   - Memory Pool Size
        - Up to 16 devices
        - Up to 64 devices
        - 4x

Interconnect
""""""""""""

.. list-table::
    :widths: auto
    :header-rows: 1 
    :stub-columns: 1    
    :align: left
      
    *   - 
        - Trainium
        - Trainium2
        - Improvement factor
    
    *   - Inter-chip Interconnect (GB/sec/device)
        - 384
        - 1280
        - 3.3x

Data movement
"""""""""""""
.. list-table::
    :widths: auto
    :header-rows: 1 
    :stub-columns: 1    
    :align: left
      
    *   - 
        - Trainium
        - Trainium2
        - Improvement factor
    
    *   - CC Cores
        - 6
        - 20
        - 3.3x
    *   - DMA barriers
        - Write-after-write
        - Strong-order-write
        - \>1x (Benefit DMA-size dependent)
    *   - SBUF memory layout
        - Row-major
        - Row-major, Col-major-2B, Col-major-4B
        - Not applicable

====================
Additional resources
====================

For a detailed description of NeuronCore-v3 hardware engines, instances powered by AWS Trainium2, and Logical NeuronCore configuration, see the following resources:

* :ref:`NeuronCore-v3 architecture <neuroncores-v3-arch>`
* :ref:`Amazon EC2 Trn2 architecture <aws-trn2-arch>`
* :ref:`Logical NeuronCore configuration <logical-neuroncore-config>`
