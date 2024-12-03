.. _aws-trn2-arch:

############################
Amazon EC2 Trn2 architecture
############################

Trn2 is an Amazon EC2 accelerated computing instance, purpose built for high-performance deep learning training and inference. This page provides 
an architecture overview of the trn2-16.48xlarge instance and the u-trn2x64 UltraServer.

.. contents::  Topics
   :local:
   :depth: 1

.. _trn2-arch:

Trn2 instance sizes
===================

Amazon EC2 Trn2 instances are available in the following sizes:

* trn2-16.48xlarge
* u-trn2x64 UltraServer

.. _trn2-16-instance:

trn2-16.48xlarge
""""""""""""""""
Trn2-16 instances are powered by 16 Trainium2 devices connected using a high-bandwidth, low-latency NeuronLink-v3 
device-to-device interconnect. The NeuronLink-v3 device-to-device interconnect enables collective communication between Trainium2 
devices during distributed training and inference. It also allows for the pooling of memory resources from all 16 Trainium2 devices.  

In a trn2-16.48xlarge instance, 16 Trainium2 devices are connected using a 4x4, 2D Torus topology. The following diagram shows the 
intra-instance connections of a Trn2-16.48xlarge instance

.. image:: /images/architecture/Trn2/trn2.48xlarge.png
    :align: center
    :width: 650
|

.. _u-trn2-ultraserver: 

u-trn2x64 UltraServer
"""""""""""""""""""""

A u-trn2x64 UltraServer comprises four trn2-16 instances connected together via the NeuronLink-v3 device-to-device interconnect. 
This allows for a total of 64 Trainium2 devices to be interconnected within a u-trn2x64 UltraServer. Trainium2 devices with the same 
coordinates in each Trn2-16 instance are connected in a ring topology. The following figure shows the inter-instance ring connection 
between Trainium2 devices.

.. image:: /images/architecture/Trn2/u-trn2x64.png
    :align: center
    :width: 650
|
Trn2 instance specifications 
============================

The following table shows the performance metrics for Trainium2 based instances.

.. list-table::
    :widths: auto
    :header-rows: 1
    :stub-columns: 1    
    :align: left
      

    *   - Perfomance specification
        - trn2-16.48xlarge
        - u-trn2x64 UltraServer
    *   - # of Trainium2 devices
        - 16
        - 64
    *   - vCPUs
        - 192
        - 768
    *   - Host Memory (GiB)
        - 2,048
        - 8,192
    *   - FP8 PFLOPS
        - 20
        - 80
    *   - FP16/BF16/TF32 PFLOPS
        - 10
        - 40
    *   - FP8/FP16/BF16/TF32 Sparse PFLOPS
        - 40
        - 160
    *   - FP32 PFLOPS
        - 2.9
        - 11.6
    *   - Device Memory (GiB)
        - 1,536
        - 6,144
    *   - Device Memory Bandwidth (TB/sec)
        - 46.4
        - 185.6
    *   - Intra-instance NeuronLink-v3 bandwidth (GB/sec/device)
        - 1,024
        - 1,024
    *   - Inter-instance NeuronLink-v3 bandwidth (GB/sec/device)
        - Not applicable
        - 256
    *   - EFA bandwidth (Gbps)
        - 3,200
        - 3,200

