.. _aws-trn2-arch:

############################
Amazon EC2 Trn2 Architecture
############################

Trn2 is an Amazon EC2 accelerated computing instance, purpose built for high-performance deep learning training and inference. This page provides 
an architecture overview of the trn2.48xlarge and trn2u.48xlarge instances, and Trn2 UltraServer.

.. contents::  Topics
   :local:
   :depth: 2

.. _trn2-arch:

Trn2 instance sizes
===================

Trn2 instances and UltraServers are available in the following sizes and configurations:

* trn2.48xlarge
* trn2u.48xlarge
* Trn2 UltraServer

.. _trn2-instance:

trn2.48xlarge / trn2u.48xlarge
""""""""""""""""""""""""""""""
Trn2 instances are powered by 16 Trainium2 chips connected using a high-bandwidth, low-latency NeuronLink-v3 
chip-to-chip interconnect. The NeuronLink-v3 chip-to-chip interconnect enables collective communication between Trainium2 
chips during distributed training and inference. It also allows for the pooling of memory resources from all 16 Trainium2 chips.  

In a trn2.48xlarge or trn2u.48xlarge instance, 16 Trainium2 chips are connected using a 4x4, 2D Torus topology. The following diagram shows the 
intra-instance connections of a trn2.48xlarge or trn2u.48xlarge instance

.. image:: /images/architecture/Trn2/trn2.48xlarge.png
    :align: center
    :width: 650
|

.. _trn2-ultraserver: 

Trn2 UltraServer
"""""""""""""""""""""

A Trn2 UltraServer comprises four trn2u.48xlarge instances connected together via the NeuronLink-v3 chip-to-chip interconnect. 
This allows for a total of 64 Trainium2 chips to be interconnected within a Trn2 UltraServer. Trainium2 chips with the same 
coordinates in each Trn2 instance are connected in a ring topology. The following figure shows the inter-instance ring connection 
between Trainium2 chips.

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
        - trn2.48xlarge / trn2u.48xlarge
        - Trn2 UltraServer
    *   - # of Trainium2 chips
        - 16
        - 64
    *   - vCPUs
        - 192
        - 768
    *   - Host Memory (GiB)
        - 2,048
        - 8,192
    *   - FP8 PFLOPS
        - 20.8
        - 83.2
    *   - FP16/BF16/TF32 PFLOPS
        - 10.7
        - 42.8
    *   - FP8/FP16/BF16/TF32 Sparse PFLOPS
        - 41
        - 164
    *   - FP32 PFLOPS
        - 2.9
        - 11.6
    *   - Chip Memory (GiB)
        - 1,536
        - 6,144
    *   - Chip Memory Bandwidth (TB/sec)
        - 46.4
        - 185.6
    *   - Intra-instance NeuronLink-v3 bandwidth (GB/sec/chip)
        - 1,024
        - 1,024
    *   - Inter-instance NeuronLink-v3 bandwidth (GB/sec/chip)
        - Not applicable
        - 256
    *   - EFAv3 bandwidth (Gbps)
        - 3,200
        - 3,200

