.. _aws-trn3-arch:

###############################
Amazon EC2 Trn3 Architecture
###############################

Amazon EC2 **Trn3** instances are accelerated computing instances powered by Trainium3 AI chips, purpose-built for high-performance deep learning training and inference. Trn3 is available in two UltraServer scale-up configurations: Gen1 with 64 Trainium3 chips per UltraServer, and Gen2 with 144 chips per UltraServer. Both configurations use NeuronSwitch-v1 interconnect technology to enable all-to-all connectivity between chips, especially optimized for workloads that leverage all-to-all communication patterns, such as Mixture of Experts models and autoregressive inference serving.

=====================
Trn3 Gen2 UltraServer
=====================

The EC2 Trn3 Gen2 UltraServers deliver 362 PetaFLOPS of dense MXFP8 compute, 706 TB/s of HBM bandwidth, and 20TB of HBM capacity. Each UltraServer consists of 36 servers with 4 Trainium3 devices per server. Trainium3 devices within the same server are connected via a first-level NeuronSwitch-v1, while devices across servers are connected via two second-level NeuronSwitch-v1 and NeuronLink-v4. Therefore, the UltraServer integrates 144 Trainium3 devices into a single scale-up domain. Like Gen1, the chip-to-chip topology features an all-to-all connectivity design optimized for Mixture of Experts models and autoregressive inference serving. The following diagram illustrates the Trn3 Gen2 UltraServer connectivity.

.. image:: /images/architecture/trn3/trn3-ultraserver-gen2.png
    :align: center

==========================================
Trn3 Gen1/Gen2 UltraServer specifications
==========================================

The following table shows the performance metrics for Tranium3 based instances.

.. list-table::
   :header-rows: 2
   :stub-columns: 1
   :widths: 30 20 20

   * - 
     - Trn3 Gen1 UltraServer
     - Trn3 Gen2 UltraServer
   * - Configuration
     - 
     - 
   * - # of Trainium3 devices
     - 64
     - 144
   * - Host vCPUs
     - 768
     - 2304
   * - Host Memory (GiB)
     - 8,192
     - 27,648
   * - **Compute**
     - 
     - 
   * - MXFP8/MXFP4 TFLOPS
     - 161,088
     - 362,448
   * - FP16/BF16/TF32 TFLOPS
     - 42,944
     - 96,624
   * - FP32 TFLOPS
     - 11,712
     - 26,352
   * - **Memory**
     - 
     - 
   * - Device Memory (GiB)
     - 9,216
     - 20,736
   * - Device Memory Bandwidth (TB/sec)
     - 313.6
     - 705.6
   * - **Interconnect**
     - 
     - 
   * - NeuronLink-v4 bandwidth (GiB/sec/device)
     - 2,048
     - 2,048
   * - EFA bandwidth (Gbps)
     - 12,800
     - 28,800
  