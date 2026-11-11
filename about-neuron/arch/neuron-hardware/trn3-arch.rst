.. meta::
   :description: Architecture of Amazon EC2 Trn3 UltraServers powered by AWS Trainium3 chips, including Gen1 and Gen2 NeuronLink-v4 and NeuronSwitch-v1 connectivity, per-chip bandwidth, and specifications for AWS Neuron.
   :keywords: Trn3, Trainium3, UltraServer, NeuronLink-v4, NeuronSwitch-v1, AWS Neuron, architecture
   :date-modified: 2026-11-04

.. _aws-trn3-arch:

###############################
Amazon EC2 Trn3 Architecture
###############################

Amazon EC2 **Trn3** instances are powered by Trainium3 AI chips, purpose-built for high-performance deep learning training and inference. Trn3 is available in two UltraServer scale-up configurations: Gen1 with 64 Trainium3 chips per UltraServer, and Gen2 with 144 chips per UltraServer.

Both configurations use NeuronSwitch-v1 interconnect technology to connect chips all-to-all, replacing the 2D-torus topology of previous generations. All chips in an UltraServer form a single unified memory space, and each chip occupies its own range in this space. To access the memory of another chip in the same UltraServer, a DMA operation uses an address in that chip's range. The switches automatically use the address to forward each data transfer to its destination chip.

Together, the switched topology and the unified memory space make Trn3 UltraServers well suited to workloads that need efficient all-to-all communication or ultra-low latency collectives, such as Mixture of Experts models and autoregressive inference serving.

=====================
Trn3 Gen1 UltraServer
=====================

The EC2 Trn3 Gen1 UltraServers deliver 161 PetaFLOPS of dense MXFP8 compute, 314 TB/s of HBM bandwidth, and 9TB of HBM capacity. Each UltraServer consists of four servers with 16 Trainium3 chips per server. Therefore, the UltraServer integrates a total of 64 Trainium3 chips into a single scale-up domain, interconnected via our latest-generation NeuronLink-v4 and the newly introduced NeuronSwitch-v1. The following diagram illustrates the Trn3 Gen1 UltraServer connectivity.

.. image:: /images/architecture/trn3/trn3-ultraserver-gen1.svg
    :align: center
    :alt: Trn3 Gen1 UltraServer connectivity: four Trn3 servers with 16 Trainium3 chips each, paired chips linked within each server, two inter-server NeuronSwitch-v1 switches, and peer-to-peer links between Server 0 and Server 2 and between Server 1 and Server 3.

Connectivity
------------

All chip-to-chip communication in the Gen1 UltraServer uses NeuronLink-v4, either through NeuronSwitch-v1 switches or over direct chip-to-chip links.

Intra-server connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~

Each server contains 16 Trainium3 chips, grouped in pairs: Trainium3 [0] and [1], [2] and [3], and so on up to [14] and [15]. The two chips of a pair are connected by a direct NeuronLink-v4 link. Each chip has 1,024 GB/s [#nl-bw]_ of bandwidth to its paired chip.

Inter-server connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~

The Trn3 Gen1 UltraServer has two racks, each with two servers and one inter-server NeuronSwitch-v1 (radix-32):

* Server [0] and Server [1] share one rack. All 32 Trainium3 chips of these two servers connect to the rack's switch.
* Server [2] and Server [3] share the other rack. All 32 Trainium3 chips of these two servers connect to the rack's switch.

Each chip has 1,024 GB/s of NeuronLink-v4 bandwidth to its inter-server switch.

Peer-to-peer connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~

Each Trainium3 chip has a direct peer-to-peer NeuronLink-v4 link to the chip in the same position in the other rack: Server [0] with Server [2], and Server [1] with Server [3]. For example, Trainium3 [0] of Server [0] connects to Trainium3 [0] of Server [2]. Each chip has 512 GB/s of bandwidth on this link.

Bandwidth summary
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Connectivity level
     - Bandwidth per chip
     - Link configuration
   * - Intra-server
     - 1,024 GB/s
     - NeuronLink-v4 direct link to the paired chip
   * - Inter-server (within rack)
     - 1,024 GB/s
     - NeuronLink-v4 via inter-server NeuronSwitch-v1 (radix-32)
   * - Peer-to-peer (across racks)
     - 512 GB/s
     - NeuronLink-v4 direct link
   * - **Total**
     - **2,560 GB/s**
     -

=====================
Trn3 Gen2 UltraServer
=====================

The EC2 Trn3 Gen2 UltraServers deliver 362 PetaFLOPS of dense MXFP8 compute, 706 TB/s of HBM bandwidth, and 20TB of HBM capacity. Each UltraServer consists of 36 servers with 4 Trainium3 chips per server. Therefore, the UltraServer integrates 144 Trainium3 chips into a single scale-up domain. The following diagram illustrates the Trn3 Gen2 UltraServer connectivity.

.. image:: /images/architecture/trn3/trn3-ultraserver-gen2.svg
    :align: center
    :alt: Trn3 Gen2 UltraServer connectivity: two racks of 18 Trn3 servers, each server with four Trainium3 chips on an intra-server NeuronSwitch-v1, two inter-server NeuronSwitch-v1 switches per rack, and peer-to-peer links between chips in the same position in the two racks.

Connectivity
------------

All chip-to-chip communication in the Gen2 UltraServer uses NeuronLink-v4, either through NeuronSwitch-v1 switches or over direct peer-to-peer links.

Intra-server connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~

Each server contains 4 Trainium3 chips connected through an intra-server NeuronSwitch-v1 (radix-4). Each chip has 1,024 GB/s of NeuronLink-v4 bandwidth to this switch.

Inter-server connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~

Each rack holds 18 servers and two inter-server NeuronSwitch-v1 (radix-36):

* Trainium3 [0] and [1] of every server in the rack connect to NeuronSwitch-v1 [0].
* Trainium3 [2] and [3] of every server in the rack connect to NeuronSwitch-v1 [1].

Each chip has 1,280 GB/s of NeuronLink-v4 bandwidth to its inter-server switch.

Peer-to-peer connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~

The Trn3 Gen2 UltraServer has two racks. Each Trainium3 chip has a direct peer-to-peer NeuronLink-v4 link to the chip in the same position in the other rack, for example Trainium3 [0] of Server [0] and Trainium3 [0] of Server [18]. Each chip has 256 GB/s of bandwidth on this link. The direct link avoids switch hops for cross-rack traffic.

Bandwidth summary
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Connectivity level
     - Bandwidth per chip
     - Link configuration
   * - Intra-server
     - 1,024 GB/s
     - NeuronLink-v4 via intra-server NeuronSwitch-v1 (radix-4)
   * - Inter-server (within rack)
     - 1,280 GB/s
     - NeuronLink-v4 via inter-server NeuronSwitch-v1 (radix-36)
   * - Peer-to-peer (across racks)
     - 256 GB/s
     - NeuronLink-v4 direct link
   * - **Total**
     - **2,560 GB/s**
     -

==========================================
Trn3 Gen1/Gen2 UltraServer specifications
==========================================

The following table shows the performance metrics for Trainium3-based instances.

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
   * - # of Trainium3 chips
     - 64
     - 144
   * - Host vCPUs
     - 768
     - 3,456
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
   * - NeuronLink-v4 bandwidth (GB/sec/chip)
     - 2,560
     - 2,560
   * - EFA bandwidth (Gbps)
     - 12,800
     - 57,600

.. rubric:: Footnotes

.. [#nl-bw] All NeuronLink-v4 bandwidth numbers on this page are the sum of both directions.
