.. _aws-trn1-arch:

AWS Trn1 Architecture
=====================

In this page, we provide an architectural overview of the AWS Trn1
instances, and the corresponding :ref:`Trainium <trainium-arch>` NeuronDevices that power them
(Trainium devices from here on).

.. contents::  Table of contents
   :local:
   :depth: 2

.. _trn1-arch:

Trn1 Architecture
-----------------

The EC2 Trn1 instance is powered by up to 16 :ref:`Trainium <trainium-arch>` devices, and allows
customers to choose between 2 instance sizes:


.. list-table::
    :widths: auto
    :header-rows: 1
    :stub-columns: 1    
    :align: left
      

    *   - Instance size
        - # of Trainium devices
        - vCPUs
        - Host Memory (GiB)
        - FP8/FP16/BF16/TF32 TFLOPS
        - FP32 TFLOPS
        - Device Memory (GiB)
        - Device Memory Bandwidth (GiB/sec)
        - NeuronLink-v2 device-to-device 
        - EFA bandwidth (Gbps)

    *   - Trn1.2xlarge
        - 1
        - 8
        - 32
        - 210
        - 52.5
        - 32
        - 820
        - N/A
        - up-to 25 

    *   - Trn1.32xlarge
        - 16
        - 128
        - 512
        - 3,360
        - 840
        - 512
        - 13,120
        - Yes
        - 800


The Trn1.2xlarge instance size allows customers to train their models on
a single Trainium device, which is useful for small model training, as
well as model experimentation. The Trn1.32xlarge instance size comes
with a high-bandwidth and low-latency NeuronLink-v2 device-to-device
interconnect, which utilizes a 4D-HyperCube topology. This is useful for
collective-communication between the Trainium devices during scale-out
training, as well as for pooling the memory capacity of all Trainium
devices, making it directly addressable from each one of the devices.

In a Trn1 server, the Trainium devices are connected in a 2D Torus topology, as depicted below:

.. image:: /images/trn1-topology.png
    :width: 400

The Trn1 instances are also available in an EC2 UltraCluster, which
enables customers to scale Trn1 instances to over 30,000 Trainium
devices, and leverage the AWS-designed non-blocking petabit-scale EFA
networking infrastructure.

.. image:: /images/trn1-ultra-cluster.png
    :width: 600



.. _trainium-arch:


Trainium Architecture
----------------------

At the heart of the Trn1 instance are Trainium devices, the second
generation purpose-built Machine Learning accelerator from AWS. The
Trainium device architecture is depicted below:

.. image:: /images/trainium-neurondevice.png
    :width: 500

Each Trainium device consists of:

-  Compute:
    * 2x :ref:`NeuronCore-v2 <neuroncores-v2-arch>` cores, delivering 420 INT8 TOPS, 210 FP16/BF16/cFP8/TF32 TFLOPS, and
      52.5 FP32 TFLOPS.

-  Device Memory:
    * 32GB of device memory (for storing model state), with 820 GB/sec of bandwidth.


-  Data movement:
    * 1 TB/sec of DMA bandwidth, with inline memory compression/decompression.

-  NeuronLink:
    * NeuronLink-v2 for device-to-device interconnect enables efficient scale-out training, as well as memory pooling between the different Trainium
      devices.

-  Programmability:
    * Trainium supports dynamic shapes and control flow, via ISA extensions of NeuronCore-v2. In addition, 
      Trainium also allows for user-programmable :ref:`rounding mode <neuron-rounding-modes>` (Round Nearest Even 
      Stochastic Rounding), and custom-operators via the deeply embedded GPSIMD Engine.


