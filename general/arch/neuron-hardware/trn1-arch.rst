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

The Trn1 instances are also available in an EC2 UltraCluster, which
enables customers to scale Trn1 instances to over 30,000 Trainium
devices, and leverage the AWS-designed non-blocking petabit-scale EFA
networking infrastructure.

.. image:: /images/trn1-ultra-cluster.png



