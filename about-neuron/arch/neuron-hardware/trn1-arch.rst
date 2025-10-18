.. _aws-trn1-arch:

Amazon EC2 Trn1/Trn1n Architecture
===================================

On this page, we provide an architectural overview of the AWS Trn1/Trn1n
instances, and the corresponding :ref:`Trainium <trainium-arch>` NeuronChips that power them
(Trainium chips from here on).

.. contents::  Table of contents
   :local:
   :depth: 2

.. _trn1-arch:

Trn1/Trn1n Architecture
-----------------------

An EC2 Trn1/Trn1n instance is powered by up to 16 :ref:`Trainium <trainium-arch>` chips.


.. list-table::
    :widths: auto
    :header-rows: 1
    :stub-columns: 1    
    :align: left
      

    *   - Instance size
        - # of Trainium chips
        - vCPUs
        - Host Memory (GiB)
        - FP8/FP16/BF16/TF32 TFLOPS
        - FP32 TFLOPS
        - Device Memory (GiB)
        - Device Memory Bandwidth (GiB/sec)
        - EFA bandwidth (Gbps)

    *   - Trn1.2xlarge
        - 1
        - 8
        - 32
        - 190
        - 47.5
        - 32
        - 820
        - N/A
        - up-to 25 

    *   - Trn1.32xlarge
        - 16
        - 128
        - 512
        - 3,040
        - 760
        - 512
        - 13,120
        - 384
        - 800

    *   - Trn1n.32xlarge
        - 16
        - 128
        - 512
        - 3,040
        - 760
        - 512
        - 13,120
        - 768
        - 1,600


The Trn1.2xlarge instance size allows customers to train their models on
a single Trainium chip, which is useful for small model training, as
well as for model experimentation. The Trn1.32xlarge and Trn1n.32xlarge instance size come
with a high-bandwidth and low-latency NeuronLink-v2 chip-to-chip
interconnect, which utilizes a 2D Torus topology. This is useful for
collective communication between the Trainium chips during scale-out
training, as well as for pooling the memory capacity of all Trainium
chips, making it directly addressable from each of the chips.

In a Trn1/Trn1n server, the Trainium chips are connected in a 2D Torus topology, as depicted below:

.. image:: /images/trn1-topology.png

The Trn1/Trn1n instances are also available in an EC2 UltraCluster, which
enables customers to scale Trn1/Trn1n instances to over 100,000 Trainium
chips, and leverage the AWS-designed non-blocking petabit-scale EFA
networking infrastructure.

.. image:: /images/ultracluster-1.png



