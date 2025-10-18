.. _aws-inf1-arch:

Amazon EC2 Inf1 Architecture
==============================

On this page, we provide an architectural overview of the Amazon EC2 Inf1
instance and the corresponding :ref:`Inferentia <inferentia-arch>` NeuronChips that power
them (:ref:`Inferentia <inferentia-arch>` chips from here on).

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _inf1-arch:

Inf1 Architecture
-----------------

The EC2 Inf1 instance is powered by 16 :ref:`Inferentia <inferentia-arch>` chips, allowing
customers to choose between four instance sizes:

.. list-table::
    :widths: auto
    :header-rows: 1
    :stub-columns: 1    
    :align: left
    

    *   - Instance size
        - # of Inferentia chips
        - vCPUs
        - Host Memory (GiB)
        - FP16/BF16 TFLOPS
        - INT8 TOPS
        - Device Memory (GiB)
        - Device Memory bandwidth (GiB/sec)
        - NeuronLink-v1 chip-to-chip bandwidth (GiB/sec/chip)
        - EFA bandwidth (Gbps)

    *   - Inf1.xlarge
        - 1
        - 4
        - 8
        - 64
        - 128
        - 8
        - 50
        - N/A
        - up-to 25


    *   - Inf1.2xlarge
        - 1
        - 8
        - 16
        - 64
        - 128
        - 8
        - 50
        - N/A
        - up-to 25

    *   - Inf1.6xlarge
        - 4
        - 24
        - 48
        - 256
        - 512
        - 32
        - 200
        - 32
        - 25

    *   - Inf1.24xlarge
        - 16
        - 96
        - 192
        - 1024
        - 2048
        - 128
        - 800
        - 32
        - 100



Inf1 offers a direct chip-to-chip interconnect called NeuronLink-v1,
which enables co-optimizing latency and throughput via the :ref:`Neuron Core Pipeline <neuroncore-pipeline>` technology. 

.. image:: /images/inf1-server-arch.png

