.. _aws-inf1-arch:

AWS Inf1 Architecture
=====================

In this page, we provide an architectural overview of the AWS Inf1
instances, and the corresponding :ref:`Inferentia <inferentia-arch>` NeuronDevices that power
them (:ref:`Inferentia <inferentia-arch>` devices from here on).

.. contents:: Table of contents
   :local:
   :depth: 2

.. _inf1-arch:

Inf1 Architecture
-----------------

The EC2 Inf1 instance is powered by 16 :ref:`Inferentia <inferentia-arch>` devices, and allows
customers to choose between four instances sizes:

.. list-table::
    :widths: auto
    :header-rows: 1
    :stub-columns: 1    
    :align: left
    

    *   - Instance size
        - # of Inferentia devices
        - vCPUs
        - Host Memory (GiB)
        - FP16/BF16 TFLOPS
        - INT8 TOPS
        - Device Memory (GiB)
        - Device Memory Bandwidth (GiB/sec)
        - NeuronLink-v1 device-to-device bandwidth (GiB/sec/device)
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



Inf1 offers a direct device-to-device interconnect called NeuronLink-v1,
which enables co-optimizing latency and throughput via the :ref:`Neuron Core Pipeline <neuroncore-pipeline>` technology. 

.. image:: /images/inf1-server-arch.png

