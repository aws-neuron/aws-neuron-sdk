.. _aws-inf2-arch:

AWS Inf2 Architecture
=====================

In this page, we provide an architectural overview of the AWS Inf2
instances, and the corresponding Inferentia2 NeuronDevices that power
them (Inferentia2 devices from here on).

Inf2 Architecture
-----------------

The EC2 Inf2 instance is powered by up to 12 :ref:`Inferentia2 devices <inferentia2-arch>`, and allows
customers to choose between four instances sizes:

.. list-table::
    :widths: auto
    :header-rows: 1
    :stub-columns: 1    
    :align: left

    *   - Instance size
        - # of Inferentia2 devices
        - vCPUs
        - Host Memory (GiB)
        - FP8/FP16/BF16/TF32 TFLOPS
        - FP32 TFLOPS
        - Device Memory (GiB)
        - Instance Memory Bandwidth (GiB/sec)
        - NeuronLink-v2 device-to-device (GiB/sec/device)

    *   - Inf2.xlarge
        - 1
        - 4
        - 16
        - 190
        - 47.5
        - 32
        - 820
        - N/A

    *   - Inf2.8xlarge
        - 1
        - 32
        - 128
        - 190
        - 47.5
        - 32
        - 820
        - N/A

    *   - Inf2.24xlarge
        - 6
        - 96
        - 384
        - 1140
        - 285
        - 192
        - 4920
        - 192

    *   - Inf2.48xlarge
        - 12
        - 192
        - 768
        - 2280
        - 570
        - 384
        - 9840
        - 192


Inf2 offers a low-latency and high-bandwidth chip-to-chip interconnect
called NeuronLink-v2, which enables high performance collective communication operations (e.g. AllReduce, AllGather).

This allows sharding large models across Inferentia2 devices (e.g. via
Tensor Parallelism), and thus optimizing latency and throughput. This
capability is especially useful when deploying Large Generative Models.

.. image:: /images/inf2-topology.jpg


