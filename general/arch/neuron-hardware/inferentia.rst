.. _inferentia-arch:


Inferentia Architecture
-----------------------

At the heart of the Inf1 instance are 16 x Inferentia devices (each Inferentia include 4 x :ref:`NeuronCore-v1 <neuroncores-v1-arch>`), as depicted
below:

.. image:: /images/inferentia-neurondevice.png

Each Inferentia device consists of:

-  Compute:
    * 4x :ref:`NeuronCore-v1 <neuroncores-v1-arch>` cores, delivering 128 INT8 TOPS and 64 FP16/BF16 TFLOPS

-  Device Memory:
    * 8GB of device DRAM memory  (for storing parameters and intermediate state), with 50 GB/sec of bandwidth

-  NeuronLink:
    * Enables co-optimization of latency
      and throughput via the :ref:`Neuron Core Pipeline <neuroncore-pipeline>`   technology

