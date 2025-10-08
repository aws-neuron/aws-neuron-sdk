.. _trainium2_arch:

Trainium2 Architecture Guide for NKI
===============================================

This guide covers hardware architecture of third-generation NeuronDevices: Trainium2.
We assume readers have gone through :doc:`Trainium/Inferentia2 Architecture Guide <trainium_inferentia2_arch>`
in detail to understand the basics of NeuronDevice Architecture.

:numref:`Fig. %s <fig-arch-neuron-device-v3>` shows a block diagram of a Trainium2 device, which consists of:

* 8 NeuronCores (v3).
* 4 HBM stacks with a total device memory capacity of 96GiB and bandwidth of 2.9TB/s.
* 128 DMA (Direct Memory Access) engines to move data within and across devices.
* 20 CC-Cores for collective communication.
* 4 NeuronLink-v3 for device-to-device collective communication.

.. _fig-arch-neuron-device-v3:

.. figure:: ../img/arch_images/neuron_device3.png
   :align: center
   :width: 70%

   Trainium2 Device Diagram.

For a high-level architecture specification comparison from Trainium1 to Trainium2, check out
`Neuron architecture guide <https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html#trainium2-performance-improvements>`_.
