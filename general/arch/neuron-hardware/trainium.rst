.. _trainium-arch:


Trainium Architecture
----------------------

At the heart of the Trn1 instance are 16 x Trainium devices (each Trainium include 2 x :ref:`NeuronCore-v2 <neuroncores-v2-arch>`). Trainium is the second
generation purpose-built Machine Learning accelerator from AWS. The
Trainium device architecture is depicted below:

.. image:: /images/trainium-neurondevice.png

Each Trainium device consists of:

-  Compute:
    * 2x :ref:`NeuronCore-v2 <neuroncores-v2-arch>` cores, delivering 420 INT8 TOPS, 190 FP16/BF16/cFP8/TF32 TFLOPS, and
      47.5 FP32 TFLOPS.

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
      Stochastic Rounding), and custom-operators via the deeply embedded GPSIMD engines.


