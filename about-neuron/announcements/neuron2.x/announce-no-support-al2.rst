.. post:: September 16, 2024
    :language: en
    :tags: end-support-al2

.. _eos-al2:

Neuron Runtime no longer supports Amazon Linux 2 (AL2)
========================================================

Starting from :ref:`Neuron release 2.20 <neuron-2.20-whatsnew>`, the Neuron Runtime (``aws-neuronx-runtime-lib``) will no longer support Amazon Linux 2 (AL2). 
The Neuron Driver (``aws-neuronx-dkms``) is now the only Neuron package that supports Amazon Linux 2. 
However, the Neuron Driver requires Linux kernel 5.10 or higher. Since default AL2 AMIs ship with kernel 4.14, you must upgrade your AL2 kernel to 5.10+ before installing driver versions 2.18 and later, or migrate to Amazon Linux 2023 or Ubuntu which include compatible kernels by default.

This change introduces the following constraint:

Customers cannot run their full Neuron-powered applications natively on an AL2-based Amazon Machine Image (AMI). To leverage Neuron functionality on an AL2 AMI, customers must containerize their applications using a Neuron supported container with non-AL2 Linux distribution (e.g., Ubuntu 22.04, Amazon Linux 2023, etc.) and then deploy those containers on an AL2-based AMI that has the Neuron Driver (``aws-neuronx-dkms``) installed.

How does this impact me?
------------------------

**I have an AL2 DLAMI**

If you are using one of the following Amazon
Linux 2 DLAMIs, please migrate to a supported DLAMI (e.g., Ubuntu 22.04, Amazon Linux 2023 (AL2023), etc.). Please see :ref:`neuron-dlami-overview` for
a list of all supported DLAMIs to migrate to.

+-----------------+------------------+-----------------------------------------------------------+
|    Framework    | Operating System |                        DLAMI Name                         |
+=================+==================+===========================================================+
|  PyTorch 1.13   |  Amazon Linux 2  |  Deep Learning AMI Neuron PyTorch 1.13 (Amazon Linux 2)   |
+-----------------+------------------+-----------------------------------------------------------+
| TensorFlow 2.10 |  Amazon Linux 2  | Deep Learning AMI Neuron TensorFlow 2.10 (Amazon Linux 2) |
+-----------------+------------------+-----------------------------------------------------------+

**I am using my own AL2 Container**

If you using your own AL2 Container, please migrate to a Neuron supported container with non-AL2 Linux distribution (e.g., Ubuntu 22.04, Amazon Linux 2023, etc.)

**I am using a base AL2 DLAMI**

If you are using a base Amazon Linux 2 DLAMI, please ensure the Neuron Driver (``aws-neuronx-dkms``) is the only Neuron package installed. Please use non AL2 (e.g., Ubuntu 22.04, Amazon Linux 2023, etc.) containers to run your Neuron applications.

.. note::
   Neuron does not supports Linux kernel versions < 5.10. Customers using
   Linux kernel versions < 5.10 must migrate to >= 5.10.
