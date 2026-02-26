.. _neuron-2-27-0-compiler:

.. meta::
   :description: The official release notes for the AWS Neuron SDK compiler component, version 2.27.0. Release date: 12/19/2025.

AWS Neuron SDK 2.27.0: Neuron Compiler release notes
====================================================

**Date of release**: December 19, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.27.0 release notes home <neuron-2-27-0-whatsnew>`
* Review older release notes in the :ref:`Previous Neuron Releases <previous-neuron-releases>` section.

Changes and improvements
-------------------------

* **Error code docs** New error code documentation has been added to help developers better understand and troubleshoot issues encountered during model compilation. Check them out here: :doc:`Neuron Compiler Error Codes </compiler/error-codes/index>`

* **Compiler accuracy flag defaults updated**: Two Neuron Compiler (neuronxcc) flags now have different default behaviors to improve accuracy. The ``--auto-cast`` flag now defaults to ``none`` (previously ``matmul``), and ``--enable-mixed-precision-accumulation`` is now enabled by default. These changes optimize accuracy but may impact performance for FP32 models and models using smaller bitwidth dtypes. To restore previous behavior, explicitly set ``--auto-cast=matmul`` and use the new ``--disable-mixed-precision-accumulation`` flag.

* **Python 3.9 no longer supported**: The Neuron Compiler requires Python 3.10 or higher. Users currently on Python 3.9 must upgrade to continue using the Neuron Compiler with Python bindings.

