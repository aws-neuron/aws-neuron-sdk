.. post:: May 20, 2026
    :language: en
    :tags: announce-dlc-dlami-ubuntu-24-04

.. _announce-dlc-dlami-ubuntu-24-04:

Neuron DLCs and DLAMIs now based on Ubuntu 24.04 starting with Neuron 2.30
--------------------------------------------------------------------------

Starting with :ref:`Neuron Release 2.30.0 <neuron-2.30.0-whatsnew>`, all Neuron Deep Learning Containers (DLCs) and single-framework Deep Learning AMIs (DLAMIs) are now based on Ubuntu 24.04. Ubuntu 22.04-based DLCs and DLAMIs are no longer published starting with this release. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc` for the related Ubuntu 22.04 end-of-support notice.

The following DLCs are now based on Ubuntu 24.04:

* Neuron DLC Ubuntu PyTorch Training
* Neuron DLC Ubuntu PyTorch Inference (vLLM-NxDI)
* Neuron DLC Ubuntu PyTorch Inference (vLLM-Neuron)
* Neuron DLC Ubuntu JAX

The following DLAMIs are now based on Ubuntu 24.04:

* Neuron DLAMI Ubuntu PyTorch (PyTorch/XLA)
* Neuron DLAMI Ubuntu PyTorch (TorchNeuron)
* Neuron DLAMI Ubuntu JAX
* Neuron DLAMI Ubuntu Multi-Frameworks (JAX and TorchNeuron)

Customers using Ubuntu 22.04-based DLCs or DLAMIs should migrate to the new Ubuntu 24.04-based images. For customers who need to continue using Ubuntu 22.04, you can use the new Neuron Base DLAMI Ubuntu 22.04 and install frameworks manually, or use previously released DLCs/DLAMIs.

.. important::

    Neuron continues to support Ubuntu 22.04 at the platform level. Customers can still install Neuron packages directly on Ubuntu 22.04. This change only affects pre-built DLC and DLAMI images.

For more information, see :ref:`neuron-dlami-overview` and :doc:`/containers/index`.

