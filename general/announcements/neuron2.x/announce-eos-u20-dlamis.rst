.. post:: December 20, 2024
    :language: en
    :tags: announce-u20-dlami-dlc-eos

.. _announce-u20-dlami-dlc-eos:

Announcing end of support for Ubuntu20 DLCs and DLAMIs
------------------------------------------------------

Starting with :ref:`Neuron Release 2.21 <neuron-2.21.0-whatsnew>`, AWS Neuron will begin phasing out support for Ubuntu20 Deep Learning Containers (DLCs) and Deep Learning AMIs (DLAMIs). Neuron 2.21 will be the last release to provide bug fixes, and by Neuron 2.22, these offerings will no longer be available.

We recommend that all customers using Ubuntu20 DLCs and DLAMIs migrate to newer versions based on Ubuntu22 or Amazon Linux 2023. For customers who need to continue using Ubuntu20, you can create custom AMIs based on the Ubuntu20 base image and install Neuron components manually. Please see :ref:`container-faq` and :ref:`neuron-dlami-overview`. 

Please note that this does not affect support for the base Ubuntu20 operating system, which will continue to receive updates as per our standard support policy. For more information, please see :ref:`sdk-maintenance-policy`
