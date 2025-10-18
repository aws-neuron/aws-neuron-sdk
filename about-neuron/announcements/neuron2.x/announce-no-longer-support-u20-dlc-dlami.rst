.. post:: April 3, 2025
    :language: en
    :tags: announce-u20-dlami-dlc-no-longer-support

.. _announce-u20-dlami-dlc-eos:

Neuron no longer includes support for Ubuntu20 DLCs and DLAMIs starting this release
-------------------------------------------------------------------------------------

Starting with :ref:`Neuron Release 2.22 <neuron-2.22.0-whatsnew>`, Neuron no longer includes offerings for Ubuntu20 Deep Learning Containers (DLCs) and Deep Learning AMIs (DLAMIs). 

Customers using Ubuntu20 DLCs and DLAMIs should migrate to newer versions based on Ubuntu22 or Amazon Linux 2023. For customers who need to continue using Ubuntu20, you can create custom AMIs based on the Ubuntu20 base image and install Neuron components manually. Please see :ref:`container-faq` and :ref:`neuron-dlami-overview`. 

Please note that this does not affect support for the base Ubuntu20 operating system, which will continue to receive updates as per our standard support policy. For more information, please see :ref:`sdk-maintenance-policy`
