.. post:: June 24, 2025
    :language: en
    :tags: announce-eos-neuron-driver-2.21-version, neuron-driver-version, inf1

.. _announce-upcoming-neuron-driver-2.21-version support changes for inf1 instance:


Upcoming changes to Neuron driver 2.21 support for Inf1 starting Neuron 2.26 release
------------------------------------------------------------------------------------

Starting with Neuron Release 2.26, Neuron driver versions above 2.21 will only support non-Inf1 instances (such as ``Trn1``, ``Inf2``, or other instance types). 
For ``Inf1`` instance users, Neuron driver versions <  2.21 will remain supported with regular security patches. 

``Inf1`` instance users are advised to pin the Neuron driver version to ``2.21.*`` in their installation script. 
Refer to the :ref:`Neuron Driver release [2.22.2.0] <neuron-driver-release-notes>` for detailed instructions on pinning the Neuron Driver.  

