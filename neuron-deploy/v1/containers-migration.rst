.. _containers-migration-to-runtime2:

Migration to Neuron Runtime 2.x (libnrt.so)
===========================================

Please refer this section only if neuron containers were already setup as per :ref:`packaging-neuron-rt-containers.rst` and are updating to the Neuron SDK software version 1.16.0 and beyond.

Application and Neuron Runtime in the same container
-----------------------------------------------------

Follow the steps in :ref:`Running Application Container<running-application-container>`

Application and Neuron Runtime in different container
-----------------------------------------------------

#. Upgrade your application container as per :ref:`running-application-container` section above.
#. With the Neuron Runtime library in the container there is no need to run a separate runtime container. You can now stop the runtime container.

Application in container and Neuron Runtime directly on host
------------------------------------------------------------

#. Upgrade your application container as per :ref:`running-application-container` section above.
#. With the Neuron Runtime library in the container there is no need to run the host runtime. You can now stop the host runtime with ``sudo systemctl stop neuron-rtd`` or ``sudo killall neuron-rtd``.


