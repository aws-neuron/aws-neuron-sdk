.. _neuron-runtime:

Neuron runtime
==============

Neuron runtime consists of kernel driver and C/C++ libraries which provides APIs to access Inferentia devices. 
Machine learning framewworks (TensorFlow, PyTorch and Apatche Mxnet) uses Neuron runtime to execute models on the Neuron Cores.
Neuron runtime load compiled deep learning models, also referred to as Neuron Executable File Format (NEFF) to the
Inferentia chips to execute inference requests. 
Runtime is optimized for high-throughput and low-latency to meet customers ML applications requirements.
In typical environment, Neuron runtime is transparent for users. User application will communicate with 
Neuron runtime through framework (TensorFlow, PyTorch, MxNet) API. 


First generation of Neuron runtime (:ref:`Neuron Runtime 1.x <nrt_v1>`) was delivered as daemon (``neuron-rtd``) which 
provided GRPC API to load and execute a ML model. Runtime 1.x was available before *Neuron 1.16.0*.

Second generation of Neuron runtime (Neuron Runtime 2.x) is available starting *Neuron 1.16.0* and is delivered as a 
shared library (``libnrt.so``), librnt.so is installed together with the framework of choice, it improves
the performance and ease of use by providing C++ library to load and execute ML models.

Visit :ref:`introduce-libnrt` for more information.

Neuron Runtime 2.x
------------------

.. toctree::
   :maxdepth: 1

   Configuration <nrt-configurable-parameters>
   Troubleshooting <nrt-troubleshoot>
   FAQ <faq>
   What's New <rn>

Neuron Runtime 1.x
------------------

.. toctree::
   :maxdepth: 1

   Runtime 1.x documentation <./v1/index>


