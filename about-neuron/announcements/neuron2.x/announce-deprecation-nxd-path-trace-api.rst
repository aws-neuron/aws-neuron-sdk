.. post:: September 18, 2025
    :language: en
    :tags: announce-deprecation-nxd-path-trace-api, al2

.. _announce-deprecation-nxd-path-trace-api:

Announcing the deprecation of the NeuronX Deep Learning Inference API path_trace function
-----------------------------------------------------------------------------------------

:ref:`Neuron release 2.26.0 <neuron-2-26-0-whatsnew>` is the last release supporting ``parallel_model_trace``. This NxD Inference function will be deprecated in the next version of the Neuron SDK in favor of the ``ModelBuilder.trace()`` method, which provides a more robust and flexible approach for tracing and compiling models for Neuron devices,  enabling more advanced features such as weight layout optimization support, as well as other quality-of-life and stability improvements for SPMD tracing.

For customers directly invoking ``parallel_model_trace``, they can now use ModelBuilderV2 APIs. For more details on these APIS, see :ref:`nxd-core-model-builder-v2`. For customers that are directly using models in NxDI, there is  no impact since NxDI models are already built on MBv1 which has no issues.