.. post:: December 16, 2025
    :language: en
    :tags: announcement-end-of-support-parallel-model-trace

.. _announcement-end-of-support-parallel-model-trace:

Neuron no longer supports parallel_model_trace API starting with Neuron 2.27
-----------------------------------------------------------------------------

Starting with the Neuron 2.27 release, the :ref:`parallel_model_trace API <nxd_tracing>` is no longer supported for inference. We introduced the :doc:`Model Builder V2 API </libraries/neuronx-distributed/model_builder_v2_api_reference>` in Neuron 2.25 as an alternative to the tracing API, and it is now the default API in Neuron for model tracing.

Customers can migrate to the Model Builder V2 API by following the reference `Llama-3.2-1B inference sample <https://github.com/aws-neuron/neuronx-distributed/tree/main/examples/inference/llama>`__.
