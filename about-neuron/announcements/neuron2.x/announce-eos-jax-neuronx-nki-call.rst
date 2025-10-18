.. post:: April 3, 2025
    :language: en
    :tags: announce-eos-jax-neuronx-features

.. _announce-eos-jax-neuronx-features:

Announcing end of support for ``jax_neuronx.nki_call`` API in ``jax-neuronx`` from  starting next release
------------------------------------------------------------------------------------------------------------

Starting with :ref:`Neuron Release 2.23 <neuron-2.23.0-whatsnew>`, Neuron will end support for ``jax_neuronx.nki_call`` API in ``jax-neuronx`` package.

For a full list of features that require ``jax-neuronx``, please see :ref:`jax-neuron-known-issues`. 

Customers using ``jax_neuronx.nki_call`` API are recommended to switch invocations to directly call functions annotated with :ref:`nki.jit API <nki_decorators>`.
