.. post:: May 15, 2025
    :language: en
    :tags: 

.. _announce-eos-jax-neuronx-features:

Neuron no longer supports ``jax_neuronx.nki_call`` API in ``jax-neuronx`` starting this release
-------------------------------------------------------------------------------------------------

:ref:`Neuron Release 2.23 <neuron-2.23.0-whatsnew>` no longer supports ``jax_neuronx.nki_call`` API in ``jax-neuronx`` package.

For a full list of features that require ``jax-neuronx``, please see :ref:`jax-neuron-known-issues`. 

Customers using ``jax_neuronx.nki_call`` API will need to switch invocations to directly call functions annotated with :ref:`nki.jit API <nki_decorators>`.
