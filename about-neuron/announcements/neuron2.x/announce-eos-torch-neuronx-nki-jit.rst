.. post:: May 15, 2025
    :language: en
    :tags: announce-eos-torch-neuronx-nki-jit

.. _announce-eos-torch-neuronx-nki-jit:

Announcing end of support for ``torch_neuronx.nki_jit`` API in ``torch-neuronx`` starting next release
---------------------------------------------------------------------------------------------------------

:ref:`Neuron Release 2.23 <neuron-2.23.0-whatsnew>` will be the last release to include support for ``torch_neuronx.nki_jit`` API in ``torch-neuronx`` package.

Customers using ``torch_neuronx.nki_jit`` API are recommended to switch invocations to directly call functions annotated with ``@nki.jit``.
