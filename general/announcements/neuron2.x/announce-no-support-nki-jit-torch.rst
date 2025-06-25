.. post:: June 24, 2025
    :language: en
    :tags: announce-no-longer-support-nki-jit

.. _announce-no-longer-support-nki-jit:

Neuron no longer supports nki_jit API in PyTorch Neuron starting this release
--------------------------------------------------------------------------------

Starting with :ref:`Neuron Release 2.24 <neuron-2.24.0-whatsnew>`, ``torch_neuronx.nki_jit`` API in ``torch-neuronx`` package is no longer supported.

**I currently use nki_jit in my PyTorch models. What do I do?**

Customers using ``torch_neuronx.nki_jit`` API are recommended to switch invocations to directly call functions annotated with :ref:`nki.jit API <nki_decorators>`.
