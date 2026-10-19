.. post:: October 16, 2026
    :language: en
    :tags: announce-intent-eos-neuron-ls-json

.. _announce-intent-eos-neuron-ls-json:

Announcing end of support for the neuron-ls JSON output format starting with Neuron 2.34
========================================================================================

Starting with the Neuron 2.34 release, ``neuron-ls`` no longer supports its current JSON
output (``--json-output``, ``-j``). Neuron 2.34 replaces it with an updated JSON output
that you select with ``--output-format json``, the same flag that Neuron Explorer uses.

The updated format adds top-level fields for information that applies to more than one
device, such as instance and UltraServer details.

If you parse ``neuron-ls`` JSON output, plan to migrate to the updated format. For current
``neuron-ls`` usage, see :doc:`/tools/neuron-sys-tools/neuron-ls`.
