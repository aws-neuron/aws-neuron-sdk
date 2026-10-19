.. _inference-legacy-home:

Legacy inference frameworks
===========================

.. meta::
   :description: Legacy NxD Inference and NxD Core documentation for Trainium/Inferentia inference on Trn1/Inf2 and custom NxDI-API serving. End of support announced starting Neuron 2.34; use vLLM Neuron for new work.
   :keywords: AWS Neuron, NxD Inference, NxDI, NxD Core, Trn1, Inf2, diffusion, Flux, vLLM Neuron, inference
   :date-modified: 2026-08-28

.. note:: End of support

   NxD Inference and the inference-context NxD Core documentation in this section are **legacy**
   and are **announced end of support on Trn2 and Trn3 starting the Neuron 2.34 release**. For
   new inference work, the recommended path is the
   :ref:`vLLM Neuron plugin <vllm-neuron-home>`; the task-oriented
   :ref:`Inference on Neuron <inference-home>` page shows where each path fits.

This section is the home for the legacy NxD Inference (NxDI) and inference-context NxD Core
documentation on the AWS Neuron SDK. It applies to developers who are still running NxDI-based
inference on Trn1/Inf2 instances or building custom serving on the NxD Inference API, while
vLLM Neuron becomes the recommended path.

The content here is live, indexed, and searchable — it is not archived. It is kept together as
a single reference section so that inference content that is legacy but no longer recommended
for new work stays discoverable in one place.

Applies to
----------

This section is applicable to:

- Inference on Trn1 and Inf2 using NxD Inference (NxDI).
- Custom serving built on the NxD Inference API, such as diffusion and Flux models.
- Inference-context NxD Core primitives used by NxDI.

What lives here
---------------

- **NxD Inference (NxDI)** — the legacy inference library: setup, model support,
  developer guides, tutorials, and API references.
- **NxD Core (inference context)** — the inference-context core primitives, developer guides,
  and tutorials that NxDI builds on.

Where to go for new work
------------------------

For new inference projects, start from the recommended path rather than this section:

- The :ref:`vLLM Neuron plugin <vllm-neuron-home>` is the recommended top-level inference
  section.
- The task-oriented :ref:`Inference on Neuron <inference-home>` overview shows where each
  serving path fits, including when NxD Inference remains the right choice.

Related reference
-----------------

- :ref:`Inference on Neuron <inference-home>` — the task-oriented entry point for choosing an
  inference path on Neuron.
- :ref:`vLLM Neuron plugin <vllm-neuron-home>` — the recommended inference section for new work.

.. toctree::
   :maxdepth: 1
   :caption: Legacy inference frameworks
   :hidden:

   NxD Inference <nxd-inference/index>
   NxD Core (inference) <nxd-core/index>
