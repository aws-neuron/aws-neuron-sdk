.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-04-30

.. _byoc-hosting-devflow:

Bring Your Own Neuron Container to SageMaker Hosting (Inf1) — Archived
======================================================================

.. warning::

   This document is archived. The Inf1 SageMaker BYOC developer flow is no
   longer maintained as a standalone page. It is provided here for reference
   only.

The original page described how to compile a model on an EC2 Inf1 instance
or a SageMaker Notebook, then deploy it to SageMaker Hosting on ``ml.inf1``
using a custom container.

For current guidance, see:

- :doc:`Compile with Framework API and Deploy on EC2 Inf1 </deploy/ec2/inference>`
  for the EC2 Inf1 deployment flow.
- The AWS guide to `Adapting Your Own Inference Container
  <https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html>`_
  for the SageMaker Hosting BYOC pattern.
- :ref:`how-to-build-neuron-container` for building a Neuron container image.
- The :ref:`BYOC HuggingFace pretrained BERT container to SageMaker tutorial
  </src/examples/pytorch/byoc_sm_bert_tutorial/sagemaker_container_neuron.ipynb>`
  for an end-to-end BYOC example.
