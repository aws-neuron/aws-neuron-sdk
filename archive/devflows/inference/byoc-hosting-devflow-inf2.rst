.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-04-30

.. _byoc-hosting-devflow-inf2:

Bring Your Own Neuron Container to SageMaker Hosting (Inf2 or Trn1) — Archived
==============================================================================

.. warning::

   This document is archived. The Inf2 and Trn1 SageMaker BYOC developer flow
   is no longer maintained as a standalone page. It is provided here for
   reference only.

The original page described how to compile a model on an EC2 instance or a
SageMaker Notebook, then deploy it to SageMaker Hosting on ``ml.inf2`` or
``ml.trn1`` using a custom container.

For current guidance, see:

- :doc:`Compile with Framework API and Deploy on EC2 Inf2 </deploy/ec2/inference-inf2>`
  for the EC2 Inf2 deployment flow.
- The AWS guide to `Adapting Your Own Inference Container
  <https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html>`_
  for the SageMaker Hosting BYOC pattern.
- :ref:`how-to-build-neuron-container` for building a Neuron container image.
- The `Compiling and Deploying HuggingFace Pretrained BERT on Inf2 on Amazon
  SageMaker sample
  <https://github.com/aws-neuron/aws-neuron-sagemaker-samples/tree/master/inference/inf2-bert-on-sagemaker>`_
  for an end-to-end Inf2 BYOC example.
