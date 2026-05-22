.. meta::
   :description: Run Neuron training workloads on AWS Batch with automatic scaling and resource management on Trainium instances.
   :keywords: AWS Batch, Neuron, training, Trainium, batch computing, distributed training, EFA
   :date-modified: 04/20/2026

.. _deploy-batch:
.. _aws_batch_flow:

AWS Batch
=========

AWS Batch provides scalable batch computing for Neuron training workloads. Configure your training job and let Batch manage orchestration, execution, and dynamic scaling of Trainium compute resources.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Train on AWS Batch
      :link: /deploy/batch/training
      :link-type: doc
      :class-card: sd-border-1

      Build a training container, configure a Batch compute environment with Trainium instances, and submit distributed training jobs.

.. toctree::
   :maxdepth: 1
   :hidden:

   Train on AWS Batch </deploy/batch/training>
