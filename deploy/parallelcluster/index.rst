.. meta::
   :description: Set up AWS ParallelCluster with Slurm for distributed training on Trainium instances using the Neuron SDK.
   :keywords: ParallelCluster, Neuron, Slurm, HPC, distributed training, Trainium, cluster
   :date-modified: 04/20/2026

.. _deploy-parallelcluster:

AWS ParallelCluster
====================

AWS ParallelCluster provides HPC cluster management with Slurm for distributed training on Trainium instances. Set up a cluster with a head node and Trn1 compute fleet, then submit training jobs using Slurm.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Train on ParallelCluster
      :link: /deploy/parallelcluster/training
      :link-type: doc
      :class-card: sd-border-1

      Set up VPC infrastructure, create a ParallelCluster with Trn1 nodes, and submit distributed training jobs with Slurm.

.. toctree::
   :maxdepth: 1
   :hidden:

   Train on ParallelCluster </deploy/parallelcluster/training>
