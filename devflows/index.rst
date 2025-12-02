.. _neuron-devflows:

.. meta::
      :description:
      :date-modified:

AWS Workload Orchestration
==========================

AWS Neuron integrates seamlessly with various AWS compute and orchestration services to accelerate deep learning workloads. This section provides deployment patterns and best practices for running Neuron-powered applications across different AWS services, from container orchestration to high-performance computing clusters.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Amazon EKS
      :link: /devflows/eks-flows
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Deploy Neuron workloads on Kubernetes with Amazon Elastic Kubernetes Service

   .. grid-item-card:: Amazon ECS
      :link: /devflows/ecs-flows
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Run containerized Neuron applications using Amazon Elastic Container Service

   .. grid-item-card:: AWS ParallelCluster
      :link: /devflows/parallelcluster-flows
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Set up HPC clusters for distributed training and inference workloads

   .. grid-item-card:: AWS Batch
      :link: /devflows/aws-batch-flows
      :link-type: doc
      :class-body: sphinx-design-class-title-small

      Execute batch ML jobs with automatic scaling and resource management

.. toctree::
    :maxdepth: 1
    :hidden:

    /devflows/eks-flows
    /devflows/ecs-flows
    /devflows/parallelcluster-flows
    /devflows/aws-batch-flows
    Amazon SageMaker </devflows/sagemaker-flows>
    Third-party Solutions </devflows/third-party-solutions>

