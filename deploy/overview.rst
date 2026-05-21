.. meta::
   :description: Choose the right deployment configuration for your Neuron workloads. Compare DLAMIs, DLCs, EKS, ECS, Batch, ParallelCluster, and SageMaker for training and inference on Trainium and Inferentia.
   :keywords: Neuron deployment, use cases, DLAMI, DLC, EKS, ECS, Batch, ParallelCluster, SageMaker, training, inference, vLLM, Trainium, Inferentia, DRA
   :date-modified: 04/20/2026

.. _deploy-overview:

Choose your deployment path
============================

AWS Neuron supports multiple deployment configurations for training and inference on Trainium and Inferentia instances. This page helps you choose the right combination of environment, compute service, and infrastructure based on your workload requirements.

.. contents:: On this page
   :local:
   :depth: 2

----

Quick start: which path is right for you?
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - I want to...
     - Recommended path
     - Get started
   * - Prototype a model on a single instance
     - DLAMI on EC2
     - :doc:`/deploy/environments/dlami`
   * - Serve an LLM with vLLM
     - vLLM DLC on EC2 or EKS
     - :doc:`/deploy/environments/quickstart-deploy-dlc`
   * - Run distributed training across multiple nodes
     - DLC on EKS or ParallelCluster
     - :doc:`/deploy/eks/index`
   * - Run batch training jobs on a schedule
     - DLC on AWS Batch
     - :doc:`/deploy/batch/index`
   * - Use managed infrastructure with minimal setup
     - Amazon SageMaker
     - :doc:`/deploy/sagemaker/index`
   * - Build a production Kubernetes inference service
     - DLC on EKS with DRA
     - :doc:`/deploy/eks/dra`
   * - Run containerized tasks without Kubernetes
     - DLC on ECS
     - :doc:`/deploy/ecs/index`

----

Choose your environment
------------------------

Your first decision is how you want the Neuron SDK installed. Neuron provides three pre-configured environment types, each suited to different workflows.

Deep Learning AMIs (DLAMIs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use when**: You want the fastest path to running code on a single EC2 instance. DLAMIs come with Neuron drivers, frameworks, and virtual environments pre-installed. Launch an instance and start working in minutes.

**Best for**:

- Interactive development with SSH or Jupyter notebooks
- Prototyping and experimentation on a single instance
- Teams that want pre-configured virtual environments for PyTorch, JAX, or vLLM

**DLAMI types**:

- **Multi-Framework DLAMI** — includes PyTorch, JAX, vLLM, and NxD libraries in separate virtual environments. Use this when you want to explore multiple frameworks or switch between training and inference workflows.
- **Single Framework DLAMI** — optimized for one framework version. Use this for production deployments where you know exactly which framework you need.
- **Base DLAMI** — includes only Neuron drivers, EFA, and tools. Use this as a foundation for containerized applications or custom builds where you install your own packages.

**Get started**: :doc:`/deploy/environments/dlami`

Deep Learning Containers (DLCs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use when**: You need portable, reproducible environments for orchestrated deployments. DLCs are Docker images pre-built with Neuron SDK and a specific framework, available in Amazon ECR.

**Best for**:

- Production deployments on EKS, ECS, or AWS Batch
- CI/CD pipelines that require consistent environments
- Multi-node distributed training where each node runs the same container
- vLLM inference serving in containerized environments

**Available containers**:

- PyTorch Training, PyTorch Inference, PyTorch vLLM Inference, JAX Training

**Get started**: :doc:`/deploy/environments/dlc-images` | :doc:`/deploy/environments/quickstart-deploy-dlc`

Custom Docker containers
^^^^^^^^^^^^^^^^^^^^^^^^^

**Use when**: You need full control over the container environment — custom dependencies, specific package versions, or a CI/CD pipeline that builds images from scratch.

**Best for**:

- Teams with existing Docker build pipelines
- Workloads requiring packages not included in DLCs
- Environments with strict security or compliance requirements

**Get started**: :doc:`/deploy/environments/docker-setup` | :doc:`/deploy/environments/customize-dlc`

----

Choose your compute service
----------------------------

Your second decision is where to run your workload. Each AWS compute service offers different trade-offs between control, automation, and operational overhead.

Amazon EC2 (direct instance access)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use when**: You want direct access to Neuron hardware on a single instance or a small number of instances. EC2 gives you full control over the instance lifecycle.

**Best for**:

- Development and prototyping
- Single-node training and inference
- Interactive debugging with SSH access
- Running Jupyter notebooks

**Typical workflow**: Launch a DLAMI, SSH in, activate a virtual environment, run your code.

**Get started**: :doc:`/deploy/ec2/index`

Amazon EKS (Kubernetes orchestration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use when**: You need Kubernetes-based orchestration for containerized Neuron workloads. EKS provides device plugins, topology-aware scheduling, health monitoring, and Dynamic Resource Allocation (DRA) for Neuron devices.

**Best for**:

- Production inference services with auto-scaling
- Multi-node distributed training with EFA networking
- Teams already using Kubernetes for workload management
- Workloads requiring topology-aware device allocation (DRA)
- Multi-node inference on Trn2 UltraServers

**Key capabilities**:

- **Neuron device plugin** — exposes Neuron hardware to the Kubernetes scheduler
- **Neuron Helm chart** — installs all infrastructure components with a single command (:doc:`/deploy/eks/helm-chart`)
- **Dynamic Resource Allocation (DRA)** — attribute-based device selection and topology-aware allocation on K8s 1.34+ (:doc:`/deploy/eks/dra`)
- **UltraServer support** — schedule MPI jobs across Trn2 UltraServer nodes (:doc:`/deploy/eks/ultraserver`)
- **Node problem detector** — automatic health monitoring and node replacement (:doc:`/deploy/infrastructure/problem-detector`)

**Get started**: :doc:`/deploy/eks/index` | :doc:`/deploy/eks/setup`

Amazon ECS (container task orchestration)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use when**: You want container orchestration without Kubernetes. ECS provides task-based scheduling for Neuron containers with simpler operational overhead than EKS.

**Best for**:

- Teams already using ECS for container workloads
- Simpler container deployments that don't need Kubernetes features
- Workloads where task-based scheduling is sufficient

**Get started**: :doc:`/deploy/ecs/index`

AWS Batch (batch job scheduling)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use when**: You have training jobs that run on a schedule or in response to events, and you want AWS to manage compute scaling automatically.

**Best for**:

- Periodic or scheduled training jobs
- Workloads with variable compute demand
- Teams that want automatic resource provisioning and cleanup

**Get started**: :doc:`/deploy/batch/index`

AWS ParallelCluster (HPC with Slurm)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use when**: You need an HPC cluster with Slurm for large-scale distributed training. ParallelCluster manages the cluster lifecycle including head nodes, compute fleets, and shared storage.

**Best for**:

- Large-scale distributed training across many Trn1 nodes
- Teams familiar with Slurm job scheduling
- Workloads requiring shared filesystems (EFS, FSx)

**Get started**: :doc:`/deploy/parallelcluster/index`

Amazon SageMaker (managed ML platform)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Use when**: You want a fully managed ML platform that handles infrastructure provisioning, training orchestration, and model deployment. SageMaker abstracts away the compute management.

**Best for**:

- Teams that prefer managed services over self-managed infrastructure
- End-to-end ML workflows (data preparation → training → deployment)
- Fine-tuning foundation models with SageMaker JumpStart
- Resilient training with SageMaker HyperPod (automatic checkpointing and recovery)

**Get started**: :doc:`/deploy/sagemaker/index`

----

Common deployment patterns
---------------------------

Single-instance development and prototyping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Launch a Multi-Framework DLAMI on an Inf2 or Trn1 instance. Activate the virtual environment for your framework and iterate on your model. This is the fastest path from zero to running code.

1. :doc:`Launch a DLAMI </deploy/environments/dlami>`
2. :doc:`Train on EC2 </deploy/ec2/training>` or :doc:`run inference </deploy/ec2/inference-inf2>`

vLLM inference serving
^^^^^^^^^^^^^^^^^^^^^^^

Deploy a vLLM DLC on EC2 for single-instance serving, or on EKS for production with auto-scaling. The vLLM DLC includes the Neuron vLLM plugin with continuous batching, speculative decoding, and OpenAI-compatible APIs.

1. :doc:`Quickstart: Deploy a DLC with vLLM </deploy/environments/quickstart-deploy-dlc>`
2. For production: :doc:`Deploy on EKS </deploy/eks/inference>` with :doc:`DRA </deploy/eks/dra>` for topology-aware device allocation

Multi-node distributed training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use DLCs on EKS or ParallelCluster for distributed training across multiple Trn1 or Trn2 nodes with EFA networking.

- **EKS path**: :doc:`Set up EKS </deploy/eks/setup>` → :doc:`Deploy training </deploy/eks/training>`
- **ParallelCluster path**: :doc:`Set up ParallelCluster </deploy/parallelcluster/training>`
- **Batch path**: :doc:`Train on AWS Batch </deploy/batch/training>`

Production Kubernetes inference with DRA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For production inference on EKS with Trn2 instances, use Dynamic Resource Allocation (DRA) for topology-aware device scheduling. DRA replaces the need for custom scheduler extensions and enables attribute-based device selection.

1. :doc:`Set up EKS </deploy/eks/setup>` with :doc:`Helm chart </deploy/eks/helm-chart>`
2. :doc:`Configure DRA </deploy/eks/dra>` for topology-aware allocation
3. For UltraServer workloads: :doc:`Schedule MPI jobs </deploy/eks/ultraserver>`

----

Further reading
----------------

- :doc:`/deploy/environments/index` — Compare DLAMIs, DLCs, and custom Docker environments
- :doc:`/deploy/infrastructure/index` — Neuron Kubernetes plugins, monitoring, and scheduling
- :doc:`/deploy/faq` — Common questions about Neuron container deployments
- :doc:`/deploy/third-party/index` — Partner integrations (Ray, Domino)
