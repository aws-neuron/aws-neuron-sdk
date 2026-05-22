.. meta::
   :description: Deploy AWS Neuron workloads on EC2, EKS, ECS, AWS Batch, ParallelCluster, and SageMaker using Deep Learning AMIs and Containers for training and inference on Trainium and Inferentia.
   :keywords: Neuron deployment, DLC, DLAMI, EKS, ECS, EC2, Batch, ParallelCluster, SageMaker, Kubernetes, containers, Docker, Trainium, Inferentia, vLLM, DRA
   :date-modified: 04/20/2026

.. _neuron-deploy:
.. _neuron_containers:
.. _neuron1-devflows:
.. _compilation-flow-target:
.. _deploym-flow-target:

Deploy on AWS
=============

Run your training and inference workloads on AWS Trainium and Inferentia instances. This section covers everything from launching your first instance to running production Kubernetes services — choose a pre-configured environment, pick a compute service, and deploy.

.. admonition:: New to Neuron deployment?

   Read :doc:`/deploy/overview` to compare deployment options side by side and find the right path for your workload.

----

Start with a pre-configured environment
-----------------------------------------

Pick a pre-configured environment based on **how you'll run the workload**, not which framework you use:

* **One EC2 instance, interactive work →** use a **Deep Learning AMI**.
* **Orchestrated containers (EKS, ECS, Batch, SageMaker) →** use a **Deep Learning Container**.
* **Need to control the OS image or the dependency set →** use a **custom Docker build**, optionally based on a DLC.

DLAMIs and DLCs share the same Neuron SDK and frameworks; the difference is the unit of deployment.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: **Deep Learning AMIs**
      :link: /deploy/environments/dlami
      :link-type: doc
      :class-card: sd-border-1

      An EC2 AMI with Neuron SDK, frameworks, and virtual environments pre-installed. Multi-framework, single-framework, and base variants. Fastest path from zero to running code on a single instance.

      +++
      *Use for: EC2 development on Inf1, Inf2, Trn1, or Trn2; prototyping; Jupyter notebooks; Slurm clusters*

   .. grid-item-card:: **Deep Learning Containers**
      :link: /deploy/environments/dlc-images
      :link-type: doc
      :class-card: sd-border-1

      Pre-built Docker images on Amazon ECR with Neuron SDK and a specific framework. PyTorch training, PyTorch inference, vLLM inference, and JAX training images.

      +++
      *Use for: EKS, ECS, AWS Batch, SageMaker, vLLM serving, multi-node training*

   .. grid-item-card:: **Custom Docker builds**
      :link: /deploy/environments/docker-setup
      :link-type: doc
      :class-card: sd-border-1

      Install Neuron drivers, configure Docker, and build containers from scratch — or extend a DLC with :doc:`custom packages </deploy/environments/customize-dlc>`. Full control over every dependency.

      +++
      *Use for: Custom dependencies, hardened base images, internal CI/CD pipelines*

----

Deploy on an AWS compute service
----------------------------------

Choose where to run your workload. EC2 pairs with a DLAMI; every other service consumes a DLC (or a custom container).

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: **Amazon EC2**
      :link: /deploy/ec2/index
      :link-type: doc
      :class-card: sd-border-1

      Direct access to Neuron hardware on individual instances. Launch a DLAMI, SSH in, and start training or serving models. Supports Inf1, Inf2, Trn1, and Trn2 instances.

      +++
      *Use for: Development, prototyping, single-node training and inference*

   .. grid-item-card:: **Amazon EKS**
      :link: /deploy/eks/index
      :link-type: doc
      :class-card: sd-border-1

      Kubernetes orchestration with Neuron device plugins, topology-aware scheduling, and :doc:`Dynamic Resource Allocation (DRA) </deploy/eks/dra>` for Trn2. Includes :doc:`Helm chart </deploy/eks/helm-chart>` for one-command infrastructure setup and :doc:`UltraServer support </deploy/eks/ultraserver>`.

      +++
      *Use for: Production inference services, distributed training, auto-scaling*

   .. grid-item-card:: **Amazon ECS**
      :link: /deploy/ecs/index
      :link-type: doc
      :class-card: sd-border-1

      Task-based container orchestration without Kubernetes. Run Neuron DLCs as ECS tasks with :doc:`node problem detection </deploy/ecs/npd>` for automatic health monitoring and recovery.

      +++
      *Use for: Container workloads without Kubernetes, simpler orchestration*

   .. grid-item-card:: **AWS Batch**
      :link: /deploy/batch/index
      :link-type: doc
      :class-card: sd-border-1

      Submit training jobs and let Batch manage compute provisioning, scaling, and cleanup. Build a container, configure a compute environment, and submit jobs.

      +++
      *Use for: Scheduled training, batch processing, variable compute demand*

   .. grid-item-card:: **AWS ParallelCluster**
      :link: /deploy/parallelcluster/index
      :link-type: doc
      :class-card: sd-border-1

      HPC cluster management with Slurm for large-scale distributed training. Set up a head node and Trn1 compute fleet with EFA networking and shared storage.

      +++
      *Use for: Multi-node distributed training, Slurm-based workflows*

   .. grid-item-card:: **Amazon SageMaker**
      :link: /deploy/sagemaker/index
      :link-type: doc
      :class-card: sd-border-1

      Fully managed ML platform. Use JumpStart for model fine-tuning, HyperPod for resilient distributed training, or SageMaker Training for on-demand compute.

      +++
      *Use for: Managed infrastructure, end-to-end ML workflows*

----

Manage Neuron infrastructure
------------------------------

Kubernetes plugins, monitoring, and operational tools for running Neuron workloads in production.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: **Infrastructure components**
      :link: /deploy/infrastructure/index
      :link-type: doc
      :class-card: sd-border-1

      Neuron device plugin, scheduler extension, DRA driver, node problem detector, and monitoring. These components manage device discovery, topology-aware scheduling, and health monitoring in Kubernetes.

   .. grid-item-card:: **Container tutorials**
      :link: /deploy/tutorials/index
      :link-type: doc
      :class-card: sd-border-1

      Step-by-step guides for Docker environment setup, building Neuron containers, configuring OCI hooks, and running inference and training in containers.

   .. grid-item-card:: **Third-party solutions**
      :link: /deploy/third-party/index
      :link-type: doc
      :class-card: sd-border-1

      Partner integrations including Ray for distributed orchestration and Domino for enterprise ML platforms.

   .. grid-item-card:: **FAQ and troubleshooting**
      :link: /deploy/faq
      :link-type: doc
      :class-card: sd-border-1

      Common questions about Neuron containers, device exposure, EFA networking, and Kubernetes scheduling.

----

Release notes
--------------

* :doc:`Container release notes </release-notes/components/containers>`
* :doc:`DLAMI release notes </release-notes/components/dlamis>`

.. toctree::
   :maxdepth: 1
   :hidden:

   Deployment Overview </deploy/overview>
   Pre-configured Environments </deploy/environments/index>
   Use Amazon EC2 </deploy/ec2/index>
   Use Amazon EKS </deploy/eks/index>
   Use Amazon ECS </deploy/ecs/index>
   Use AWS Batch </deploy/batch/index>
   Use AWS ParallelCluster </deploy/parallelcluster/index>
   Use Amazon SageMaker </deploy/sagemaker/index>
   Infrastructure Components </deploy/infrastructure/index>
   Tutorials </deploy/tutorials/index>
   Docker Examples </deploy/docker-examples/index>
   Third-party Solutions </deploy/third-party/index>
   FAQ </deploy/faq>
   Troubleshooting </deploy/troubleshooting>
   Container Release Notes </release-notes/components/containers>
   DLAMI Release Notes </release-notes/components/dlamis>
