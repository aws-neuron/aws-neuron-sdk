.. meta::
   :description: AWS Neuron UltraServer Operator
   :keywords: AWS, Neuron, DRA, Kubernetes, Dynamic Resource Allocation, UltraServer Operator

.. _neuron-ultraserver-operator:

====================================
AWS Neuron UltraServer Operator Beta
====================================

What is the Neuron UltraServer Operator?
----------------------------------------

The Neuron UltraServer Operator is a Kubernetes operator that enables topology-aware provisioning and lifecycle management of Neuron UltraServer workloads on Amazon EKS. 
An UltraServer is a group of Trainium instances interconnected by high-bandwidth NeuronLinks, forming a tightly-coupled multi-node compute unit for distributed training and inference.

The operator works alongside the :ref:`Neuron DRA Driver<neuron-dra>` to automate UltraServer discovery, workload allocation, 
and resource claim generation — eliminating the need for manual init containers, node label matching, and post-launch validation that were required with the device plugin approach.

Why use the UltraServer Operator?
---------------------------------

Without the operator, deploying workloads on UltraServers requires:

* Init containers for pre-launch validation (mode support, topology validation, env var assignment)
* Manual affinity rules to match on UltraServer mode labels
* No guarantee that all pods in a distributed job can be scheduled (sequential pod scheduling)

The UltraServer Operator addresses these issues by:

* **Automatically discovering UltraServer topology** from DRA ResourceSlices
* **Pre-provisioning UltraServers** for workloads before pod scheduling begins
* **Generating ResourceClaimTemplates** so pods are scheduled only on allocated UltraServer nodes


Architecture
------------

The operator consists of two main components:

**Unified Controller (cluster-level)**

* Watches ResourceSlices published by the DRA driver and groups nodes into UltraServer CRDs based on ``ultraserverId`` (capacity block ID)
* Processes ``NeuronUltraServerWorkload`` CRDs created by users
* Allocates UltraServer nodes to workloads and generates ``ResourceClaimTemplate`` objects
* Manages UltraServer lifecycle (allocation, deallocation, failure recovery)

**DRA Driver (per-node)**

* Publishes ResourceSlices with device attributes (instance type, driver version, topology, UltraServer IDs)
* Watches ``NeuronUltraServer`` CRD updates and tags devices with ``workloadId`` in ResourceSlices
* Validates workloadId during ``NodePrepareResources`` before binding devices to pods


Custom Resource Definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**NeuronUltraServer** (system-generated) — Represents a physical UltraServer group:

* Lists member nodes
* Tracks available configurations (1-node, 2-node, 4-node)
* Records link topology status 
* Tracks allocation status per workload
  
**NeuronUltraServerWorkload** (user-created) — Declares a workload's UltraServer requirements:

* ``instanceType``: Instance type of the UltraServer nodes
* ``topologySize``: Number of nodes per UltraServer (1, 2, or 4)
* ``serverCount``: How many UltraServers to allocate
* ``allocationPolicy.failureStrategy``: How to handle node failures (``RestartAll``, ``ReplaceAffected``, or ``NoOp``)
* ``resourceClaimTemplate.name``: Name for the auto-generated RCT

Beta Access
-----------

.. note::
    Neuron UltraServer Operator is in private beta. Contact the Neuron product team to get access, and for installation instructions.


Prerequisites for the preview
-----------------------------

* **Kubernetes version** - Please use K8s control plane 1.34+
* **Instance type** - Trn3pds launched with K8s version 1.34.2+

For instructions on how to setup an EKS cluster, please refer to :ref:`prerequisites<k8s-prerequisite>`.

Operator Deployment Node Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The UltraServer Operator controller should be deployed on an instance type equivalent to or larger than ``m5.24xlarge``.
The operator performs topology discovery and resource allocation across all UltraServer nodes in the cluster, which
requires sufficient CPU and memory for processing ResourceSlices and managing CRD state at scale.

EFA DRA Driver for Multi-UltraServer Workloads
-----------------------------------------------

For workloads that span multiple UltraServers (``serverCount`` > 1), the EFA DRA driver must be installed to enable
high-performance inter-node networking across UltraServer boundaries. EFA (Elastic Fabric Adapter) provides the
low-latency, high-throughput communication required for distributed training across multiple UltraServers.

For installation instructions, see `EFA DRA driver setup <https://docs.aws.amazon.com/eks/latest/userguide/device-management-efa.html#efa-dra-driver>`_.

.. note::
    The EFA DRA driver is only required for multi-UltraServer workloads (``serverCount`` > 1). Single UltraServer
    workloads communicate over NeuronLink and do not require EFA.

Usage
-----

Step 1: Create a NeuronUltraServerWorkload
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define your UltraServer requirements:

.. code-block:: yaml

    apiVersion: neuron.aws.com/v1beta1
    kind: NeuronUltraServerWorkload
    metadata:
        name: my-training-job
    spec:
        instanceType: <trn3pds instance type> # UltraServer instance type
        topologySize: 4                       # 4-node UltraServer
        serverCount: 2                        # Allocate 2 UltraServers (8 nodes total)
        allocationPolicy:
            failureStrategy: NoOp             # Options: RestartAll, ReplaceAffected, NoOp
        resourceClaimTemplate:
            name: my-training-job-rct         # Name for the auto-generated RCT

Apply it:

.. code-block:: bash

    kubectl apply -f ultraserver-workload.yaml

The operator will:

1. Find available UltraServers matching the requested configuration
2. Allocate nodes and update NeuronUltraServer CRDs
3. Create a ``ResourceClaimTemplate`` named ``my-training-job-rct``
4. Update allocated ``ResourceSlices`` with ``workloadId: my-training-job``

Check allocation status:

.. code-block:: bash

    kubectl get neuronultraserverworkload my-training-job -o yaml

.. code-block:: yaml

    status:
        state: Bound
        workloadId: my-training-job-rct
        allocatedUltraservers:
            - us-id-0
            - us-id-1
        totalNodesAllocated: 8
        allocationTime: "2025-01-15T10:30:00Z"

Step 2: Reference the RCT in your workload
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    apiVersion: apps/v1
    kind: Deployment
    metadata:
        name: my-training-workload-deployment
    spec:
        replicas: 8
        selector:
            matchLabels:
                app: ultraserver-training
        template:
            metadata:
            labels:
                app: ultraserver-training
            spec:
            resourceClaims:
            - name: neuron-claim
                resourceClaimTemplateName: my-training-job-rct
            containers:
            - name: training-container
                image: public.ecr.aws/docker/library/alpine:latest
                command: ["sleep", "infinity"]
                resources:
                    claims:
                    - name: neuron-claim

For multi-UltraServer workloads, also create a ``ResourceClaimTemplate`` that allocates all EFA devices on each node:

.. code-block:: yaml

    apiVersion: resource.k8s.io/v1
    kind: ResourceClaimTemplate
    metadata:
        name: efa-claim-template
    spec:
        spec:
            devices:
                requests:
                - name: efa
                  exactly:
                    deviceClassName: efa.networking.k8s.aws
                    allocationMode: All

Then attach the EFA claim alongside the Neuron resource claim:

.. code-block:: yaml

    spec:
        resourceClaims:
        - name: neuron-claim
            resourceClaimTemplateName: my-training-job-rct
        - name: efa-claim
            resourceClaimTemplateName: efa-claim-template
        containers:
        - name: training-container
            resources:
                claims:
                - name: neuron-claim
                - name: efa-claim

Step 3: Cleanup
^^^^^^^^^^^^^^^
Delete the workload to release UltraServer allocations:

.. code-block:: bash

    kubectl delete neuronultraserverworkload my-training-job

This triggers cascade deletion: the operator releases UltraServers, clears device attributes, and removes the RCT and associated pods.

NeuronUltraServerWorkload Spec Reference
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Field
     - Required
     - Description
   * - ``spec.instanceType``
     - Yes
     - Trn3pds instance type for the UltraServer nodes
   * - ``spec.topologySize``
     - Yes
     - Number of nodes per UltraServer (1, 2, or 4)
   * - ``spec.serverCount``
     - Yes
     - Number of UltraServers to allocate
   * - ``spec.allocationPolicy.failureStrategy``
     - No
     - ``RestartAll``, ``ReplaceAffected``, or ``NoOp`` (default: ``NoOp``)
   * - ``spec.resourceClaimTemplate.name``
     - No
     - Name for the generated RCT (defaults to workload name)
   * - ``spec.config.logicalNeuronCore``
     - No
     - Neuron device configuration parameters for this workload, such as logical NeuronCore count

**Failure Strategies**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - Behavior
   * - ``NoOp``
     - Report failure status only. No automatic recovery.
   * - ``RestartAll``
     - Release all UltraServers and reallocate from scratch.
   * - ``ReplaceAffected``
     - Migrate affected pods to healthy UltraServer nodes.

NeuronUltraServerWorkload Status Reference
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``status.state``
     - Current state of the workload allocation. See table below for possible values.
   * - ``status.allocatedUltraServers``
     - List of ``NeuronUltraServer`` resource names allocated to this workload. Each entry corresponds to a NeuronUltraServer CR that references this workload via ``WorkloadRef`` (in the format ``namespace/name``).

**Workload States**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - State
     - Description
   * - ``Pending``
     - Allocation request is pending, waiting for resources.
   * - ``Allocating``
     - Allocation is in progress.
   * - ``Allocated``
     - All requested NeuronUltraServers have been successfully allocated.
   * - ``PartiallyAllocated``
     - Some but not all requested servers were allocated.
   * - ``Failed``
     - Allocation failed and cannot proceed.
   * - ``Deallocating``
     - Resources are being released.

**Viewing NeuronUltraServer Status**

List discovered UltraServers:

.. code-block:: bash

    kubectl get neuronultraservers -n neuron-dra-driver

Inspect a specific UltraServer:

.. code-block:: bash

    kubectl get neuronultraserver <us-id> -n neuron-dra-driver -o yaml

The status shows:

* ``state``: ``Available``, ``PartiallyAllocated``, or ``FullyAllocated``
* ``allocations``: Which workloads are using which nodes
* ``topology``: Mode config for Trn3 UltraServer instances

FAQs
----

Can the UltraServer Operator coexist with the Neuron Device Plugin?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
No. The operator requires the Neuron DRA driver, which cannot coexist with the device plugin on the same node.

What instance types are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Trn3pds UltraServer instance types.

What Kubernetes versions are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Kubernetes 1.34+ (same as the DRA driver requirement).