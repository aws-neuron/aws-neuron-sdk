.. _containers-how-to-ultraserver:

How to schedule MPI jobs to run on Neuron UltraServer on EKS
============================================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Trn2 UltraServers represent a sophisticated computing infrastructure designed to connect multiple Trainium instances
through NeuronLinkV3 (Read more here: :ref:`aws-trn2-arch`). For many advanced and complex models, customers can use UltraServers to greatly reduce training
and inference times compared to previous distributed job setups.

This page explains the two setups needed to properly schedule and run MPI jobs on the Neuron UltraServer on EKS:

* UltraServer init script for the launcher pod
* Affinity configuration for the worker pods

How it works
~~~~~~~~~~~~

The UltraServer init script will:

* Validate the node config and deployment of the MPI job worker pods
* Write environment variables that are required for runtime to each MPI worker pod
* Write a new hostfile to ``/root/ultraserver_init/new_hostfile``

The validation process includes making sure the node config is a valid number (4, 2, or 1), and that the worker pods
are deployed correctly to UltraServer nodes. More about the how to set the node config can been found below.

The environment variables that are being written are:

* NEURON_GLOBAL_TOPOID: The topology ID of the worker pod
* NEURON_GLOBAL_TOPOID0_HOST: The FQDN of the worker pod that's the “leader” (topology ID of 0)
* NEURON_RT_ULTRASERVER_MODE: The mode of the UltraServer node that’s passed to the Neuron runtime
* NEURON_RT_ULTRASERVER_SERVER_ID: The server ID of the UltraServer node that’s passed to the Neuron runtime
* NEURON_RT_ULTRASERVER_NODE_ID: The node ID of the UltraServer node that’s passed to the Neuron runtime

The affinity performs two functions:

* Prevents worker pods from being scheduled together with worker pods from other jobs
* Requires/Encourages worker pods from the same job to be scheduled together

These configurations are needed in order to properly schedule your MPI job worker pods.

The pod anti-affinity prevents scheduling your workload onto UltraServer topologies where worker pods from other jobs
already exist. For example, if you have an UltraServer that already has a 2-node job running on it, the pod
anti-affinity will prevent scheduling a 4-node job on that UltraServer since 2 of the 4 nodes are already occupied.

The pod affinity will make sure that worker pods of the same job are scheduled together in the same UltraServer
topology. For example, if you have an 2 UltraServers with no jobs running on either of them, the pod affinity would
make sure that the worker pods of a 4-node job are all scheduled on the same UltraServer and not split between the two.

Prerequisites
-------------

* An EKS cluster with trn2 UltraServers (:ref:`kubernetes-getting-started`)
* Neuron Device Plugin installed on the cluster with version >= 2.26.26.0 (:ref:`tutorials/k8s-neuron-device-plugin`)
* MPI operator installed on the cluster
* An MPI job spec

Instructions
------------

UltraServer Init Script
~~~~~~~~~~~~~~~~~~~~~~~

Download the UltraServer init script :download:`k8s-ultraserver-init-script.sh </src/k8/k8s-ultraserver-init-script.sh>`

To use the script, either:
- add it to your MPI job Dockerfile and build the image OR
- create a new Dockerfile and build a new image from your MPI job image

Example:

.. code-block:: dockerfile

    FROM 123456789012.dkr.ecr.us-west-2.amazonaws.com/ultraserver:mpijob
    COPY ultraserver-init-script.sh /tmp/
    RUN chmod +x /tmp/ultraserver-init-script.sh
    ENTRYPOINT ["/tmp/ultraserver-init-script.sh"]

Then add the 2 required init containers to the launcher pod.

The first init container should utilize the /etc/mpi/discover_hosts.sh script to ensure that all worker pods are ready
before continuing on to the UltraServer init script.

The second init container should use the image containing ultraserver-init-script.sh. You can specify a value for
NEURON_ULTRASERVER_NODE_CONFIG, which determines what UltraServer node config your MPI job will use, i.e. how many
UltraServer nodes to use. Possible values are 4, 2, and 1, and the default value is 4.

Example:

.. code-block:: yaml

    apiVersion: kubeflow.org/v2beta1
    kind: MPIJob
    metadata:
      name: &job_name <MPI-JOB-NAME>
      namespace: default
    spec:
      mpiReplicaSpecs:
        Launcher:
          replicas: 1
          template:
            spec:
              containers:
              - name: mpitest
                image: 123456789012.dkr.ecr.us-west-2.amazonaws.com/ultraserver:mpijob
              ...
              initContainers:
              - name: wait-hostfilename
                image: 123456789012.dkr.ecr.us-west-2.amazonaws.com/ultraserver:mpijob
                command:
                - bash
                - -cx
                - |
                  if [[ $(cat /etc/mpi/discover_hosts.sh | wc -l) != 1 ]]; then
                    date
                    echo "Ready"
                    cat /etc/mpi/discover_hosts.sh
                  else
                    date
                    echo "not ready ..."
                    sleep 10
                    exit 1
                  fi
                  while read host; do
                    while ! ssh $host echo $host; do
                      date
                      echo "Pod $host is not up ..."
                      sleep 10
                    done
                    date
                    echo "Pod $host is ready"
                  done <<< "$(/etc/mpi/discover_hosts.sh)"
                resources: {}
                volumeMounts:
                - mountPath: /etc/mpi
                  name: mpi-job-config
                - mountPath: /root/.ssh
                  name: ssh-auth
              - name: ultraserver-init-container
                image: 123456789012.dkr.ecr.us-west-2.amazonaws.com/ultraserver:init-container
                env:
                - name: NEURON_ULTRASERVER_NODE_CONFIG
                  value: <"4", "2", OR "1">
                volumeMounts:
                - mountPath: /etc/mpi
                  name: mpi-job-config
                - mountPath: /root/.ssh
                  name: ssh-auth
                - mountPath: /root/ultraserver_init
                  name: ultraserver-init
              ...
              volumes:
              - name: ultraserver-init
                emptyDir: {}

MPI Worker Pod Affinity
~~~~~~~~~~~~~~~~~~~~~~~

Single-node Job
^^^^^^^^^^^^^^^

2-node job

.. code-block:: yaml

    apiVersion: kubeflow.org/v2beta1
    kind: MPIJob
    metadata:
      name: &job_name <MPI-JOB-NAME>
      namespace: default
      ...
    spec:
      mpiReplicaSpecs:
        Launcher:
          ...
        Worker:
          replicas: 2
          template:
            spec:
              nodeSelector:
                node.kubernetes.io/instance-type: trn2u.48xlarge
              affinity:
                podAntiAffinity:
                  requiredDuringSchedulingIgnoredDuringExecution:
                  - labelSelector:
                      matchExpressions:
                      - key: training.kubeflow.org/job-name
                        operator: NotIn
                        values:
                        - *job_name
                      matchLabels:
                        training.kubeflow.org/job-role: worker
                    topologyKey: neuron.amazonaws.com/ultraserver-server-id-2
                podAffinity:
                  requiredDuringSchedulingIgnoredDuringExecution:
                  - labelSelector:
                      matchLabels:
                        training.kubeflow.org/job-role: worker
                        training.kubeflow.org/job-name: *job_name
                    topologyKey: neuron.amazonaws.com/ultraserver-server-id-2
        ...

4-node job

.. code-block:: yaml

    apiVersion: kubeflow.org/v2beta1
    kind: MPIJob
    metadata:
      name: &job_name <MPI-JOB-NAME>
      namespace: default
      ...
    spec:
      mpiReplicaSpecs:
        Launcher:
          ...
        Worker:
          replicas: 4
          template:
            spec:
              nodeSelector:
                node.kubernetes.io/instance-type: trn2u.48xlarge
              affinity:
                podAntiAffinity:
                  requiredDuringSchedulingIgnoredDuringExecution:
                  - labelSelector:
                      matchExpressions:
                      - key: training.kubeflow.org/job-name
                        operator: NotIn
                        values:
                        - *job_name
                      matchLabels:
                        training.kubeflow.org/job-role: worker
                    topologyKey: neuron.amazonaws.com/ultraserver-server-id-4
                podAffinity:
                  requiredDuringSchedulingIgnoredDuringExecution:
                  - labelSelector:
                      matchLabels:
                        training.kubeflow.org/job-role: worker
                        training.kubeflow.org/job-name: *job_name
                    topologyKey: neuron.amazonaws.com/ultraserver-server-id-4
        ...

Multi-node job
^^^^^^^^^^^^^^

.. code-block:: yaml

    apiVersion: kubeflow.org/v2beta1
    kind: MPIJob
    metadata:
      name: &job_name <MPI-JOB-NAME>
      namespace: default
      ...
    spec:
      mpiReplicaSpecs:
        Launcher:
          ...
        Worker:
          replicas: 16
          template:
            spec:
              nodeSelector:
                node.kubernetes.io/instance-type: trn2u.48xlarge
              affinity:
                podAntiAffinity:
                  requiredDuringSchedulingIgnoredDuringExecution:
                  - labelSelector:
                      matchExpressions:
                      - key: training.kubeflow.org/job-name
                        operator: NotIn
                        values:
                        - *job_name
                      matchLabels:
                        training.kubeflow.org/job-role: worker
                    topologyKey: neuron.amazonaws.com/ultraserver-server-id-4
                podAffinity:
                  preferredDuringSchedulingIgnoredDuringExecution:
                  - weight: 100
                    podAffinityTerm:
                      labelSelector:
                        matchLabels:
                          training.kubeflow.org/job-role: worker
                          training.kubeflow.org/job-name: *job_name
                      topologyKey: neuron.amazonaws.com/ultraserver-server-id-4
        ...

To use the affinity configuration, replace <MPI-JOB-NAME> with your MPI job name and add it to your workload yaml spec.

Confirm your work
-----------------

To validate that the init container is working:

.. code-block::

    # Find the worker pods associated with your MPI job
    kubectl get pods

    # Get the logs of the init container
    kubectl logs <LAUNCHER-POD-NAME> -c ultraserver-init-container

You should see logs under the init container.

Example:

.. code-block::

    $ kubectl get pods
    NAME                                       READY   STATUS     RESTARTS   AGE
    demo-launcher-42lh9                        0/1     Init:0/2   0          4s
    demo-worker-0                              1/1     Running    0          4s
    demo-worker-1                              1/1     Running    0          4s
    demo-worker-2                              1/1     Running    0          4s
    demo-worker-3                              1/1     Running    0          4s

    $ kubectl logs demo-launcher-42lh9 -c ultraserver-init-container
    Using 4-node config
    ...

To validate that the affinity configuration is working:

.. code-block::

    # Find the worker pods and the nodes they are scheduled to
    kubectl get pods -o=custom-columns='POD_NAME:metadata.name,NODE_NAME:spec.nodeName'

    # Compare the labels of the nodes to the
    kubectl get nodes \
        -l neuron.amazonaws.com/ultraserver-mode \
        -o=custom-columns='NAME:metadata.name,MODE:metadata.labels.neuron\.amazonaws\.com/ultraserver-mode,ULTRASERVER_SERVER_ID_2:metadata.labels.neuron\.amazonaws\.com/ultraserver-server-id-2,ULTRASERVER_NODE_ID_2:metadata.labels.neuron\.amazonaws\.com/ultraserver-node-id-2,ULTRASERVER_SERVER_ID_4:metadata.labels.neuron\.amazonaws\.com/ultraserver-server-id-4,ULTRASERVER_NODE_ID_4:metadata.labels.neuron\.amazonaws\.com/ultraserver-node-id-4' | awk 'NR==1{print;next}{print | "sort -k3,3 -k4,4"}'

When looking at the nodes used by the worker pods, they should share the same ULTRASERVER_SERVER_ID_2 or
ULTRASERVER_SERVER_ID_4 label based on which config you chose.

Example when choosing a 4-node config:

.. code-block::

    $ kubectl get pods -o=custom-columns='POD_NAME:metadata.name,NODE_NAME:spec.nodeName'
    POD_NAME                                   NODE_NAME
    demo-launcher-42lh9                        ip-172-32-5-227.ap-southeast-4.compute.internal
    demo-worker-0                              ip-172-32-5-227.ap-southeast-4.compute.internal
    demo-worker-1                              ip-172-32-11-17.ap-southeast-4.compute.internal
    demo-worker-2                              ip-172-32-13-57.ap-southeast-4.compute.internal
    demo-worker-3                              ip-172-32-9-4.ap-southeast-4.compute.internal

    $ kubectl get nodes \
        -l neuron.amazonaws.com/ultraserver-mode \
        -o=custom-columns='NAME:metadata.name,MODE:metadata.labels.neuron\.amazonaws\.com/ultraserver-mode,ULTRASERVER_SERVER_ID_2:metadata.labels.neuron\.amazonaws\.com/ultraserver-server-id-2,ULTRASERVER_NODE_ID_2:metadata.labels.neuron\.amazonaws\.com/ultraserver-node-id-2,ULTRASERVER_SERVER_ID_4:metadata.labels.neuron\.amazonaws\.com/ultraserver-server-id-4,ULTRASERVER_NODE_ID_4:metadata.labels.neuron\.amazonaws\.com/ultraserver-node-id-4' | awk 'NR==1{print;next}{print | "sort -k3,3 -k4,4"}'

    NAME                                              MODE    ULTRASERVER_SERVER_ID_2   ULTRASERVER_NODE_ID_2   ULTRASERVER_SERVER_ID_4   ULTRASERVER_NODE_ID_4
    ip-172-32-11-17.ap-southeast-4.compute.internal   1_2_4   u5wy80u0o2saugxy          0                       bog79p1y8tetj5uu          0
    ip-172-32-13-57.ap-southeast-4.compute.internal   1_2_4   u5wy80u0o2saugxy          1                       bog79p1y8tetj5uu          1
    ip-172-32-5-227.ap-southeast-4.compute.internal   1_2_4   ygml2651y0lwdd46          0                       bog79p1y8tetj5uu          2
    ip-172-32-9-4.ap-southeast-4.compute.internal     1_2_4   ygml2651y0lwdd46          1                       bog79p1y8tetj5uu          3

Common issues
-------------

Init script fails to start
~~~~~~~~~~~~~~~~~~~~~~~~~~

If at least one of the worker pods isn't scheduled to a node, the init script will fail to start.

Example:

.. code-block::

    $ kubectl get pods -o=custom-columns='POD_NAME:metadata.name,NODE_NAME:spec.nodeName'
    POD_NAME                                   NODE_NAME
    demo-launcher-96xsl                        ip-172-32-9-4.ap-southeast-4.compute.internal
    demo-worker-0                              <none>
    demo-worker-1                              <none>
    demo-worker-2                              <none>
    demo-worker-3                              <none>

    $ kubectl logs demo-launcher-96xsl -c ultraserver-init-container
    Error from server (BadRequest): container "ultraserver-init-container" in pod "demo-launcher-96xsl" is waiting to start: PodInitializing

Possible solution: Check your pods for affinity/scheduling issues.

.. code-block::

    $ kubectl describe pod demo-worker-0
    Events:
      Type     Reason            Age    From               Message
      ----     ------            ----   ----               -------
      Warning  FailedScheduling  3m13s  default-scheduler  0/4 nodes are available: 4 node(s) didn't match pod affinity rules. preemption: 0/4 nodes are available: 4 Preemption is not helpful for scheduling.

Related Information
-------------------

- :ref:`kubernetes-getting-started` - Information about how to use Neuron on EKS
- :ref:`tutorials/k8s-neuron-device-plugin` - Information about Neuron Device Plugin
- :ref:`aws-trn2-arch` - Information about trn2 UltraServer architecture
- :ref:`general-troubleshooting` - Information about general troubleshooting for Neuron
- `MPI Operator <https://github.com/kubeflow/mpi-operator>`_ - Information about MPI Operator
- `MPI User Guide <https://www.kubeflow.org/docs/components/trainer/legacy-v1/user-guides/mpi/>`_ - Information about MPI jobs
- `Kubernetes Pod Affinity <https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity>`_ - Information about pod affinity rules
- `YAML anchors <https://support.atlassian.com/bitbucket-cloud/docs/yaml-anchors/>`_ - Information about YAML anchors
