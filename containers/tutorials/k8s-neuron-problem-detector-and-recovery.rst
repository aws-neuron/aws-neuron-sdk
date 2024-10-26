.. _k8s-neuron-problem-detector-and-recovery:

Neuron node problem detector and recovery artifact checks the health of Neuron devices on each Kubernetes node. After detecting an unrecoverable Neuron error, it triggers a node replacement. In order to get started with Neuron node problem detector and recovery, make sure that the following requirements are satisfied:

* The Neuron node problem detector and recovery requires Neuron driver 2.15+, and it requires the runtime to be at SDK 2.18 or later.
* Make sure prerequisites are satisfied. This includes prerequisites for getting started with Kubernetes containers and prerequisites for the Neuron node problem detector and recovery.
* Install the Neuron node problem detector and recovery as a DaemonSet on the cluster with the following command:

    .. note::

        The installation pulls the container image from the upstream repository for node problem detector registry.k8s.io/node-problem-detector.

    .. code:: bash

        helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart

* By default, the Neuron node problem detector and recovery has monitor only mode enabled. To enable the recovery functionality:

    .. code:: bash

        helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
            --set "npd.nodeRecovery.enabled=true"

* Verify that the Neuron device plugin is running:

    .. code:: bash

        kubectl get pod -n neuron-healthcheck-system

    Expected result (with 4 nodes in cluster):

    .. code:: bash

        NAME                          READY   STATUS    RESTARTS   AGE
        node-problem-detector-7qcrj   1/1     Running   0          59s
        node-problem-detector-j45t5   1/1     Running   0          59s
        node-problem-detector-mr2cl   1/1     Running   0          59s
        node-problem-detector-vpjtk   1/1     Running   0          59s


* When any unrecoverable error occurs, Neuron node problem detector and recovery publishes a metric under the CloudWatch namespace NeuronHealthCheck. It also reflects in NodeCondition and can be seen with kubectl describe node.
