.. _k8s-neuron-problem-detector-and-recovery:

Neuron node problem detector and recovery artifact checks the health of Neuron devices on each Kubernetes node. After detecting an unrecoverable Neuron error, it triggers a node replacement. In order to get started with Neuron node problem detector and recovery, make sure that the following requirements are satisfied:

* The Neuron node problem detector and recovery requires Neuron driver 2.15+, and it requires the runtime to be at SDK 2.18 or later.
* Make sure prerequisites are satisfied. This includes prerequisites for getting started with Kubernetes containers and prerequisites for the Neuron node problem detector and recovery.
* Download the Neuron node problem detector and recovery YAML file: :download:`k8s-neuron-problem-detector-and-recovery.yml </src/k8/neuron-problem-detector/k8s-neuron-problem-detector-and-recovery.yml>`.

    .. note::

        This YAML pulls the container image from the upstream repository for node problem detector registry.k8s.io/node-problem-detector.

* Download the Neuron node problem detector and recovery configuration file: :download:`k8s-neuron-problem-detector-and-recovery-config.yml </src/k8/neuron-problem-detector/k8s-neuron-problem-detector-and-recovery-config.yml>`.
* Download the Neuron node problem detector and recovery RBAC YAML file. This enables permissions for the Neuron node problem detector and recovery to update the node condition: :download:`k8s-neuron-problem-detector-and-recovery-rbac.yml </src/k8/neuron-problem-detector/k8s-neuron-problem-detector-and-recovery-rbac.yml>`.
* By default, the Neuron node problem detector and recovery has monitor only mode enabled. To enable the recovery functionality, update the environment variable in the YAML file:

    .. code:: bash

        - name: ENABLE_RECOVERY
          value: "true"

Apply the Neuron node problem detector and recovery as a DaemonSet on the cluster with the following command:

    .. code:: bash

        kubectl create ns neuron-healthcheck-system
        kubectl apply -f k8s-neuron-problem-detector-and-recovery-rbac.yml
        kubectl apply -f k8s-neuron-problem-detector-and-recovery-config.yml
        kubectl apply -f k8s-neuron-problem-detector-and-recovery.yml
 
Verify that the Neuron device plugin is running:

    .. code:: bash

        kubectl get pod -n neuron-healthcheck-system

    Expected result (with 4 nodes in cluster):

    .. code:: bash

        NAME                          READY   STATUS    RESTARTS   AGE
        node-problem-detector-7qcrj   1/1     Running   0          59s
        node-problem-detector-j45t5   1/1     Running   0          59s
        node-problem-detector-mr2cl   1/1     Running   0          59s
        node-problem-detector-vpjtk   1/1     Running   0          59s


When any unrecoverable error occurs, Neuron node problem detector and recovery publishes a metric under the CloudWatch namespace NeuronHealthCheck. It also reflects in NodeCondition and can be seen with kubectl describe node.