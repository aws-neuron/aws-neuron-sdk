.. _neuron-k8-scheduler-ext:

Neuron Kubernetes Scheduler Extension
=====================================

This document describes how the Neuron K8 scheduler extension works and
how to use it in your cluster. The scheduler is required for scheduling
pods that require more than one Neuron device resource. Please use this
scheduler when working with inf1.6xlarge and inf1.24xlarge.

The k8-neuron-scheduler extends the default scheduler in these two ways:

1. Filter out nodes with non-contiguous device ids.
2. Enforces allocation of contiguous device ids for the PODs requiring
   it.

Flow Diagram
------------

::




                                                                           +------------------------+
                                                                           | POD Manifest           |
                                                                           | with Request           |
                                                                           | aws.amazon.com/neuron:2|
                                                                           |                        |
                                                                           |                        |
                                                       2                   +-------------+----------+
                                            +--------------------------------+           |
                                            |                                |           |
                                            |                                |           | 3
             +------------------------------+-----+                          |           |
             |           Kubelet in INF1 Node     |                          |           |
             |                                    +<-----------+             |           |
             +-----+---------------------+--------+            |       +-----v-----------v--------------+
                   |                     ^                     |       |          Kube-Scheduler        |
                   |                     |                     |       |                                |
                   |                     |                     |       +--^------+---------------+------+
                 9 |                  1  |                     |          |      |               |
                   |                     |                    8|         5|      |4              |
                   |                     |                     |          |      |               |
                   |                     |                     |          |      |               |6
                   v                     |                     |          |      |               |
             +-----+---------------------+--------+            |       +--+------v---------------v------+
             |    neuron-device-plugin            |            +-------+       neuron|scheduler|ext     |
             |    in INF1 node                    |                    +---------------------+----------+
             +----+----------------------+--------+                                          |
                  |                      |                                                   |7
                  |                      |10                                                 |
                  |                      |                                                   v
                11|                      |                                         +---------+-------+
                  |                      |                                         |POD Manifest:    |
                  |                      |                                         |Annotation:      |
                  |                      |                                         |NEURON_DEVS: 2,3 |
                  v                      +---------------------------------------->+                 |
   ENV: AWS_NEURON_VISIBLE_DEVICES: 2,3                                            |                 |
                                                                                   |                 |

   1. neuron-device-plugin returns the list of Neuron devices to kublet
   2. Kubelet advertises the Device list to K8s API server (in turn to kube-scheduler)
   3. POD Request for neuron devices [Kube-Scheduler picks up the POD creation request]
   4. kube-scheduler calls the neuron-scheduler-extn filter function with list of nodes and POD Specification
   5. neuron-scheduler-extn scans through the nodes and filters out nodes with non
   contiguous devices and returns the nodes that are capable of supporing the given POD specification
   6. kube-scheduler calls the neuron-scheduler-extn bind function with pod and node
   7. neuron-scheduler-extn updates the POD annotation with allocated neuron device Ids (contiguous)
   8. neuron-scheduler-extn sends the bind request to kubelet of the selected node
   9. Kubelet calls the Alloc function of the neuron-device-plugin
   10. neuron-device-plugin queries the POD Annotation for allocated device Ids
   11. neuron-device-plugin exports the visisble devices to container runtime

Neuron Components
-----------------

1. k8s-neuron-sheduler - scheduler extension that handles filter and
   bind request. ECR:
   790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-scheduler:latest. It is deployed to a cluster using the provided: :download:`k8s-neuron-scheduler.yml </src/k8/k8s-neuron-scheduler.yml>`
2. :download:`k8s-neuron-scheduler-configmap.yml </src/k8/k8s-neuron-scheduler-configmap.yml>` - ConfigMap to register scheduler
   extension with Kube-scheduler.
3. k8s-neuron-device-plugin - manages neuron devices. ECR:
   790709498068.dkr.ecr.us-east-1.amazonaws.com/neuron-device-plugin:latest.It is deployed to a cluster using the provided: :download:`k8s-neuron-device-plugin.yml </src/k8/k8s-neuron-device-plugin.yml>`
4. :download:`k8s-neuron-device-plugin-rbac.yml </src/k8/k8s-neuron-device-plugin-rbac.yml>` - configuration to enable
   permissions for device plugin to update the node and Pod annotations

Installation
------------

For EKS, please follow the EKS documentation. If you are using Kops or
similar follow these steps:

1. Enable the kube-scheduler with option to use configMap for scheduler
   policy. In your cluster.yml Please update the spec section with the
   following
   ::

      spec:
        kubeScheduler:
        usePolicyConfigMap: true

2. Launch the cluster
   ::

      kops create -f cluster.yml
      kops create secret --name neuron-test-1.k8s.local sshpublickey admin -i ~/.ssh/id_rsa.pub
      kops update cluster --name neuron-test-1.k8s.local --yes

3. Apply the k8s-neuron-scheduler-configmap.yml [Registers
   neuron-scheduler-extension with kube-scheduler]
   ::

      kubectl apply -f k8s-neuron-scheduler-configmap.yml

4. Launch the neuron-scheduler-extension
   ::

      kubectl apply -f k8s-neuron-scheduler.yml

5. Apply k8s-neuron-device-plugin-rbac.yml
   ::

      kubectl apply -f k8s-neuron-device-plugin-rbac.yml

6. Apply the k8s-neuron-device-plugin.yml
   ::

      kubectl apply -f k8s-neuron-device-plugin.yml

Sample logs:
^^^^^^^^^^^^

::

   NAMESPACE     NAME                                                                  READY   STATUS    RESTARTS   AGE
   kube-system   dns-controller-865fd96754-s5x2p                                       1/1     Running   0          12h
   kube-system   etcd-manager-events-ip-172-20-92-213.us-west-2.compute.internal       1/1     Running   0          12h
   kube-system   etcd-manager-main-ip-172-20-92-213.us-west-2.compute.internal         1/1     Running   0          12h
   kube-system   k8s-neuron-scheduler-546bb6b45-k4x6s                                  1/1     Running   0          11h
   kube-system   kops-controller-h7t4s                                                 1/1     Running   0          12h
   kube-system   kube-apiserver-ip-172-20-92-213.us-west-2.compute.internal            1/1     Running   1          12h
   kube-system   kube-controller-manager-ip-172-20-92-213.us-west-2.compute.internal   1/1     Running   0          12h
   kube-system   kube-dns-autoscaler-594dcb44b5-bkgjl                                  1/1     Running   0          12h
   kube-system   kube-dns-b84c667f4-5qv86                                              3/3     Running   0          12h
   kube-system   kube-dns-b84c667f4-8x75m                                              3/3     Running   0          11h
   kube-system   kube-proxy-ip-172-20-75-104.us-west-2.compute.internal                1/1     Running   0          11h
   kube-system   kube-proxy-ip-172-20-92-213.us-west-2.compute.internal                1/1     Running   0          12h
   kube-system   kube-proxy-ip-172-20-95-42.us-west-2.compute.internal                 1/1     Running   0          11h
   kube-system   kube-scheduler-ip-172-20-92-213.us-west-2.compute.internal            1/1     Running   8          12h
   kube-system   neuron-device-plugin-daemonset-75llq                                  1/1     Running   0          11h
   kube-system   neuron-device-plugin-daemonset-9wfnl                                  1/1     Running   0          11h
