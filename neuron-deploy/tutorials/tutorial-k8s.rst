.. _tutorial-k8s-env-setup-for-neuron:

Tutorial: Kubernetes environment setup for Neuron
=================================================

Introduction
------------

Customers that use Kubernetes can conveniently integrate Inf1 instances
into their workflows.

A device plugin is provided which advertises Inferentia devices as a
system hardware resource. It is deployed to a cluster as a daemon set
using the provided: :download:`k8s-neuron-device-plugin.yml </src/k8/k8s-neuron-device-plugin.yml>`
:download:`k8s-neuron-device-plugin-rbac.yml </src/k8/k8s-neuron-device-plugin-rbac.yml>` - configuration to enable
permissions for device plugin to update the node and Pod annotations

This tutorial will go through deploying the daemon set and running an example
application.

Prerequisite:
^^^^^^^^^^^^^

-  Working kubernetes cluster
-  Inf1 instances as worker nodes with attached roles allowing:

   -  ECR read access policy to retrieve container images from ECR:
      **arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly**
   -  Kubernetes **should** use **docker** as container runtime.
   -  :ref:`tutorial-docker-environment-setup-for-neuron-runtime-10`: to install aws-neuron-runtime-base, install oci-add-hooks and setup docker to use oci-neuron OCI runtime
   -  Kubernetes node object has instance-type set to inf1 types. For ex, ``"node.kubernetes.io/instance-type": "inf1.2xlarge"``

Steps:
------

This example will walk through running inferences against RN50 model
through a K8s service running on Inferentia.

Step 1: Deploy the neuron device plugin:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A device plugin exposes Inferentia to kubernetes as a resource. The
device plugin container is provided through an ECR repo, pointed at in
attached device plugin file.

Run the following commands:

.. code:: bash

   kubectl apply -f k8s-neuron-device-plugin-rbac.yml
   kubectl apply -f k8s-neuron-device-plugin.yml

Make sure that you see Neuron device plugin running successfully:

.. code:: bash

   kubectl get ds neuron-device-plugin-daemonset --namespace kube-system

Expected result:

.. code:: bash

   NAME                             DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
   neuron-device-plugin-daemonset   1         1         1       1            1           <none>          17h

You can now require Inferentia devices in a k8s manifest as in the
following example. The number of Inferentia devices can be adjusted
using the *aws.amazon.com/neuron* resource.

::

           resources:
             limits:
               aws.amazon.com/neuron: 1
             requests:
               memory: 1024Mi

Step 2: Optional: Deploy an application requiring Inferentia resource
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`example-deploy-rn50-as-k8s-service`
tutorial provides an example how to use k8s with Inferentia.
