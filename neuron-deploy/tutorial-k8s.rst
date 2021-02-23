Tutorial: Kubernetes environment setup for Neuron
=================================================

Introduction
------------

Customers that use Kubernetes can conveniently integrate Inf1 instances
into their workflows.

A device plugin is provided which advertises Inferentia devices as a
system hardware resource. It is deployed to a cluster as a daemon set
using the provided: :download:`k8s-neuron-device-plugin.yml <../../docs/neuron-container-tools/k8s-neuron-device-plugin.yml>`  This
tutorial will go through deploying the daemon set and running an example
application.

Prerequisite:
^^^^^^^^^^^^^

-  :ref:`tutorial-docker-env-setup-for-neuron`: to
   setup Docker support on worker nodes.
-  Inf1 instances as worker nodes with attached roles allowing:

   -  ECR read access policy to retrieve container images from ECR:
      **arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly**

Steps:
------

This example will walk through running inferences against BERT model
through a K8s service running on Inferentia.

Step 1: Deploy the neuron device plugin:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A device plugin exposes Inferentia to kubernetes as a resource. The
device plugin container is provided through an ECR repo, pointed at in
attached device plugin file.

Run the following command:

.. code:: bash

   kubectl apply -f https://github.com/aws/aws-neuron-sdk/blob/master/docs/neuron-containers-tools/k8s-neuron-device-plugin.yml 

Make sure that you see Neuron device plugin running successfully:

.. code:: bash

   kubectl get ds neuron-device-plugin-daemonset --namespace kube-system

Expected result:

.. code:: bash

   NAME                             DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
   neuron-device-plugin-daemonset   1         1         1       1            1           <none>          17h

You can now require Inferentia devices in a k8s manifest as in the
following example. The number of Inferentia devices can be adjusted
using the *aws.amazon.com/neuron* resource. The runtime expects 128 2-MB
pages per inferentia device, therefore, *hugepages-2Mi* has to be set to
256 \* number of Inferentia devices.

::

           resources:
             limits:
               hugepages-2Mi: 256Mi
               aws.amazon.com/neuron: 1
             requests:
               memory: 1024Mi

Step 2: Optional: Deploy an application requiring Inferentia resource
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`example-deploy-bert-as-k8s-service`
tutorial provides an example how to use k8s with Inferentia.
