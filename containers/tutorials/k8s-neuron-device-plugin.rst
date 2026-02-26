The Neuron Device Plugin is a Kubernetes device plugin that exposes Neuron hardware resources to the cluster's scheduler. It discovers available Neuron devices on each node, advertises them as allocatable resources, and manages their lifecycle. When Pods request Neuron resources, the device plugin handles the allocation and ensures exclusive access to the assigned devices. This integration enables Kubernetes to treat Neuron accelerators as first-class schedulable resources, similar to GPUs or other specialized hardware.

The device plugin registers two resource types with Kubernetes:

* ``aws.amazon.com/neuroncore`` - Used for allocating individual Neuron cores to containers
* ``aws.amazon.com/neuron`` - Used for allocating entire Neuron devices to containers (all cores belonging to the device)

**Deploy Neuron Device Plugin**

**Prerequisites**

Ensure that all :ref:`prerequisites<k8s-prerequisite>` are satisfied before proceeding.

**Installation**

Apply the Neuron Device Plugin as a DaemonSet on the cluster:

.. code:: bash

    helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
        --set "npd.enabled=false"

**Verify Installation**

Verify that the Neuron Device Plugin is running:

.. code:: bash

    kubectl get ds neuron-device-plugin -n kube-system

Expected output (example with 2 nodes in cluster):

.. code:: bash

    NAME                   DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
    neuron-device-plugin   2         2         2       2            2           <none>          18h

**Verify Allocatable Resources**

Verify that nodes have allocatable Neuron cores:

.. code:: bash

    kubectl get nodes "-o=custom-columns=NAME:.metadata.name,NeuronCore:.status.allocatable.aws\.amazon\.com/neuroncore"

Expected output:

.. code:: bash

    NAME                                          NeuronCore
    ip-192-168-65-41.us-west-2.compute.internal   32
    ip-192-168-87-81.us-west-2.compute.internal   32

Verify that nodes have allocatable Neuron devices:

.. code:: bash

    kubectl get nodes "-o=custom-columns=NAME:.metadata.name,NeuronDevice:.status.allocatable.aws\.amazon\.com/neuron"

Expected output:

.. code:: bash

    NAME                                          NeuronDevice
    ip-192-168-65-41.us-west-2.compute.internal   16
    ip-192-168-87-81.us-west-2.compute.internal   16
