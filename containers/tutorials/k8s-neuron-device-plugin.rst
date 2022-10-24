.. _k8s-neuron-device-plugin:

Neuron device plugin exposes Neuron cores & devices to kubernetes as a resource. aws.amazon.com/neuroncore, aws.amazon.com/neurondevice, aws.amazon.com/neuron are the resources that the neuron device plugin registers with the kubernetes. aws.amazon.com/neuroncore is used for allocating neuron cores to the container. aws.amazon.com/neurondevice is used for allocating neuron devices to the container. When neurondevice is used all the cores belonging to the device will be allocated to container. aws.amazon.com/neuron also allocates neurondevices and this exists just to be backward compatible with already existing installations. aws.amazon.com/neurondevice is the recommended resource for allocating devices to the container.

* Make sure :ref:`prequisite<k8s-prerequisite>` are satisified
* Download the neuron device plugin yaml file. :download:`k8s-neuron-device-plugin.yml </src/k8/k8s-neuron-device-plugin.yml>`
* Download the neuron device plugin rbac yaml file. This enables permissions for device plugin to update the node and Pod annotations. :download:`k8s-neuron-device-plugin-rbac.yml </src/k8/k8s-neuron-device-plugin-rbac.yml>`
* Apply the Neuron device plugin as a daemonset on the cluster with the following command

    .. code:: bash

        kubectl apply -f k8s-neuron-device-plugin-rbac.yml
        kubectl apply -f k8s-neuron-device-plugin.yml
 
* Verify that neuron device plugin is running

    .. code:: bash

        kubectl get ds neuron-device-plugin-daemonset --namespace kube-system

    Expected result (with 2 nodes in cluster):

    .. code:: bash

        NAME                             DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
        neuron-device-plugin-daemonset   2         2         2       2            2           <none>          27h

* Verify that the node has allocatable neuron cores and devices with the following command

    .. code:: bash

        kubectl get nodes "-o=custom-columns=NAME:.metadata.name,NeuronCore:.status.allocatable.aws\.amazon\.com/neuroncore"    

    Expected result:

    .. code:: bash

        NAME                                          NeuronCore
        ip-192-168-65-41.us-west-2.compute.internal   32
        ip-192-168-87-81.us-west-2.compute.internal   32

    .. code:: bash

        kubectl get nodes "-o=custom-columns=NAME:.metadata.name,NeuronDevice:.status.allocatable.aws\.amazon\.com/neurondevice"    

    Expected result:

    .. code:: bash

        NAME                                          NeuronDevice
        ip-192-168-65-41.us-west-2.compute.internal   16
        ip-192-168-87-81.us-west-2.compute.internal   16