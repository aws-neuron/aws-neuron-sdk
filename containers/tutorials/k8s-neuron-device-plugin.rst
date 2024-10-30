.. _k8s-neuron-device-plugin:

Neuron device plugin exposes Neuron cores & devices to kubernetes as a resource. aws.amazon.com/neuroncore and aws.amazon.com/neuron are the resources that the neuron device plugin registers with the kubernetes. aws.amazon.com/neuroncore is used for allocating neuron cores to the container. aws.amazon.com/neuron is used for allocating neuron devices to the container. When resource name 'neuron' is used, all the cores belonging to the device will be allocated to container.

Deploy Neuron Device Plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Make sure :ref:`prequisite<k8s-prerequisite>` are satisified
* Apply the Neuron device plugin as a daemonset on the cluster with the following command

    .. code:: bash

        helm upgrade --install neuron-helm-chart oci://public.ecr.aws/neuron/neuron-helm-chart \
            --set "npd.enabled=false"

* Verify that neuron device plugin is running

    .. code:: bash

        kubectl get ds neuron-device-plugin -n kube-system

    Expected result (with 2 nodes in cluster):

    .. code:: bash

        NAME                   DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
        neuron-device-plugin   2         2         2       2            2           <none>          18h

* Verify that the node has allocatable neuron cores and devices with the following command

    .. code:: bash

        kubectl get nodes "-o=custom-columns=NAME:.metadata.name,NeuronCore:.status.allocatable.aws\.amazon\.com/neuroncore"

    Expected result:

    .. code:: bash

        NAME                                          NeuronCore
        ip-192-168-65-41.us-west-2.compute.internal   32
        ip-192-168-87-81.us-west-2.compute.internal   32

    .. code:: bash

        kubectl get nodes "-o=custom-columns=NAME:.metadata.name,NeuronDevice:.status.allocatable.aws\.amazon\.com/neuron"  

    Expected result:

    .. code:: bash

        NAME                                          NeuronDevice
        ip-192-168-65-41.us-west-2.compute.internal   16
        ip-192-168-87-81.us-west-2.compute.internal   16
