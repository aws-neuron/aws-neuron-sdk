.. _container-faq:

Neuron Containers FAQ
=====================

.. contents:: Table of Contents
   :local:
   :depth: 1

Where can I find DLC images
---------------------------
* The Inference/Training DLC images can be found `here <https://github.com/aws/deep-learning-containers/blob/master/available_images.md#user-content-neuron-containers>`_.
* In the `DLC release page <https://github.com/aws/deep-learning-containers/releases>`_ do a search for neuron to get the ECR repo location of specific neuron DLC release.


What is OCI Neuron Hook and do we need that
-------------------------------------------
Neuron devices are exposed to the containers using the --device option in the docker run command.
Docker runtime (runc) does not yet support the ALL option to expose all neuron
devices to the container. 

With OCI neuron hook support is added to expose ALL devices to container using an environment variable,
“AWS_NEURON_VISIBLE_DEVICES=ALL". For more details please refer :ref:`oci neuron hook <tutorial-oci-hook>`

What container runtimes are supported
-------------------------------------
Neuron containers have been tested to work with docker, containerd, cri-o runtimes without any changes.
If the oci neuron hook is used then they need to be enabled in the runtime config. For more details please refer :ref:`oci neuron hook <tutorial-oci-hook>`


How to expose Neuron Devices to Container
-----------------------------------------
Neuron Device: Represents the number of Inferentia/Trainium chips in the instance. Refer :ref:`Container Devices <container-devices>` for more details


How to expose Neuron Cores to Container
---------------------------------------
Neuron Core: Represents the number of Neuron Cores in the instance. Each Inferentia 
Chip has 4 Neuron Cores and each Trainium chip has 2 Neuron Cores.
When the devices are exposed to the containers all the cores in the device are available
for use in the container.  Please refer :ref:`nrt-configuration` to see how the environment variables NEURON_RT_VISIBLE_CORES and NEURON_RT_NUM_CORES 
can be used to assign core to containers

Can Neuron Devices be shared by different Containers running in the same Host
-----------------------------------------------------------------------------
Yes, except in Kubernetes environment where the devices cannot be shared

Can Neuron Cores be shared by different Containers running in the same Host
-----------------------------------------------------------------------------
No

When would you use Neuron K8 Scheduler Extension
-------------------------------------------------
The neuron cores/devices that are exposed to the container needs to be contiguous. The kubernetes device plugin
does not guarantee the devices to be contiguous. The K8 Neuron Scheduler Extension takes care of 
assigning contiguous devices to the containers.

How to add EFA devices to the container
---------------------------------------
The EFA devices are exposed to the container using the --device option

::

   --device /dev/infiniband/uverbs0 

In the kubernetes environment the EFA device plugin is used to detect and advertise 
EFA interfaces. 

::
   kubectl apply -f https://raw.githubusercontent.com/aws-samples/aws-efa-eks/main/manifest/efa-k8s-device-plugin.yml

Application can use the resource type vpc.amazonaws.com/efa in a pod request spec

::

   vpc.amazonaws.com/efa: 4



Can distributed training jobs be run without EFA devices in container
---------------------------------------------------------------------
No. For distributed training jobs in Trainium all the EFA inrerfaces in trn1.32xlarge needs to be
exposed to the containers
