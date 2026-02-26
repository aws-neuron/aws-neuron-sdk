The Neuron Scheduler Extension is a Kubernetes scheduler plugin that provides intelligent, topology-aware scheduling for Neuron workloads. While the device plugin handles basic resource allocation, the scheduler extension optimizes Pod placement by considering Neuron core topology, NeuronCore-to-NeuronCore connectivity, and workload requirements. It ensures efficient utilization of Neuron devices by placing Pods on nodes where the requested Neuron cores are optimally configured. This component is optional and primarily beneficial for workloads that require specific subsets of Neuron devices or cores rather than consuming all available resources on a node.

The scheduler extension is required for scheduling Pods that request more than one Neuron core or device resource. It finds sets of directly connected devices with minimal communication latency when scheduling containers, ensuring optimal performance for multi-device workloads.

For a graphical depiction of how the Neuron Scheduler Extension works, see :ref:`k8s-neuron-scheduler-flow`.

**Device Allocation by Instance Type**

The Neuron Scheduler Extension applies topology-aware scheduling rules based on instance type to ensure consistent and high performance regardless of which cores and devices are assigned to containers.

**Inf1 and Inf2 Instances (Ring Topology)**

Devices are connected through a ring topology with no restrictions on the number of devices requested (as long as it is fewer than the total devices on a node). When N devices are requested, the scheduler finds a node where N contiguous devices are available to minimize communication latency. It will never allocate non-contiguous devices to the same container.

For example, when a container requests 3 Neuron devices, the scheduler might assign devices 0, 1, 2 if available, but never devices 0, 2, 4 because those devices are not directly connected.

The figure below shows examples of device sets on an Inf2.48xlarge node that could be assigned to a container requesting 2 devices:

|eks-inf2-device-set|

**Trn1.32xlarge and Trn1n.32xlarge Instances (2D Torus Topology)**

Devices are connected via a 2D torus topology. The scheduler enforces that containers request 1, 4, 8, or all 16 devices. If your container requires a different number of devices (such as 2 or 5), we recommend using an Inf2 instance instead to benefit from more flexible topology support.

If you request an invalid number of devices (such as 7), your Pod will not be scheduled and you will receive a warning:

``Instance type trn1.32xlarge does not support requests for device: 7. Please request a different number of devices.``

When requesting 4 devices, your container will be allocated one of the following device sets if available:

|eks-trn1-device-set4|

When requesting 8 devices, your container will be allocated one of the following device sets if available:

|eks-trn1-device-set8|

.. note::

    For all instance types, requesting one or all Neuron cores or devices is always valid.

**Deploy Neuron Scheduler Extension**

.. tab-set::

   .. tab-item:: Multiple Scheduler Approach

      .. include:: /containers/tutorials/k8s-multiple-scheduler.rst

   .. tab-item:: Default Scheduler Approach

      .. include:: /containers/tutorials/k8s-default-scheduler.rst


.. |eks-inf2-device-set| image:: /images/eks-inf2-device-set.png
.. |eks-trn1-device-set4| image:: /images/eks-trn1-device-set4.png
.. |eks-trn1-device-set8| image:: /images/eks-trn1-device-set8.png
