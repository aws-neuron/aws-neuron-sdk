.. _neuron_scheduler:

The Neuron scheduler extension is required for scheduling pods that require more than one Neuron core or device resource.
For a graphical depiction of how the Neuron scheduler extension works, see :ref:`k8s-neuron-scheduler-flow`.
The Neuron scheduler extension finds sets of directly connected devices with minimal communication latency when scheduling containers.
On Inf1 and Inf2 instance types where Neuron devices are connected through a ring topology, the scheduler finds sets of contiguous devices. For example, for a container requesting 3 Neuron devices
the scheduler might assign Neuron devices 0,1,2 to the container if they are available but never devices 0,2,4 because those devices are not directly connected.
On Trn1.32xlarge and Trn1n.32xlarge instance types where devices are connected through a 2D torus topology, the Neuron scheduler enforces additional constraints that containers request 1, 4, 8, or all 16 devices.
If your container requires a different number of devices, such as 2 or 5, we recommend that you use an Inf2 instance instead of Trn1 to benefit from more advanced topology.

Container Device Allocation On Different Instance Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Neuron scheduler extension applies different rules when finding devices to allocate to a container on Inf1 and Inf2 instances than on Trn1.
These rules ensure that when users request a specific number of resources, Neuron delivers *consistent* and *high* performance regardless of which
cores and devices are assigned to the container.

On Inf1 and Inf2 Neuron devices are connected through a ring topology.
There are no restrictions on the number of devices requested as long as it is fewer than the number of devices on a node.
When the user requests N devices, the scheduler finds a node where N contiguous devices are available. It will never allocate
non-contiguous devices to the same container. The figure below shows examples of device sets on an Inf2.48xlarge node which
could be assigned to a container given a request for 2 devices.

|eks-inf2-device-set|

Devices on Trn1.32xlarge and Trn1n.32xlarge nodes are connected via a 2D torus topology. On Trn1 nodes
containers can request 1, 4, 8, or all 16 devices.  In the case you request an invalid number of devices, such as 7,
your pod will not be scheduled and you will receive a warning:

``Instance type trn1.32xlarge does not support requests for device: 7. Please request a different number of devices.``

When requesting 4 devices, your container will be allocated one of the following sets of devices if they are available.

|eks-trn1-device-set4|

When requesting 8 devices, your container will be allocated one of the following sets of devices if they are available.

|eks-trn1-device-set8|

For all instance types, requesting one or all Neuron cores or devices is valid.


Deploy Neuron Scheduler Extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

   .. tab-item:: Multiple Scheduler Approach

      .. include:: /containers/tutorials/k8s-multiple-scheduler.rst

   .. tab-item:: Default Scheduler Approach

      .. include:: /containers/tutorials/k8s-default-scheduler.rst


.. |eks-inf2-device-set| image:: /images/eks-inf2-device-set.png
.. |eks-trn1-device-set4| image:: /images/eks-trn1-device-set4.png
.. |eks-trn1-device-set8| image:: /images/eks-trn1-device-set8.png
