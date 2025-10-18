.. post:: June 28, 2024
    :language: en
    :tags: announce-eos-neuron-device, neuron-device

.. _announce-eos-neurondevice:

Announcing end of support for 'neurondevice' resource name in Neuron Device K8s plugin
----------------------------------------------------------------------------------------

:ref:`Neuron release 2.19 <neuron-2.19.0-whatsnew>` will be the last release to include resource name 'neurondevice'. 

Neuron device plugin is a Neuron Software component that gets installed in Kubernetes environment. The resource name 'neurondevice' enables customers to allocate devices to the Neuron K8s container.

In future releases, we will rename resource name 'neurondevice' to 'neuron' to maintain consistency. Customers who are using the resource name 'neurondevice' in their YAML file will need to update to use 'neuron'.

Please see :ref:`k8s-neuron-device-plugin` for more details.
