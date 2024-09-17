.. post:: September 16, 2024
    :language: en
    :tags: eos-neuron-device, neuron-device-

.. _eos-neurondevice:

'neurondevice' resource name in Neuron Device K8s plugin no longer supported
------------------------------------------------------------------------------

Starting with :ref:`Neuron release 2.20 <neuron-2.20.0-whatsnew>`, Neuron no longer supports resource name 'neurondevice'. 

Neuron device plugin is a Neuron Software component that gets installed in Kubernetes environment. The resource name 'neurondevice' enables customers to allocate devices to the Neuron K8s container.

In this release, we renamed resource name 'neurondevice' to 'neuron' to maintain consistency. Customers who are using the resource name 'neurondevice' in their YAML file need to update to use 'neuron'.

Please see :ref:`k8s-neuron-device-plugin` for more details.
