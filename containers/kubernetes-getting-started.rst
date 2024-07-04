Containers - Kubernetes - Getting Started
=========================================

The Neuron device plugin is a DaemonSet run on all Inferentia and Trainium nodes that enables the containers in your Kubernetes cluster to request and use Neuron cores or devices.
The Neuron scheduler extension is required for containers in your Kubernetes cluster that request multiple Neuron resources. 
It helps find optimal sets of Neuron resources to minimize inter-resource communication costs. 
Below are directions for installing and using the Neuron device plugin and scheduler extension.

    
.. include:: /containers/kubernetes-getting-started.txt