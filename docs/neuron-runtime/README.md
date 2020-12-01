</br>
</br>

Please view our documentation at **[https://awsdocs-neuron.readthedocs-hosted.com/](https://awsdocs-neuron.readthedocs-hosted.com/)** 

**Please note this file will be deprecated.**

</br>
</br>



# Neuron Runtime

Neuron Runtime is a userspace application that provides developers flexibility to deploy their inference applications, and optimize for high-throughput and low-latency to meet their specific application needs. 
Neuron runtime takes compiled models, also referred to as Neuron Executable File Format (NEFF), and loads them to the Inferentia chips to execute inference requests.  


Neuron runtime provides the ability to control where a model is deployed to, and how the inferences are executed in the system. For example, using runtime commands developers can assign different models to separate NeuronCore Groups in a flexible and scalable way. This allows to run the same or multiple models to maximize the hardware utilization to ensure it fits their specific application requirements.

To get started, read the Neuron runtime [Getting started](./nrt_start.md) guide. If your application is container-based, learn how to integrate it with containers by referring to our [configuring Neuron containers](../../docs/neuron-container-tools/README.md) tutorial. The Neuron runtime is prebuilt into the AWS DLAMI, but developers can also install it in their own environments, which can be custom AMIs or containers. 

Looking for support?  Please checkout our [Troubleshooting Neuron Runtime](./nrt-troubleshoot.md) doc or contact us directly by filing an [issue](https://github.com/aws/aws-neuron-sdk/issues). 
