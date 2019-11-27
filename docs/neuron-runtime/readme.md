# Neuron runtime


The AWS Neuron runtime provides developers flexibility to deploy their inference workloads, and optimize for high-throughput and low-latency, to meet their specific application requirements. 

Neuron runtime takes compiled models, also reffered to as Neuron Executable File Format (NEFF), and loads it to the Inferentia chips to execute inference requests.  

The Neuron runtime is prebuilt is the AWS DLAMI, but developers can also install it in their own environments, which can be custom AMIs or containers. 

Neuron runtime provides developers with the ability to control where is their model deployed to, and how the inferences are executed in the system. For example, using runtime commands developers can assign different models to separate NeuronCore Groups in a flexible and scalable way. This allows to run the same or multiple models in parallel. With NeuronCore Groups developers can maximize the hardware utilization by controlling the NeuronCore compute resources allocated to each NeuronCore Group to ensure it fits their specific application requirements.
