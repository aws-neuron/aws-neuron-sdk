.. _neuronperf_faq:

NeuronPerf FAQ
==============

- When should I use NeuronPerf?
	- When you want to measure the highest achievable performance for your model with Neuron.

- When should I **not** use NeuronPerf?
	- When measuring end-to-end performance that includes your network serving stack. Instead, your should compare your e2e numbers to those obtained by NeuronPerf to optimize your serving overhead.

- Is NeuronPerf Open Source?
	- Yes. You can :download:`download the source here</src/neuronperf/neuronperf.tar.gz>`.

- What is the secret to obtaining the best numbers?
	- There is no secret sauce. NeuronPerf follows best practices.

- What are the "best practices" that NeuronPerf uses?
	- These vary slightly by framework and how your model was compiled
	- For a model compiled for a single NeuronCore (DataParallel):
		- To maximize throughput, for ``N`` models, use ``2 * N`` worker threads
		- To minimize latency, use 1 worker thread per model
	- Use a new Python process for each model to avoid GIL contention
	- Ensure you benchmark long enough for your numbers to stabilize
	- Ignore outliers at the start and end of inference benchmarking

