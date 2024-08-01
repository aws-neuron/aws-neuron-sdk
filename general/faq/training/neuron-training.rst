.. _neuron-training-faq:

Training with Neuron - FAQ
==========================

.. contents:: Table of contents
   :local:
   :depth: 2

Compute
-------

How do I get started with training my model on Trn1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you select your machine learning framework, you can get started here: :ref:`docs-quick-links`


How do I setup EFA for multi-node training?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For setting up EFA that is needed for multi-node training, please see :ref:`setup-trn1-multi-node-execution`


How do I know if I can train my models with Trainium?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We aim to support a broad set of models and distribution libraries. We continuously add more capabilities and enable new features via Neuron SDK releases and suggest you will follow our public roadmap and join our slack and email lists.

How should I size Trainium NeuronCores vs GPUs?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For simplicity, you should consider each NeuronCore within your instances as an independent deep learning compute engine, the equivalent of a GPU. As point of comparison, a trn1.32xlarge has 32 NeuronCores, and their max performance is 40% higher than of P4d for BF16/FP16/FP8, 2.5X faster for TF32, and 5X faster for FP32. Each NeuronCore is independent and connected to the rest of the NeuronCores within the instance via NeuronLink, and across instances with EFA. Each NeuronCore has also full access to the accelerator memory in the instance, which helps scale large models across NeuronCores using various collective compute ops techniques. 

What are the time to train advantages of Trn1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the answer is largely model dependent, training performance on Trn1 is fast due thanks for multiple system wide optimizations working in concert. Dependent on the data type, you should expect between 1.4-5X higher throughput on Trn1 as compared to the latest GPUs instances (P4d). For distributed workloads, 800Gbps EFA gives customers lower latency, and 2x the throughput as compared to P4d. (a Trn1n 1.6Tb option is coming soon). Each Trainium also has a dedicated collective compute (CC) engine, which enables running the CC ops in parallel to the NeuronCores compute. This enables another 10-15% acceleration of the overall workload. Finally, stochastic rounding enables running at half precision speeds (BF16) while maintaining accuracy at near full precision, this is not only simplifying model development (no need for mixed precision) it also helps the loss function converge faster and reduce memory footprint.

What are some of the training performance results for Trn1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

They are great! please refer to the :ref:`benchmark` page for open-source model performance results. We encourage you to try it for your own models/application.

Can I use CUDA libraries with AWS Trainium?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AWS Trainium and Neuron are plugged into popular frameworks, and is automatically optimizing model deployment on Neuron devices like Inferentia and Trainium. The Neuron SDK automatically optimizes for Trainium without using closed source dependencies like Nvidia CUDA, not requiring any application level code changes to accelerate models. We believe this intentional approach allows developers freedom of choice with their code and models. If you have applications dependencieson CUDA (or other 3rd party closed source artifacts) you will need to strip them out, and from that point the Neuron compiler will take the model as is and optimize it at the hardware level. 


Networking
----------

What’s important to know about the networking in Trn1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trn1 have the fastest EFA in AWS, clocked at 800Gbps they enable more collective communication as compared to other training instances, which is important if your training job spans across multiple servers. You should also expect lower latency as we streamline the communication path between the dedicated collective communication engine on Trainium, and the AWS Nitro EFA NICs.

How does Trainium accelerates collective communication  operations?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trainium introduces a dedicated collective compute engine, that runs in parallel to the compute cores (aka NeuronCores). This improves convergence time of intermediate steps as the communication happens in parallel to the compute. This capability, in addition to the faster and optimized EFA, results in better scalability and faster time to train, as compared to other training instances in AWS.

What does Strong/Weak Scaling mean?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable strong scaling, we optimized Trainium to be efficient at small batch sizes. Compared to GPUs, Trn1 maintains high efficiency even for small batch sizes. This allows you to scale-out to thousands of devices without increasing the global mini-batch size at the same rate, which in turn leads to faster end-to-end training convergence.

In weak scaling setup, we show the optimal throughput with sufficiently large batch size per Trainium. The large batch size is set to leverage the high core utilization so that the overall end-to-end training will be fast. This setup also enables a large global batch size as it scales with the total number of nodes in the cluster.

Usability
---------

What have AWS done to improve usability of Trainium?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stochastic rounding enables running at half precision speeds (BF16) while maintaining accuracy at near full precision. This of course helps the loss function converge faster and reduce memory footprint, but equally important, it is simplifying model development as you can write your model in FP32, and Neuron/Trainium will auto-cast the model to BF16, and execute it with SR enabled. There is no need to loss accuracy with pure BF16 runs, and more importantly no need for experimenting with  mixed precision strategies to find the optimal settings.

Eager debug mode provides a convenient utility to step through the code and evaluate operator correctness as part of your model creation/debug. For more details, please refer to the Neuron documentation

What other AWS services work with Trn1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trn1 via its Neuron SDK supports Amazon ECS, EKS, ParallelCluster, Batch, and Amazon SageMaker. Customers can also choose to run in a Neuron container within their self-managed containers orchestration service (e.g., Kubernetes and Ray).

What tools are available to develop models with Trn1?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When running training, evaluation or inference workloads you can use Neuron 2.x CLI tools such as neuron-ls and neuron-top to get insights into the NeuronCores and NeuronDevices performance and memory utilization, topology and host vCPU performance and memory utilization. In addition, the Neuron Plugin for TensorBoard provides a standard GUI that enables profile and debug of models. TensorBoard views include:

- Model overview: provide a summary of the model and the utilization on the Host and NeuronDevice
- Operators’ view: provide a breakdown of ML framework and HLO operators on both Host and NeuronDevice
- Code trace view: show a timeline of the model execution at the framework and HLO operators level 
- Hardware trace view: show a timeline of the model execution at the level of hardware (Host, NeuronDevice, Data Transfer)
- Topology view: show the NeuronDevices topology within an instance


How will compile time impact my work flow?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We understand compilation is a new step with Trainium, but as long as the overall time to train and cost to train is optimized, the compilation impact on these two metrics is minimized. To further help reduce compilation time impact on usability, Neuron supports a persistent cache, where artifacts that have not changed since the last run can be reused, skipping compilation all together. For developing and experimenting with new models, you can use the eager debug mode, that compiles (and caches) op-by-op, enabling quick evaluation without compiling large models. We are also working on Neuron model analyzer (see Neuron roadmap) that will recommend optimized hyper parameters, skipping full compilation per experiment.
