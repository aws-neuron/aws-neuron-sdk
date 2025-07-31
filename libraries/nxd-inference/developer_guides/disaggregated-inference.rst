.. _nxdi-disaggregated-inference:

Disaggregated Inference [BETA]
==============================


Overview
--------

Disaggregated Inference (DI), also known as disaggregated serving, disaggregated prefill, P/D disaggregation,
is an LLM serving architecture that separates the prefill and decode phases of inference onto different hardware resources. 
To achieve this, prefill worker needs to transfer the computed KV cache to the decode worker to resume decoding.
Separating the compute intensive prefill phase from the memory bandwidth intensive 
decode phase can improve the LLM serving experience by

1. Removing prefill interruptions to decode from continuous batching to reduce inter token latency (ITL). These gains can be used to
achieve higher throughput by running with a higher decode batch size while staying under Service Level Objectives (SLO).

2. Adapt to changing traffic patterns while still remaining under application SLOs.

3. Enable independent scaling of resources and parallelism strategies for prefill (compute bound) and decode (memory bound).


.. note::

    This feature is still in beta. Currently only a single decode server and a single prefill server is 
    supported (1P1D). Automatic Prefix Caching is not supported with DI.


Neuron Implementation Details
-----------------------------

Disaggregated Inference is mainly implemented through Neuron's vLLM fork 
https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.25 
and the Neuron Runtime.

There are three main components to a DI workflow.

1. The router. Its job is to orchestrate requests to servers inside the prefill and decode clusters.

2. The prefill cluster. This represents all of the prefill servers ready to run a DI workload.

3. The decode cluster. This represents all of the decode servers ready to run a DI workload.

Below is an example lifespan of a single request through the DI flow.

.. image:: /libraries/nxd-inference/developer_guides/images/di_high_level_architecture.png
    :alt: High Level Disaggregated Inference Architecture

1. A request is sent to the router (1), a component responsible for orchestrating (2) the requests to both
the prefill and decode servers. It receives responses from the prefill and decode servers and 
streams the results back to the user. 

2. The prefill server receives the request from the router (3a) and starts prefilling. After the prefill completes (4),
it updates the status of the request for the decode server by sending information through another ZMQ server.
Then, it listens for a "pull request" from the decode server to initiate the KV cache transfer.
We use Neuron runtime APIs to transfer the KV cache through EFA from Neuron device to Neuron device.
This is a zero copy transfer, meaning that we do not copy the KV cache from a Neuron device to CPU to transfer, 
but rather directly transfer KV cache from Neuron device to Neuron device.
The transfer is also asynchronous. This means that the prefill server can immediately start 
prefilling the next request while the KV cache of the previous request is being transferred. This 
ensures that TTFT is not impacted for other requests while the KV cache for older request is being transferred to decode.

3. The decode server also receives a request from the router at the same time as the prefill server (3b).
It waits until it receives a signal that its corresponding prefill is done from the prefill server by listening
on the ZMQ server. Then, if there is a free spot in the decode batch, the scheduler will schedule the request and send
a "pull request" to the prefill server. This initiates the asynchronous KV cache transfer (red arrow) 
through EFA by calling the Neuron Runtime API. The receive also needs to be asynchronous to ensure
smooth ITL. While the receive is happening other decode requests will still run. As soon as the receive is
finished the scheduler will add the request to the next decode batch (5).


Prefill Decode Interference When Colocating Prefill and Decode
--------------------------------------------------------------

In traditional continuous batching, prefill requests are prioritized over decode requests. Prefills
are run as batch size 1 because they are compute intensive whereas decodes can be run at a higher 
batch size because it is constrained on memory bandwidth not compute. To ensure the highest
throughput, continuous batching schedulers prioritize new prefills if the decode batch is not at max capacity.
As soon as a decode request finishes, another prefill is scheduled to fill the finished request's place. 
However, all other ongoing decodes pause while the new prefill is running because that prefill uses
the same compute resources. This effect is known as prefill stall or prefill contention.

Disaggregated Inference avoids prefill stall because the decode workflow is never interrupted by a prefill as
it receives KV caches asynchronously while decoding. The overall ITL on DI is affected
by the transfer time of the KV cache but this does not scale with batch size. For example, in a continuous
batching workload of batch size 8 each request will on average be interrupted 7 times whereas in DI each 
request is only affected by a single transfer since it happens asynchronously.

Another advantage of DI is its ability to adapt to traffic patterns while maintaining a consistent
ITL. For example, if prefill requests double in length the application can double the amount of available prefill
servers in the prefill cluster to match the new traffic pattern. Continuous batching workloads would suffer because
longer prefill requests increase tail ITL whereas DI workloads continue to deliver a low variance and a predictable customer experience.

Additionally, DI also allows users to tailor their parallelism strategies differently for prefill and decode. 
For example, a model with 32 attention heads may prefer to run two decode servers Data Parallel=2 (DP). 
each with Tensor Parallel=32 (TP) in order to reduce KV replication instead of TP=64. Such replication will get worse if using Group Query Attention (GQA).

DI does not necessarily improve throughput directly but it can help depending on the workload. Continuous
batching is a technique optimized for throughput at the cost of ITL. An application may have an SLO to ensure 
that ITL is under a certain threshold. Because increasing the batch size
increases the amount of prefill stall, and therefore increases ITL, many applications run on smaller than ideal batch sizes 
when using continuous batching. DI can allow an application to run at a higher batch size while still keeping ITL
under the application defined SLO.


Trade-Offs
----------

Because DI runs prefill and decode separately, each part of the inference process needs to operate at an
equal level of efficiency to maximize throughput and hardware resources. For example, if you can process 4 prefill
requests per second and two decode requests per second the application will be stuck processing
two requests per second. It is also important to note that the prefill and decode efficiency can vary based on
the prompt length and the number of tokens for a response respectively. Continuous batching and chunked prefill
do not have this issue as these techniques run prefill and decode on the same hardware.

One technique to remediate this is to run with a dynamic amount of prefill and decode servers. We call this
dynamic xPyD. In the above example, we could run with 1 prefill and 2 decode servers so that our prefill and 
decode efficiency will be balanced. This is being actively worked on and currently only static configurations
of one prefill to one decode (1P1D) are supported.


Example Usage
-------------

Refer to the `offline inference DI example <https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.25/examples/offline_inference/neuron_di.py>`_
for a quick example to get started.

Refer to the :ref:`Disaggregated Inference Tutorial<nxdi-disaggregated-inference-tutorial>` for a detailed usage guide.