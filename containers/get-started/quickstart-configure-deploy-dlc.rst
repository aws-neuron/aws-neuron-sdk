.. meta::
   :description: Learn how to configure and deploy a vLLM Deep Learning Container with AWS Neuron SDK on Trainium and Inferentia instances.
   :date_updated: 08/18/2025

.. _dlc_quickstart_configure_deploy:

Quickstart: Configure and deploy a Deep Learning Container (DLC) with AWS Neuron SDK
====================================================================================

This topic guides you through configuring and deploying a vLLM Deep Learning Container using the AWS Neuron SDK. When you complete this tutorial, you will run a vLLM inference server on AWS Trainium and Inferentia instances.

Overview
--------

You will pull a vLLM Docker image, configure it for Neuron devices, and start an inference server. This process lets you deploy large language models on AWS ML accelerators for high-performance inference workloads.

Before you start
----------------

This tutorial assumes that you have experience in the following areas:

* Docker container management
* AWS EC2 instance administration
* Command-line interface operations

Prerequisites
-------------

Before you begin, ensure you have:

* AWS Trainium or Inferentia instance access
* Docker installed on your instance
* SSH access to your instance

Prepare your environment
------------------------

Launch an AWS Trainium or Inferentia instance with sufficient resources for your model requirements.

Step 1: Pull the vLLM Docker image
-----------------------------------

In this step, you will download the vLLM Docker image from AWS ECR.

Pull the vLLM Docker image from AWS ECR:

.. code-block:: bash

   docker pull <image_uri>

Replace ``<image_uri>`` with the specific vLLM image URI from AWS ECR.

Step 2: Start the Docker container
-----------------------------------

In this step, you will run the container with access to Neuron devices.

Run the container interactively with access to Neuron devices:

.. code-block:: bash

   docker run -it \
   --device=/dev/neuron0 \
   --device=/dev/neuron1 \
   --device=/dev/neuron2 \
   --device=/dev/neuron3 \
   --device=/dev/neuron4 \
   --device=/dev/neuron5 \
   --device=/dev/neuron6 \
   --device=/dev/neuron7 \
   --device=/dev/neuron8 \
   --device=/dev/neuron9 \
   --device=/dev/neuron10 \
   --device=/dev/neuron11 \
   --device=/dev/neuron12 \
   --device=/dev/neuron13 \
   --device=/dev/neuron14 \
   --device=/dev/neuron15 \
   --cap-add SYS_ADMIN \
   --cap-add IPC_LOCK \
   -p 8080:8080 \
   --name <server_name> \
   <image_uri> \
   bash

.. note::
   Adjust the number of Neuron devices (``--device=/dev/neuronX``) based on your instance type and requirements.

Step 3: Start the vLLM server
------------------------------

In this step, you will launch the vLLM inference server inside the container.

Inside the container, start the vLLM inference server:

.. code-block:: bash

   VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference' python vllm.entrypoints.openai.api_server \
   --model='TinyLlama/TinyLlama-1.1B-Chat-v1.0' \
   --max-num-seqs=4 \
   --max-model-len=128 \
   --tensor-parallel-size=8 \
   --port=8080 \
   --device 'neuron' \
   --override-neuron-config '{"enable_bucketing":false}'

.. important::
   * Choose the appropriate model for your use case
   * Set ``--tensor-parallel-size`` to be less than or equal to the number of Neuron devices you specified in Step 2
   * Server startup typically takes 5-10 minutes

Step 4: Verify server status
-----------------------------

In this step, you will confirm the server starts successfully.

Wait for the server to fully initialize. You will see output showing available API routes:

.. code-block:: text

   INFO 08-12 00:04:47 [launcher.py:28] Available routes are:
   INFO 08-12 00:04:47 [launcher.py:36] Route: /health, Methods: GET
   INFO 08-12 00:04:47 [launcher.py:36] Route: /v1/chat/completions, Methods: POST
   INFO 08-12 00:04:47 [launcher.py:36] Route: /v1/completions, Methods: POST

All complete! Now, let's confirm everything works.

Confirmation
------------

Test the API to confirm your setup works correctly.

Open a separate terminal and make an API call:

.. code-block:: bash

   curl http://localhost:8080/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
     "messages": [
       {
         "role": "user",
         "content": "What is the capital of Italy?"
       }
     ]
   }'

You should receive a response similar to:

.. code-block:: json

   {
     "id": "chatcmpl-ac7551dd2f2a4be3bd2c1aabffa79b4c",
     "object": "chat.completion",
     "created": 1754958455,
     "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
     "choices": [
       {
         "index": 0,
         "message": {
           "role": "assistant",
           "content": "The capital of Italy is Rome...",
           "tool_calls": []
         },
         "finish_reason": "stop"
       }
     ],
     "usage": {
       "prompt_tokens": 23,
       "total_tokens": 106,
       "completion_tokens": 83
     }
   }

Congratulations! You have successfully deployed a vLLM inference server using AWS Neuron SDK. If you encountered any issues, see the **Common issues** section below.

Available API endpoints
-----------------------

The server provides various endpoints for different use cases:

* **Health Check**: ``GET /health``
* **Chat Completions**: ``POST /v1/chat/completions``
* **Text Completions**: ``POST /v1/completions``
* **Embeddings**: ``POST /v1/embeddings``
* **Models Info**: ``GET /v1/models``
* **API Documentation**: ``GET /docs``

Common issues
-------------

Did you encounter an error while working through this tutorial? Here are common issues and solutions:

- **Server won't start**: Check that you have sufficient Neuron devices allocated
- **Connection refused**: Verify the container is running and port 8080 is properly mapped
- **Slow performance**: Ensure your ``tensor-parallel-size`` matches your available Neuron devices
- **Memory issues**: Consider using a larger instance type or reducing model size

For additional help, refer to the complete vLLM User Guide for NxD Inference documentation.

Clean up
--------

To clean up resources after completing this tutorial:

1. Stop the Docker container:

   .. code-block:: bash

      docker stop <server_name>

2. Remove the container:

   .. code-block:: bash

      docker rm <server_name>

3. Terminate your EC2 instance if no longer needed.

Next steps
----------

Now that you've completed this tutorial, explore these related topics:

* Learn more about vLLM configuration options in the vLLM User Guide for NxD Inference
* Explore model optimization techniques for better performance
* Set up production deployment with load balancing and monitoring

Further reading
---------------

- `vLLM User Guide for NxD Inference <#>`_ - Complete documentation for vLLM on Neuron
- `AWS Neuron SDK Documentation <https://awsdocs-neuron.readthedocs-hosted.com/>`_ - Full Neuron SDK reference