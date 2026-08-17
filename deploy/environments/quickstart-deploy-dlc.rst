.. meta::
   :description: Learn how to deploy a vLLM server using preconfigured Neuron Deep Learning Container with on Trainium and Inferentia instances.
   :date_updated: 07/21/2026

.. _quickstart_vllm_dlc_deploy:

Quickstart: Configure and deploy a vLLM server using Neuron Deep Learning Container (DLC)
==========================================================================================

This topic guides you through deploying a vLLM server on Trainium and Inferentia instances using a Deep Learning Container preconfigured with AWS Neuron SDK artifacts. When you complete this tutorial, you will be able run a vLLM inference server on AWS Trainium and Inferentia instances.

Overview
--------
In this quickstart, you will pull a vLLM Docker image, configure it for Neuron devices, and start an inference server running vLLM. This process lets you deploy large language models on AWS ML accelerators for high-performance inference workloads.

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
* Docker installed on your instance. You can set up docker environment according to :ref:`tutorial-docker-env-setup`
* SSH access to your instance

.. note::
   This tutorial is compatible with Trn2 and Trn3 instances only, and requires the vLLM 0.21+ DLC.

Prepare your environment
------------------------

Launch an AWS Trainium or Inferentia instance with sufficient resources for your model requirements. We recommend using one of the base DLAMIs to launch your instance - `Neuron Base DLAMI <#>`.

Step 1: Pull the vLLM Docker image
-----------------------------------

In this step, you will download the pre-configured vLLM DLC from AWS ECR. This image comes with the vLLM Neuron plugin already installed.

Get the latest vLLM Docker image from Neuron's ECR public gallery `pytorch-inference-vllm-neuronx <https://gallery.ecr.aws/neuron/pytorch-inference-vllm-neuronx>`_ repository, and then get the latest published image tag and use it in the command below:

.. code-block:: bash

   docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:<image_tag>

For example, replace ``<image_tag>`` with an SDK 2.32.0 released DLC image tag such as ``0.24.0.1.1.0-neuronx-py313-sdk2.32.0-ubuntu24.04``

.. _quickstart_vllm_dlc_step2:

Step 2: Start the Docker container
-----------------------------------

In this step, you will run the container with access to Neuron devices. For this tutorial, we are using a trn2.48xlarge instance.

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
   The trn2.48xlarge instance provides 16 Neuron devices. Adjust the number of Neuron devices (``--device=/dev/neuronX``) based on your instance type and requirements.

Step 3: Start the vLLM server
------------------------------

In this step, you will launch the vLLM inference server inside the container.

Inside the container, start the vLLM inference server:

.. code-block:: bash

   vllm serve openai/gpt-oss-20b \
   --max-model-len 10240 \
   --max-num-batched-tokens 2048 \
   --max-num-seqs 2 \
   --tensor-parallel-size 8 \
   --hf-overrides '{"quantization_config": {}}' \
   --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [2048], "num_seqs_buckets": [2]}}' \
   --port=8080

.. note::
   If EFA (Elastic Fabric Adapter) is not installed on your host instance, the server will fail to start. To work around this, prepend ``NEURON_SKIP_EFA_AFFINITY=1`` to the ``vllm serve`` command.

.. important::
   * Choose the appropriate model for your use case
   * Set ``--tensor-parallel-size`` to be less than or equal to total number of NeuronCores (or TP ranks) available from your devices, accounting for cores per device and logical core configuration
   * Server startup typically takes 5-10 minutes

Step 4: Verify server status
-----------------------------

In this step, you will confirm the server starts successfully.

Wait for the server to fully initialize. You will see output showing the server has started and available API routes:

.. code-block:: text

   INFO 07-21 17:17:07 [api_server.py:617] Starting vLLM server on http://0.0.0.0:8080
   INFO 07-21 17:17:07 [launcher.py:37] Available routes are:
   INFO 07-21 17:17:07 [launcher.py:46] Route: /health, Methods: GET
   INFO 07-21 17:17:07 [launcher.py:46] Route: /v1/models, Methods: GET
   INFO 07-21 17:17:07 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
   INFO 07-21 17:17:07 [launcher.py:46] Route: /v1/completions, Methods: POST
   ...
   INFO:     Application startup complete.

.. note::
   During startup, you may see warning logs similar to the following, which can be safely ignored:

   .. code-block:: text

      No module named 'vllm._version'
        from .version import __version__, __version_tuple__  # isort:skip
      WARNING [__init__.py:25] The vLLM package was not found, so its version could not be inspected. This may cause platform detection to fail.
      INFO [__init__.py:243] Automatically detected platform neuron.
      WARNING [_custom_ops.py:21] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")

All complete! Now, let's confirm everything works.

Step 5: Inference service confirmation
---------------------------------------

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
     "id": "chatcmpl-a59b03830e834b17",
     "object": "chat.completion",
     "created": 1784598578,
     "model": "openai/gpt-oss-20b",
     "choices": [
       {
         "index": 0,
         "message": {
           "role": "assistant",
           "content": "The capital of Italy is **Rome**.",
           "tool_calls": []
         },
         "finish_reason": "stop"
       }
     ],
     "usage": {
       "prompt_tokens": 74,
       "total_tokens": 108,
       "completion_tokens": 34
     }
   }

Congratulations! You have successfully deployed a vLLM inference server using a preconfigured Neuron DLC. If you encountered any issues, see the **Common issues** section below.

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

- `vLLM Neuron Plugin <https://github.com/vllm-project/vllm-neuron>`_ - Source code and configuration details for the vLLM Neuron plugin
- `vLLM User Guide for NxD Inference <#>`_ - Complete documentation for vLLM on Neuron
- `AWS Neuron SDK Documentation <https://awsdocs-neuron.readthedocs-hosted.com/>`_ - Full Neuron SDK reference
