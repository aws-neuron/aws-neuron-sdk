.. meta::
   :description: Learn how to run PyTorch inference using preconfigured Neuron Deep Learning Container with Llama-2-7b on Trainium instances.
   :date_updated: 02/17/2026

.. _quickstart_pytorch_inference_dlc:

Quickstart: Run PyTorch inference using Neuron Deep Learning Container (DLC)
=============================================================================

This topic guides you through running PyTorch inference on Trainium instances using a Deep Learning Container preconfigured with AWS Neuron SDK artifacts. When you complete this tutorial, you will be able to run inference with the Llama-2-7b model on AWS Trainium instances.

Overview
--------
In this quickstart, you will pull a PyTorch inference Docker image, download the Llama-2-7b model from S3, and run an inference demo that compiles, validates, and benchmarks the model. This process lets you deploy large language models on AWS ML accelerators for high-performance inference workloads.

Before you start
----------------

This tutorial assumes that you have experience in the following areas:

* Docker container management
* AWS EC2 instance administration
* Command-line interface operations
* AWS S3 operations

Prerequisites
-------------

Before you begin, ensure you have:

* AWS Trainium instance access (trn2.48xlarge recommended)
* Docker installed on your instance. You can set up docker environment according to :ref:`tutorial-docker-env-setup`
* SSH access to your instance
* AWS credentials configured with access to the model S3 bucket

Prepare your environment
------------------------

Launch an AWS Trainium instance with sufficient resources for your model requirements. We recommend using one of the base DLAMIs to launch your instance - `Neuron Base DLAMI <#>`.

Step 1: Pull the PyTorch inference Docker image
------------------------------------------------

In this step, you will download the PyTorch inference Docker image from AWS ECR.

Get the latest PyTorch inference Docker image from Neuron's ECR public gallery `pytorch-inference-neuronx <https://gallery.ecr.aws/neuron/pytorch-inference-neuronx>`_ repository, and then get the latest published image tag and use it in the command below:

.. code-block:: bash

   docker pull public.ecr.aws/neuron/pytorch-inference-neuronx:<image_tag>

For example, replace ``<image_tag>`` with an SDK 2.28.0 released DLC image tag such as ``2.9.0-neuronx-py312-sdk2.28.0-ubuntu24.04``

Step 2: Download the Llama-2-7b model
--------------------------------------

In this step, you will download the Llama-2-7b model from HuggingFace to an S3 bucket, then copy it to your instance.

First, download the model from HuggingFace and upload to your S3 bucket:

.. code-block:: bash

   # Install HuggingFace CLI if not already installed
   pip install huggingface-hub

   # Login to HuggingFace (you'll need to accept the Llama-2 license first)
   hf auth login

   # Download the model
   hf download meta-llama/Llama-2-7b --local-dir ./Llama-2-7b

   # Upload to your S3 bucket
   aws s3 cp --recursive ./Llama-2-7b s3://your-bucket-name/models/Llama-2-7b/

Then, on your Trainium instance, download the model from S3:

.. note::
   Change ``/home/ec2-user`` to ``/home/ubuntu`` if you're using an Ubuntu AMI.

.. code-block:: bash

   # Create directory for the model
   mkdir -p /home/ec2-user/model_hf/Llama-2-7b

   # Download from S3
   aws s3 cp --recursive s3://your-bucket-name/models/Llama-2-7b/ /home/ec2-user/model_hf/Llama-2-7b/

   # Verify the model downloaded successfully
   ls /home/ec2-user/model_hf/Llama-2-7b/config.json

.. note::
   You must accept the Llama-2 license on HuggingFace before you can download the model. Visit https://huggingface.co/meta-llama/Llama-2-7b to request access.

Step 3: Start the Docker container
-----------------------------------

In this step, you will run the container with access to Neuron devices and mount the model directory. For this tutorial, we are using a trn2.48xlarge instance.

Run the container interactively with access to all Neuron devices:

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
   -v /home/ec2-user/model_hf/Llama-2-7b:/root/model_hf/Llama-2-7b \
   --cap-add SYS_ADMIN \
   --cap-add IPC_LOCK \
   --name pytorch-inference-demo \
   public.ecr.aws/neuron/pytorch-inference-neuronx:<image_tag> \
   bash

.. note::
   The trn2.48xlarge instance provides 12 Neuron devices. Adjust the number of Neuron devices (``--device=/dev/neuronX``) based on your instance type and requirements.

Step 4: Run the inference demo
-------------------------------

In this step, you will run the inference demo script that compiles the model, checks accuracy, and benchmarks performance.

Inside the container, run the inference demo:

.. code-block:: bash

   inference_demo \
   --model-type llama \
   --task-type causal-lm \
   run \
   --model-path /root/model_hf/Llama-2-7b/ \
   --compiled-model-path /root/traced_model/Llama-2-7b-demo/ \
   --torch-dtype bfloat16 \
   --tp-degree 96 \
   --batch-size 2 \
   --max-context-length 32 \
   --seq-len 64 \
   --on-device-sampling \
   --enable-bucketing \
   --top-k 1 \
   --do-sample \
   --pad-token-id 2 \
   --prompt 'I believe the meaning of life is' \
   --prompt 'The color of the sky is' \
   --check-accuracy-mode token-matching \
   --benchmark

.. important::
   * The inference demo takes approximately 20 minutes to complete on a trn2.48xlarge instance
   * The script will compile the model, validate accuracy, and run benchmarks
   * Set ``--tp-degree`` to match the number of NeuronCores you want to use (96 for trn2.48xlarge)

Step 5: Verify the results
---------------------------

In this step, you will confirm the inference demo completed successfully and review the benchmark results.

Wait for the demo to complete. You will see output showing benchmark results:

.. code-block:: text

   Benchmark completed and its result is as following
   {
     "e2e_model": {
       "latency_ms_p50": 8539.34,
       "latency_ms_p90": 8627.43,
       "latency_ms_p95": 8646.97,
       "latency_ms_p99": 8652.62,
       "latency_ms_p100": 8654.03,
       "latency_ms_avg": 8533.13,
       "throughput": 480.01
     },
     "context_encoding_model": {
       "latency_ms_p50": 132.42,
       "latency_ms_p90": 133.47,
       "latency_ms_p95": 133.59,
       "latency_ms_p99": 133.81,
       "latency_ms_p100": 133.86,
       "latency_ms_avg": 132.52,
       "throughput": 30908.75
     },
     "token_generation_model": {
       "latency_ms_p50": 7.84,
       "latency_ms_p90": 8.39,
       "latency_ms_p95": 8.47,
       "latency_ms_p99": 8.63,
       "latency_ms_p100": 28.96,
       "latency_ms_avg": 7.87,
       "throughput": 520434.73
     }
   }
   Completed saving result to benchmark_report.json

.. note::
   You may see several red ``ERROR NRT:nrt_tensor_free`` errors at the end of the script output. These can be safely ignored - the actual benchmark results appear above these error messages.

All complete! The benchmark results are saved to ``benchmark_report.json`` in the container.

Understanding the results
-------------------------

The benchmark output provides three key metrics:

* **e2e_model**: End-to-end model performance including context encoding and token generation
* **context_encoding_model**: Performance of processing the input prompt
* **token_generation_model**: Performance of generating output tokens

Each metric includes:

* Latency percentiles (p50, p90, p95, p99, p100) in milliseconds
* Average latency in milliseconds
* Throughput in tokens per second

Common issues
-------------

Did you encounter an error while working through this tutorial? Here are common issues and solutions:

- **Model download fails**: Verify you have accepted the Llama-2 license on HuggingFace and have valid AWS credentials
- **Container won't start**: Check that you have sufficient Neuron devices allocated
- **Compilation fails**: Ensure you have enough memory and the correct PyTorch version
- **Slow performance**: Verify your ``tp-degree`` matches your available Neuron devices
- **Memory issues**: Consider using a larger instance type or reducing batch size

For additional help, refer to the complete NeuronX Distributed Inference documentation.

Clean up
--------

To clean up resources after completing this tutorial:

1. Exit the container:

   .. code-block:: bash

      exit

2. Stop and remove the container:

   .. code-block:: bash

      docker stop pytorch-inference-demo
      docker rm pytorch-inference-demo

3. Remove the model files if no longer needed:

   .. code-block:: bash

      rm -rf /home/ec2-user/model_hf/Llama-2-7b

4. Terminate your EC2 instance if no longer needed.

Next steps
----------

Now that you've completed this tutorial, explore these related topics:

* Learn more about NeuronX Distributed Inference configuration options
* Explore different model architectures and optimization techniques
* Set up production deployment with monitoring and logging

Further reading
---------------

- `NeuronX Distributed Inference Documentation <#>`_ - Complete documentation for inference on Neuron
- `AWS Neuron SDK Documentation <https://awsdocs-neuron.readthedocs-hosted.com/>`_ - Full Neuron SDK reference
- `Llama-2 Model Card <https://huggingface.co/meta-llama/Llama-2-7b>`_ - Model details and license information
