.. meta::
   :description: Run your first inference workload on AWS Inferentia with PyTorch and Neuron SDK
   :keywords: neuron, inference, quickstart, pytorch, inferentia, inf2, getting started
   :instance-types: inf2, trn1
   :content-type: quickstart
   :date-modified: 2026-03-03

.. _inference-quickstart:

Quickstart: Run Inference on Inferentia
========================================

This quickstart guides you through running your first PyTorch inference workload on AWS Inferentia. You'll launch an Inf2 instance, compile a model for Neuron, and run predictions. When you complete this quickstart, you'll understand the basic workflow for deploying models on Inferentia.

**This quickstart is for**: ML engineers and developers deploying inference workloads

**Time to complete**: ~10 minutes

Prerequisites
-------------

Before you begin, ensure you have:

- An AWS account with EC2 launch permissions
- AWS CLI configured with your credentials
- SSH key pair for EC2 access
- Basic familiarity with PyTorch
- Terminal access (Linux, macOS, or WSL on Windows)

Step 1: Launch an Inferentia instance
--------------------------------------

In this step, you will launch an Inf2 instance using the AWS Deep Learning AMI.

Launch an Inf2.xlarge instance with the latest Deep Learning AMI:

.. code-block:: bash

   aws ec2 run-instances \
       --image-id resolve:ssm:/aws/service/deep-learning-base-neuron/ubuntu-22-04/latest \
       --instance-type inf2.xlarge \
       --key-name YOUR_KEY_NAME \
       --security-group-ids YOUR_SECURITY_GROUP \
       --subnet-id YOUR_SUBNET_ID

.. note::
   
   Replace ``YOUR_KEY_NAME``, ``YOUR_SECURITY_GROUP``, and ``YOUR_SUBNET_ID`` with your values.
   
   Alternatively, launch the instance through the `EC2 Console <https://console.aws.amazon.com/ec2/>`_.

Connect to your instance via SSH:

.. code-block:: bash

   ssh -i YOUR_KEY.pem ubuntu@YOUR_INSTANCE_IP

Verify Neuron devices are available:

.. code-block:: bash

   neuron-ls

You should see output showing available NeuronCores:

.. code-block:: text

   +--------+--------+--------+---------+
   | NEURON | NEURON | NEURON |   PCI   |
   | DEVICE | CORES  | MEMORY |   BDF   |
   +--------+--------+--------+---------+
   | 0      | 2      | 32 GB  | 00:1e.0 |
   +--------+--------+--------+---------+

Step 2: Set up your environment
--------------------------------

In this step, you will create a Python virtual environment and install PyTorch with Neuron support.

Create and activate a virtual environment:

.. code-block:: bash

   python3 -m venv neuron_env
   source neuron_env/bin/activate

Install PyTorch Neuron and dependencies:

.. code-block:: bash

   pip install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com

Verify the installation:

.. code-block:: bash

   python -c "import torch; import torch_neuronx; print(f'PyTorch: {torch.__version__}')"

You should see output confirming PyTorch is installed:

.. code-block:: text

   PyTorch: 2.9.0+cpu

Step 3: Compile a model for Neuron
-----------------------------------

In this step, you will create a simple model and compile it for Neuron inference.

Create a file named ``compile_model.py``:

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch_neuronx
   
   # Simple neural network
   class SimpleNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(784, 128)
           self.fc2 = nn.Linear(128, 10)
           self.relu = nn.ReLU()
       
       def forward(self, x):
           x = self.relu(self.fc1(x))
           return self.fc2(x)
   
   # Create model and set to eval mode
   model = SimpleNet()
   model.eval()
   
   # Create example input
   example_input = torch.randn(1, 784)
   
   # Trace and compile for Neuron
   print("Compiling model for Neuron...")
   neuron_model = torch_neuronx.trace(model, example_input)
   
   # Save compiled model
   neuron_model.save('simple_net_neuron.pt')
   print("Model compiled and saved to simple_net_neuron.pt")

Run the compilation script:

.. code-block:: bash

   python compile_model.py

You should see compilation progress and success message:

.. code-block:: text

   Compiling model for Neuron...
   INFO:Neuron:Compiling function _NeuronGraph$1 with neuronx-cc
   INFO:Neuron:Compilation successful
   Model compiled and saved to simple_net_neuron.pt

.. note::
   
   Model compilation happens once. The compiled model (``simple_net_neuron.pt``) can be reused for inference without recompiling.

Step 4: Run inference
----------------------

In the final step, you will load the compiled model and run predictions.

Create a file named ``run_inference.py``:

.. code-block:: python

   import torch
   import torch_neuronx
   
   # Load compiled model
   print("Loading compiled model...")
   neuron_model = torch.jit.load('simple_net_neuron.pt')
   
   # Create sample input
   sample_input = torch.randn(1, 784)
   
   # Run inference
   print("Running inference...")
   with torch.no_grad():
       output = neuron_model(sample_input)
   
   # Get prediction
   predicted_class = output.argmax(dim=1).item()
   print(f"Predicted class: {predicted_class}")
   print(f"Output logits: {output[0][:5].tolist()}")  # Show first 5 logits
   
   # Run multiple inferences to measure throughput
   print("\nRunning 100 inferences...")
   import time
   start = time.time()
   
   with torch.no_grad():
       for _ in range(100):
           output = neuron_model(sample_input)
   
   elapsed = time.time() - start
   throughput = 100 / elapsed
   print(f"Throughput: {throughput:.2f} inferences/second")
   print(f"Latency: {elapsed/100*1000:.2f} ms per inference")

Run the inference script:

.. code-block:: bash

   python run_inference.py

You should see inference results:

.. code-block:: text

   Loading compiled model...
   Running inference...
   Predicted class: 7
   Output logits: [0.123, -0.456, 0.789, -0.234, 0.567]
   
   Running 100 inferences...
   Throughput: 245.67 inferences/second
   Latency: 4.07 ms per inference

Monitor Neuron device utilization in another terminal:

.. code-block:: bash

   neuron-top

This shows real-time NeuronCore utilization and inference metrics.

Confirmation
------------

Congratulations! You've successfully run inference on AWS Inferentia. You should have:

- ✅ Launched an Inf2 instance with Neuron SDK
- ✅ Installed PyTorch with Neuron support
- ✅ Compiled a model for Neuron inference
- ✅ Ran predictions and measured throughput
- ✅ Monitored inference with Neuron tools

If you encountered any issues, see the **Common issues** section below.

Common issues
-------------

**Issue**: ``ModuleNotFoundError: No module named 'torch_neuronx'``

**Solution**: Ensure you activated the virtual environment and installed packages:

.. code-block:: bash

   source neuron_env/bin/activate
   pip install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com

**Issue**: ``RuntimeError: No Neuron devices found``

**Solution**: Verify you're on an Inferentia instance and devices are visible:

.. code-block:: bash

   neuron-ls

If no devices appear, check instance type and driver installation.

**Issue**: Compilation takes a long time

**Solution**: Model compilation is a one-time cost. For this simple model, compilation should take 1-2 minutes. Larger models take longer but only need to be compiled once. The compiled model can be saved and reused.

**Issue**: Lower throughput than expected

**Solution**: This quickstart uses a small model and batch size for demonstration. For production workloads:

- Use larger batch sizes (e.g., 4, 8, 16)
- Enable dynamic batching
- Use multiple NeuronCores in parallel
- See :doc:`/frameworks/torch/torch-neuronx/programming-guide/inference/index` for optimization techniques

Clean up
--------

To avoid ongoing charges, terminate your instance when finished:

.. code-block:: bash

   # From your local machine
   aws ec2 terminate-instances --instance-ids YOUR_INSTANCE_ID

Or use the EC2 Console to terminate the instance.

Next steps
----------

Now that you've completed this quickstart, explore more advanced inference topics:

- :doc:`/frameworks/torch/torch-neuronx/programming-guide/inference/index` - Comprehensive inference guide
- :doc:`/libraries/nxd-inference/index` - Production inference with NeuronX Distributed
- :doc:`/libraries/nxd-inference/vllm/quickstart-vllm-online-serving` - Deploy LLMs with vLLM
- :doc:`/about-neuron/models/index` - Pre-tested model samples
- :doc:`/tools/neuron-explorer/index` - Profile and optimize inference performance

Further reading
---------------

- :doc:`/setup/pytorch/index` - Detailed PyTorch installation options
- :doc:`/deploy/ec2/index` - EC2 deployment workflows
- :doc:`/frameworks/torch/index` - Complete PyTorch Neuron documentation
- :doc:`/compiler/index` - Understanding Neuron compilation
