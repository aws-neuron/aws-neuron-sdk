.. meta::
   :description: Train your first model on AWS Trainium with PyTorch and Neuron SDK
   :keywords: neuron, training, quickstart, pytorch, trainium, trn1, getting started
   :instance-types: trn1, trn2, trn3
   :content-type: quickstart
   :date-modified: 2026-03-03

.. _training-quickstart:

Quickstart: Train a Model on Trainium
======================================

This quickstart guides you through training your first PyTorch model on AWS Trainium. You'll launch a Trn1 instance, install Neuron SDK, and run a simple training script. When you complete this quickstart, you'll understand the basic workflow for training models with Neuron.

**This quickstart is for**: ML engineers and data scientists new to AWS Trainium

**Time to complete**: ~15 minutes

Prerequisites
-------------

Before you begin, ensure you have:

- An AWS account with EC2 launch permissions
- AWS CLI configured with your credentials
- SSH key pair for EC2 access
- Basic familiarity with PyTorch
- Terminal access (Linux, macOS, or WSL on Windows)

Step 1: Launch a Trainium instance
-----------------------------------

In this step, you will launch a Trn1 instance using the AWS Deep Learning AMI.

First, launch a Trn1.2xlarge instance with the latest Deep Learning AMI:

.. code-block:: bash

   aws ec2 run-instances \
       --image-id resolve:ssm:/aws/service/deep-learning-base-neuron/ubuntu-22-04/latest \
       --instance-type trn1.2xlarge \
       --key-name YOUR_KEY_NAME \
       --security-group-ids YOUR_SECURITY_GROUP \
       --subnet-id YOUR_SUBNET_ID

.. note::
   
   Replace ``YOUR_KEY_NAME``, ``YOUR_SECURITY_GROUP``, and ``YOUR_SUBNET_ID`` with your values.
   
   Alternatively, launch the instance through the `EC2 Console <https://console.aws.amazon.com/ec2/>`_.

Once the instance is running, connect via SSH:

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
   | 1      | 2      | 32 GB  | 00:1f.0 |
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

Step 3: Create a training script
---------------------------------

In this step, you will create a simple PyTorch training script that uses Neuron acceleration.

Create a file named ``train_simple.py``:

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
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
   
   # Create model and move to Neuron device
   model = SimpleNet().to('neuron')
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(), lr=0.01)
   
   # Generate dummy training data
   batch_size = 32
   num_batches = 100
   
   print("Starting training...")
   model.train()
   
   for batch_idx in range(num_batches):
       # Create dummy batch
       inputs = torch.randn(batch_size, 784).to('neuron')
       targets = torch.randint(0, 10, (batch_size,)).to('neuron')
       
       # Training step
       optimizer.zero_grad()
       outputs = model(inputs)
       loss = criterion(outputs, targets)
       loss.backward()
       optimizer.step()
       
       if batch_idx % 10 == 0:
           print(f"Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
   
   print("Training complete!")

This script creates a simple neural network, moves it to the Neuron device, and trains it on synthetic data.

Step 4: Run training
---------------------

In the final step, you will run the training script and monitor its progress.

Execute the training script:

.. code-block:: bash

   python train_simple.py

You should see training progress output:

.. code-block:: text

   Starting training...
   Batch 0/100, Loss: 2.3156
   Batch 10/100, Loss: 2.2845
   Batch 20/100, Loss: 2.2534
   ...
   Training complete!

Monitor Neuron device utilization in another terminal:

.. code-block:: bash

   neuron-top

This shows real-time NeuronCore utilization, memory usage, and other metrics.

Confirmation
------------

Congratulations! You've successfully trained your first model on AWS Trainium. You should have:

- ✅ Launched a Trn1 instance with Neuron SDK
- ✅ Installed PyTorch with Neuron support
- ✅ Created and ran a training script on Neuron devices
- ✅ Monitored training with Neuron tools

If you encountered any issues, see the **Common issues** section below.

Common issues
-------------

**Issue**: ``ModuleNotFoundError: No module named 'torch_neuronx'``

**Solution**: Ensure you activated the virtual environment and installed packages:

.. code-block:: bash

   source neuron_env/bin/activate
   pip install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com

**Issue**: ``RuntimeError: No Neuron devices found``

**Solution**: Verify you're on a Trainium instance and devices are visible:

.. code-block:: bash

   neuron-ls

If no devices appear, check instance type and driver installation.

**Issue**: Training is slower than expected

**Solution**: This quickstart uses a small model for demonstration. For production workloads:

- Use larger batch sizes
- Enable XLA compilation with ``torch.compile()``
- See :doc:`/frameworks/torch/torch-neuronx/programming-guide/training/pytorch-neuron-programming-guide` for optimization techniques

Clean up
--------

To avoid ongoing charges, terminate your instance when finished:

.. code-block:: bash

   # From your local machine
   aws ec2 terminate-instances --instance-ids YOUR_INSTANCE_ID

Or use the EC2 Console to terminate the instance.

Next steps
----------

Now that you've completed this quickstart, explore more advanced training topics:

- :doc:`/frameworks/torch/torch-neuronx/programming-guide/training/pytorch-neuron-programming-guide` - Comprehensive training guide
- :doc:`/libraries/nxd-training/index` - Distributed training with NeuronX Distributed
- :doc:`/about-neuron/models/index` - Pre-tested model samples
- :doc:`/tools/neuron-explorer/index` - Profile and optimize training performance

Further reading
---------------

- :doc:`/setup/pytorch/index` - Detailed PyTorch installation options
- :doc:`/deploy/ec2/index` - EC2 deployment workflows
- :doc:`/frameworks/torch/index` - Complete PyTorch Neuron documentation
