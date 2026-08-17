.. meta::
   :description: Install PyTorch Neuron using AWS Deep Learning AMI on Inf2, Trn1, Trn2, Trn3
   :keywords: pytorch, neuron, dlami, installation, ami
   :framework: pytorch
   :installation-method: dlami
   :instance-types: inf2, trn1, trn2, trn3
   :os: ubuntu-24.04
   :python-versions: 3.11, 3.12
   :content-type: installation-guide
   :estimated-time: 5 minutes
   :date-modified: 2026-08-03

Install PyTorch via Deep Learning AMI
======================================

Install PyTorch with Neuron support using pre-configured AWS Deep Learning AMIs.

⏱️ **Estimated time**: 5 minutes

.. note::
   Want to read about Neuron's Deep Learning machine images (DLAMIs) before diving in? Check out the :doc:`/deploy/environments/dlami`.

.. warning::
   The NeuronX Distributed (NxD) library for training (``neuronx_distributed_training``) is no longer included on Neuron DLAMIs and DLCs as of release v2.31.0. To manually configure your environment to use ``neuronx_distributed_training``, see :doc:`Install PyTorch via manual installation </setup/pytorch/manual>`.


----

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Requirement
     - Details
   * - Instance Type
     - Inf2, Trn1, Trn2, or Trn3
   * - AWS Account
     - With EC2 permissions
   * - SSH Key Pair
     - For instance access
   * - AWS CLI
     - Configured with credentials (optional)

Installation steps
------------------

**Step 1: Find the latest AMI**

Get the latest Neuron Multi-Framework DLAMI for Ubuntu 24.04 using the AWS CLI:

.. code-block:: bash
   
   aws ec2 describe-images \
     --owners amazon \
     --filters "Name=name,Values=Deep Learning AMI Neuron (Ubuntu 24.04)*" \
     --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
     --output text

Or use the SSM parameter to resolve the latest image directly:

.. code-block:: bash

   aws ssm get-parameter \
     --region us-east-1 \
     --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id \
     --query "Parameter.Value" \
     --output text

You can also use the AWS EC2 parameter store to find the ID of a DLAMI. See `Find a DLAMI ID <https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html>`__ for details. Record the ID (``image-id``) for the next step.

**Step 2: Launch instance**

Launch a Trn1 or Inf2 instance with the AMI using the AWS CLI:

.. code-block:: bash
   
   aws ec2 run-instances \
     --image-id ami-xxxxxxxxxxxxxxxxx \
     --instance-type trn1.2xlarge \
     --key-name your-key-pair \
     --security-group-ids sg-xxxxxxxxx \
     --subnet-id subnet-xxxxxxxxx

Replace:

- ``ami-xxxxxxxxxxxxxxxxx`` with AMI ID from Step 1
- ``your-key-pair`` with your SSH key pair name
- ``sg-xxxxxxxxx`` with your security group ID
- ``subnet-xxxxxxxxx`` with your subnet ID

You can also launch your DLAMI through the AWS EC2 web console, which also provides hints for security group and subnet IDs. For more details, see `Launch a DLAMI <https://docs.aws.amazon.com/dlami/latest/devguide/launch.html>`__.

**Step 3: Connect to instance**

.. code-block:: bash
   
   ssh -i your-key-pair.pem ubuntu@<instance-public-ip>

**Step 4: Activate environment**

The multi-framework DLAMI includes pre-configured virtual environments. Activate the vLLM environment for LLM inference:

.. code-block:: bash
   
   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate

Or activate the JAX environment:

.. code-block:: bash

   source /opt/aws_neuronx_venv_jax_0_10/bin/activate

For a full list of available virtual environments, see :ref:`neuron-dlami-multifw-venvs`.

**Step 5: Verify installation**

.. code-block:: bash

   neuron-ls
   pip list | grep -E "vllm|neuron|torch"

You should see output similar to this (the versions, instance IDs, and details should match your expected ones, not the ones in this example):

**Expected output**:

.. code-block:: text
   
   +--------+--------+--------+-----------+
   | DEVICE | CORES  | MEMORY | CONNECTED |
   +--------+--------+--------+-----------+
   | 0      | 2      | 32 GB  | Yes       |
   | 1      | 2      | 32 GB  | Yes       |
   +--------+--------+--------+-----------+

.. dropdown:: ⚠️ Troubleshooting: Module not found
   :color: warning
   :animate: fade-in
   
   If you see ``ModuleNotFoundError`` when importing packages:
   
   1. Verify virtual environment is activated:
      
      .. code-block:: bash
         
         which python
         # Should show a path under /opt/aws_neuronx_venv_*
   
   2. Check Python version:
      
      .. code-block:: bash
         
         python --version
         # Should be 3.12 or higher
   
   3. Verify installed packages:
      
      .. code-block:: bash
         
         pip list | grep -E "neuron|vllm|torch"

.. dropdown:: ⚠️ Troubleshooting: No Neuron devices found
   :color: warning
   :animate: fade-in
   
   If ``neuron-ls`` shows no devices:
   
   1. Verify instance type:
      
      .. code-block:: bash
         
         curl http://169.254.169.254/latest/meta-data/instance-type
         # Should show trn1.*, trn2.*, trn3.*, or inf2.*
   
   2. Check Neuron driver:
      
      .. code-block:: bash
         
         lsmod | grep neuron
         # Should show neuron driver loaded
   
   3. Restart Neuron runtime:
      
      .. code-block:: bash
         
         sudo systemctl restart neuron-monitor
         neuron-ls

Update an existing installation
--------------------------------

To update PyTorch versions or Neuron drivers on an existing DLAMI, see
:doc:`update-dlami`.


.. tip:: **vLLM for LLM inference**
   
   Neuron provides a dedicated vLLM DLAMI with vLLM and the vLLM-Neuron Plugin pre-installed.
   Launch the **Deep Learning AMI Neuron PyTorch Inference vLLM (Ubuntu 24.04)** and activate
   the pre-configured environment:
   
   .. code-block:: bash
      
      source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
   
   vLLM provides an OpenAI-compatible API, continuous batching, and supports models like
   Llama 2/3.1/3.3/4, Qwen 2.5/3, and multimodal models with quantization support (INT8/FP8).
   
   The vLLM environment is also available in the multi-framework DLAMI. For more details
   on available DLAMIs and SSM parameters, see :doc:`/deploy/environments/dlami`.

Next steps
----------

Now that PyTorch is installed:

1. **Try a Quick Example**:
   
   .. code-block:: python
      
      import torch
      import torch_neuronx

      # Simple tensor operation on Neuron
      x = torch.randn(3, 3)
      model = torch.nn.Linear(3, 3)

      # Compile for Neuron
      trace = torch_neuronx.trace(model, x)
      print(trace(x))

2. **Follow Tutorials**:
   
   - :doc:`Training with torch-neuronx [archived content] </archive/nxd-training/index>`
   - :doc:`/frameworks/torch/inference-torch-neuronx`

3. **Read Documentation**:
   
   - :doc:`Training developer guide [archived content] </archive/nxd-training/index>`
   - :doc:`/frameworks/torch/index`

4. **Explore Tools**:
   
   - :doc:`/tools/neuron-explorer/index`
   - :doc:`/tools/neuron-sys-tools/neuron-top-user-guide`

5. **Deploy LLM inference**: :doc:`/deploy/environments/dlami` (vLLM on Neuron)

Additional resources
--------------------

- :doc:`/deploy/environments/dlami` - DLAMI documentation
- :doc:`/deploy/index` - Container-based deployment
- :doc:`../troubleshooting` - Common issues and solutions
- :doc:`/release-notes/index` - Version compatibility information
