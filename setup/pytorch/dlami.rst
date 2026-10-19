.. meta::
   :description: Install PyTorch Neuron (TorchNeuron) using AWS Deep Learning AMI on Trn2 and Trn3
   :keywords: pytorch, neuron, dlami, installation, ami
   :framework: pytorch
   :installation-method: dlami
   :instance-types: trn2, trn3
   :os: ubuntu-26.04, ubuntu-24.04, al2023
   :python-versions: 3.12, 3.13, 3.14
   :content-type: installation-guide
   :estimated-time: 5 minutes
   :date-modified: 2026-10-16

Install PyTorch via Deep Learning AMI
======================================

Install PyTorch with Neuron support using pre-configured AWS Deep Learning AMIs.

⏱️ **Estimated time**: 5 minutes

.. note::
   Want to read about Neuron's Deep Learning machine images (DLAMIs) before diving in? Check out the :doc:`/deploy/environments/dlami`.


----

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Requirement
     - Details
   * - Instance Type
     - Trn2 or Trn3
   * - AWS Account
     - With EC2 permissions
   * - SSH Key Pair
     - For instance access
   * - AWS CLI
     - Configured with credentials (optional)

Installation steps
------------------

**Step 1: Find the latest AMI**

List the Neuron DLAMIs for Neuron SDK 2.33.0 with the AWS CLI. In the output, find the multi-framework DLAMI for your operating system and record its ``ImageId``:

.. code-block:: bash
   
   aws ec2 describe-images \
     --owners amazon \
     --filters "Name=name,Values=Deep Learning *Neuron*" \
               "Name=description,Values=*Neuron SDK 2.33.0*" \
     --query 'Images[].{Name:Name, ImageId:ImageId}' \
     --output table \
     --region us-east-1

Or use the SSM parameter to resolve the latest image directly:

.. code-block:: bash

   aws ssm get-parameter \
     --region us-east-1 \
     --name /aws/service/neuron/dlami/multi-framework/ubuntu-26.04/latest/image_id \
     --query "Parameter.Value" \
     --output text

You can also use the AWS EC2 parameter store to find the ID of a DLAMI. See `Find a DLAMI ID <https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html>`__ for details. Record the ID (``image-id``) for the next step.

**Step 2: Launch instance**

Launch a Trn2 or Trn3 instance with the AMI using the AWS CLI:

.. code-block:: bash
   
   aws ec2 run-instances \
     --image-id ami-xxxxxxxxxxxxxxxxx \
     --instance-type trn2.48xlarge \
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
   
   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate

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
         # Should show trn2.* or trn3.*
   
   2. Check Neuron driver:
      
      .. code-block:: bash
         
         lsmod | grep neuron
         # Should show neuron driver loaded
   
   3. Restart Neuron runtime:
      
      .. code-block:: bash
         
         sudo systemctl restart neuron-monitor
         neuron-ls

Use a single-framework PyTorch DLAMI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use a single-framework PyTorch DLAMI instead of the multi-framework DLAMI, follow the steps for your operating system:

- **Ubuntu 26.04 and Ubuntu 24.04:** use the **Deep Learning AMI Neuron torch-neuronx 2.14.0** DLAMI. It provides TorchNeuron (``torch-neuronx``) for PyTorch 2.14, which supports Trn2 and Trn3 instances.
- **Amazon Linux 2023:** the torch-neuronx 2.14.0 DLAMI is not available. The Amazon Linux 2023 tab covers the PyTorch 2.9 (PyTorch/XLA) DLAMI from Neuron SDK 2.31.0 and earlier.

.. tab-set::

   .. tab-item:: Ubuntu 26.04
      :sync: ubuntu-26-04
      
      **Step 1: Find the latest AMI**
      
      List the Neuron DLAMIs for Neuron SDK 2.33.0 with the AWS CLI. In the output, find the torch-neuronx 2.14.0 DLAMI for Ubuntu 26.04 and record its ``ImageId``:
      
      .. code-block:: bash
         
         aws ec2 describe-images \
           --owners amazon \
           --filters "Name=name,Values=Deep Learning *Neuron*" \
                     "Name=description,Values=*Neuron SDK 2.33.0*" \
           --query 'Images[].{Name:Name, ImageId:ImageId}' \
           --output table \
           --region us-east-1

      You can also use the AWS EC2 parameter store to find the ID of a DLAMI. See `Find a DLAMI ID <https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html>`__ for details. Record the ID (``image-id``) for the next step.
      
      **Step 2: Launch instance**
      
      Launch a Trn2 or Trn3 instance with the AMI using the AWS CLI:
      
      .. code-block:: bash
         
         aws ec2 run-instances \
           --image-id ami-xxxxxxxxxxxxxxxxx \
           --instance-type trn2.48xlarge \
           --key-name your-key-pair \
           --security-group-ids sg-xxxxxxxxx \
           --subnet-id subnet-xxxxxxxxx
      
      Replace:
      
      - ``ami-xxxxxxxxxxxxxxxxx`` with AMI ID from Step 1
      - ``your-key-pair`` with your SSH key pair name
      - ``sg-xxxxxxxxx`` with your security group ID
      - ``subnet-xxxxxxxxx`` with your subnet ID
      
      **Step 3: Connect to instance**
      
      .. code-block:: bash
         
         ssh -i your-key-pair.pem ubuntu@<instance-public-ip>
      
      **Step 4: Activate environment**
      
      The DLAMI includes a pre-configured virtual environment:
      
      .. code-block:: bash
         
         source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
      
      **Step 5: Verify installation**
      
      .. code-block:: bash

         python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
         neuron-ls

      You should see output similar to this (the versions, instance IDs, and details should match your expected ones, not the ones in this example):
      
      **Expected output**:
      
      .. code-block:: text
         
         PyTorch 2.14.0+cpu, torch-neuronx 2.14.3.0
         
         +--------+--------+--------+-----------+
         | DEVICE | CORES  | MEMORY | CONNECTED |
         +--------+--------+--------+-----------+
         | 0      | 2      | 32 GB  | Yes       |
         | 1      | 2      | 32 GB  | Yes       |
         +--------+--------+--------+-----------+
      
      .. dropdown:: ⚠️ Troubleshooting: Module not found
         :color: warning
         :animate: fade-in
         
         If you see ``ModuleNotFoundError: No module named 'torch_neuronx'``:
         
         1. Verify virtual environment is activated
         2. Check Python version: ``python --version`` (should be 3.12 or higher)
         3. Reinstall: ``pip install --force-reinstall torch-neuronx``
      
      .. dropdown:: ⚠️ Troubleshooting: No Neuron devices found
         :color: warning
         :animate: fade-in
         
         If ``neuron-ls`` shows no devices:
         
         1. Verify instance type
         2. Check Neuron driver: ``lsmod | grep neuron``
         3. Restart runtime: ``sudo systemctl restart neuron-monitor``

   .. tab-item:: Ubuntu 24.04
      :sync: ubuntu-24-04
      
      **Step 1: Find the latest AMI**
      
      List the Neuron DLAMIs for Neuron SDK 2.33.0 with the AWS CLI. In the output, find the torch-neuronx 2.14.0 DLAMI for Ubuntu 24.04 and record its ``ImageId``:
      
      .. code-block:: bash
         
         aws ec2 describe-images \
           --owners amazon \
           --filters "Name=name,Values=Deep Learning *Neuron*" \
                     "Name=description,Values=*Neuron SDK 2.33.0*" \
           --query 'Images[].{Name:Name, ImageId:ImageId}' \
           --output table \
           --region us-east-1

      You can also use the AWS EC2 parameter store to find the ID of a DLAMI. See `Find a DLAMI ID <https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html>`__ for details. Record the ID (``image-id``) for the next step.
      
      **Step 2: Launch instance**
      
      Launch a Trn2 or Trn3 instance with the AMI using the AWS CLI:
      
      .. code-block:: bash
         
         aws ec2 run-instances \
           --image-id ami-xxxxxxxxxxxxxxxxx \
           --instance-type trn2.48xlarge \
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
      
      The DLAMI includes a pre-configured virtual environment:
      
      .. code-block:: bash
         
         source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
      
      **Step 5: Verify installation**
      
      .. code-block:: bash

         python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
         neuron-ls

      You should see output similar to this (the versions, instance IDs, and details should match your expected ones, not the ones in this example):
      
      **Expected output**:
      
      .. code-block:: text
         
         PyTorch 2.14.0+cpu, torch-neuronx 2.14.3.0
         
         +--------+--------+--------+-----------+
         | DEVICE | CORES  | MEMORY | CONNECTED |
         +--------+--------+--------+-----------+
         | 0      | 2      | 32 GB  | Yes       |
         | 1      | 2      | 32 GB  | Yes       |
         +--------+--------+--------+-----------+
      
      .. dropdown:: ⚠️ Troubleshooting: Module not found
         :color: warning
         :animate: fade-in
         
         If you see ``ModuleNotFoundError: No module named 'torch_neuronx'``:
         
         1. Verify virtual environment is activated:
            
            .. code-block:: bash
               
               which python
               # Should show /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/python
         
         2. Check Python version:
            
            .. code-block:: bash
               
               python --version
               # Should be 3.12 or higher
         
         3. Reinstall torch-neuronx:
            
            .. code-block:: bash
               
               pip install --force-reinstall torch-neuronx
      
      .. dropdown:: ⚠️ Troubleshooting: No Neuron devices found
         :color: warning
         :animate: fade-in
         
         If ``neuron-ls`` shows no devices:
         
         1. Verify instance type:
            
            .. code-block:: bash
               
               curl http://169.254.169.254/latest/meta-data/instance-type
               # Should show trn2.* or trn3.*
         
         2. Check Neuron driver:
            
            .. code-block:: bash
               
               lsmod | grep neuron
               # Should show neuron driver loaded
         
         3. Restart Neuron runtime:
            
            .. code-block:: bash
               
               sudo systemctl restart neuron-monitor
               neuron-ls

   .. tab-item:: Amazon Linux 2023
      :sync: al2023

      .. note::

         The **Deep Learning AMI Neuron torch-neuronx 2.14.0** DLAMI is available on Ubuntu 24.04
         and Ubuntu 26.04. The following steps are for the PyTorch 2.9 (PyTorch/XLA) DLAMI, which
         is published for Neuron SDK 2.31.0 and earlier only.
      
      **Step 1: Find the latest AMI**
      
      List the Neuron DLAMIs for Neuron SDK 2.31.0 with the AWS CLI. In the output, find the PyTorch 2.9 DLAMI for Amazon Linux 2023 and record its ``ImageId``:
      
      .. code-block:: bash
         
         aws ec2 describe-images \
           --owners amazon \
           --filters "Name=name,Values=Deep Learning *Neuron*" \
                     "Name=description,Values=*Neuron SDK 2.31.0*" \
           --query 'Images[].{Name:Name, ImageId:ImageId}' \
           --output table \
           --region us-east-1
      
      **Step 2: Launch instance**
      
      .. code-block:: bash
         
         aws ec2 run-instances \
           --image-id ami-xxxxxxxxxxxxxxxxx \
           --instance-type trn1.2xlarge \
           --key-name your-key-pair \
           --security-group-ids sg-xxxxxxxxx \
           --subnet-id subnet-xxxxxxxxx
      
      **Step 3: Connect to instance**
      
      .. code-block:: bash
         
         ssh -i your-key-pair.pem ec2-user@<instance-public-ip>
      
      .. note::
         
         Amazon Linux 2023 uses ``ec2-user`` instead of ``ubuntu``.
      
      **Step 4: Activate environment**
      
      .. code-block:: bash
         
         source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
      
      **Step 5: Verify installation**
      
      .. code-block:: bash

         python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
         neuron-ls

      You should see output similar to this (the versions, instance IDs, and details should match your expected ones, not the ones in this example):
      
      **Expected output**:
      
      .. code-block:: text
         
         PyTorch 2.9.0+cpu, torch-neuronx 2.9.0.1.0
         
         +--------+--------+--------+-----------+
         | DEVICE | CORES  | MEMORY | CONNECTED |
         +--------+--------+--------+-----------+
         | 0      | 2      | 32 GB  | Yes       |
         | 1      | 2      | 32 GB  | Yes       |
         +--------+--------+--------+-----------+
      
      .. dropdown:: ⚠️ Troubleshooting: Module not found
         :color: warning
         :animate: fade-in
         
         If you see ``ModuleNotFoundError: No module named 'torch_neuronx'``:
         
         1. Verify virtual environment is activated
         2. Check Python version: ``python --version`` (should be 3.12 or later)
         3. Reinstall: ``pip install --force-reinstall torch-neuronx``
      
      .. dropdown:: ⚠️ Troubleshooting: No Neuron devices found
         :color: warning
         :animate: fade-in
         
         If ``neuron-ls`` shows no devices:
         
         1. Verify instance type
         2. Check Neuron driver: ``lsmod | grep neuron``
         3. Restart runtime: ``sudo systemctl restart neuron-monitor``

Update an existing installation
--------------------------------

To update PyTorch versions or Neuron drivers on an existing DLAMI, see
:doc:`update-dlami`.


.. tip:: **vLLM for LLM inference**
   
   Neuron provides a dedicated vLLM DLAMI with vLLM and the vLLM-Neuron Plugin pre-installed.
   Launch the **Deep Learning AMI Neuron PyTorch Inference vLLM 0.29.0.1.2.0 (Ubuntu 26.04)** and activate
   the pre-configured environment:
   
   .. code-block:: bash
      
      source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
   
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
   
   - :doc:`TorchNeuron tutorials </torch/tutorials/index>`
   - :doc:`vLLM Neuron online serving quickstart </vllm-neuron/docs/getting-started/quickstart-online-serving>`

3. **Read Documentation**:
   
   - :doc:`TorchNeuron quickstart </torch/get-started/quickstart-build-first-model>`
   - :doc:`/torch/index`

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
