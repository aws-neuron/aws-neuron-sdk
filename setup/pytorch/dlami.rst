.. meta::
   :description: Install PyTorch Neuron using AWS Deep Learning AMI on Inf2, Trn1, Trn2, Trn3
   :keywords: pytorch, neuron, dlami, installation, ami
   :framework: pytorch
   :installation-method: dlami
   :instance-types: inf2, trn1, trn2, trn3
   :os: ubuntu-24.04, ubuntu-22.04, al2023
   :python-versions: 3.11, 3.12
   :content-type: installation-guide
   :estimated-time: 5 minutes
   :date-modified: 2026-03-03

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
     - Inf2, Trn1, Trn2, or Trn3
   * - AWS Account
     - With EC2 permissions
   * - SSH Key Pair
     - For instance access
   * - AWS CLI
     - Configured with credentials (optional)

Installation steps
------------------

.. tab-set::

   .. tab-item:: Ubuntu 24.04
      :sync: ubuntu-24-04
      
      **Step 1: Find the latest AMI**
      
      Get the latest PyTorch DLAMI for Ubuntu 24.04 using the AWS CLI:
      
      .. code-block:: bash
         
         aws ec2 describe-images \
           --owners amazon \
           --filters "Name=name,Values=Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 24.04)*" \
           --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
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
      
      The DLAMI includes a pre-configured virtual environment:
      
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
         
         1. Verify virtual environment is activated:
            
            .. code-block:: bash
               
               which python
               # Should show:  source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
         
         2. Check Python version:
            
            .. code-block:: bash
               
               python --version
               # Should be 3.11 or higher
         
         3. Reinstall torch-neuronx:
            
            .. code-block:: bash
               
               pip install --force-reinstall torch-neuronx
      
      .. dropdown:: ⚠️ Troubleshooting: No Neuron devices found
         :color: warning
         :animate: fade-in
         
         If ``neuron-ls`` shows no devices:
         
         4. Verify instance type:
            
            .. code-block:: bash
               
               curl http://169.254.169.254/latest/meta-data/instance-type
               # Should show trn1.*, trn2.*, trn3.*, or inf2.*
         
         5. Check Neuron driver:
            
            .. code-block:: bash
               
               lsmod | grep neuron
               # Should show neuron driver loaded
         
         6. Restart Neuron runtime:
            
            .. code-block:: bash
               
               sudo systemctl restart neuron-monitor
               neuron-ls

   .. tab-item:: Ubuntu 22.04
      :sync: ubuntu-22-04
      
      **Step 1: Find the latest AMI**

      .. important::
         Ubuntu 22.04 has reached end-of-support on Neuron. Neuron no longer provides Ubuntu 22.04 DLAMIs or container images. New deployments should use Ubuntu 24.04. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc`.
      
      Get the latest PyTorch DLAMI for Ubuntu 22.04:
      
      .. code-block:: bash
         
         aws ec2 describe-images \
           --owners amazon \
           --filters "Name=name,Values=Deep Learning AMI Neuron PyTorch 2.9 (Ubuntu 22.04)*" \
           --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
           --output text
      
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
         
         ssh -i your-key-pair.pem ubuntu@<instance-public-ip>
      
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
         2. Check Python version: ``python --version`` (should be 3.11+)
         3. Reinstall: ``pip install --force-reinstall torch-neuronx``
      
      .. dropdown:: ⚠️ Troubleshooting: No Neuron devices found
         :color: warning
         :animate: fade-in
         
         If ``neuron-ls`` shows no devices:
         
         1. Verify instance type
         2. Check Neuron driver: ``lsmod | grep neuron``
         3. Restart runtime: ``sudo systemctl restart neuron-monitor``

   .. tab-item:: Amazon Linux 2023
      :sync: al2023
      
      **Step 1: Find the latest AMI**
      
      Get the latest PyTorch DLAMI for Amazon Linux 2023:
      
      .. code-block:: bash
         
         aws ec2 describe-images \
           --owners amazon \
           --filters "Name=name,Values=Deep Learning AMI Neuron PyTorch 2.9 (Amazon Linux 2023)*" \
           --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
           --output text
      
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
         2. Check Python version: ``python --version`` (should be 3.11+)
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
   Launch the **Deep Learning AMI Neuron PyTorch Inference vLLM (Ubuntu 24.04)** and activate
   the pre-configured environment:
   
   .. code-block:: bash
      
      source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
   
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
   
   - :doc:`/frameworks/torch/training-torch-neuronx`
   - :doc:`/frameworks/torch/inference-torch-neuronx`

3. **Read Documentation**:
   
   - :doc:`/frameworks/torch/torch-neuronx/programming-guide/training/index`
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
