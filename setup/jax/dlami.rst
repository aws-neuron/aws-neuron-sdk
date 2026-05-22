.. meta::
   :description: Install JAX Neuron using AWS Deep Learning AMI on Inf2, Trn1, Trn2, Trn3
   :keywords: jax, neuron, dlami, installation, ami
   :framework: jax
   :installation-method: dlami
   :instance-types: inf2, trn1, trn2, trn3
   :os: ubuntu-24.04, ubuntu-22.04, al2023
   :python-versions: 3.11, 3.12
   :content-type: installation-guide
   :estimated-time: 5 minutes
   :date-modified: 2026-03-03

Install JAX via Deep Learning AMI
===================================

Install JAX with Neuron support using pre-configured AWS Deep Learning AMIs. 

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

Installation Steps
------------------

.. tab-set::

   .. tab-item:: Ubuntu 24.04
      :sync: ubuntu-24-04
      
      **Step 1: Find the Latest AMI**
      
      Get the latest JAX DLAMI for Ubuntu 24.04:
      
      .. code-block:: bash
         
         aws ec2 describe-images \
           --owners amazon \
           --filters "Name=name,Values=Deep Learning AMI Neuron JAX * (Ubuntu 24.04)*" \
           --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
           --output text
      
      **Step 2: Launch Instance**
      
      Launch a Trn1 or Inf2 instance with the AMI:
      
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
      
      **Step 3: Connect to Instance**
      
      .. code-block:: bash
         
         ssh -i your-key-pair.pem ubuntu@<instance-public-ip>
      
      **Step 4: Activate Environment**
      
      The DLAMI includes a pre-configured virtual environment:
      
      .. code-block:: bash
         
         source /opt/aws_neuronx_venv_jax/bin/activate
      
      **Step 5: Verify Installation**
      
      .. code-block:: python
         
         python3 << EOF
         import jax
         import jax_neuronx
         
         print(f"JAX version: {jax.__version__}")
         print(f"Devices: {jax.devices()}")
         
         # Check Neuron devices
         import subprocess
         result = subprocess.run(['neuron-ls'], capture_output=True, text=True)
         print(result.stdout)
         EOF
      
      **Expected output**:
      
      .. code-block:: text
         
         JAX version: 0.7.0
         Devices: [NeuronDevice(id=0), NeuronDevice(id=1)]
         
         +--------+--------+--------+-----------+
         | DEVICE | CORES  | MEMORY | CONNECTED |
         +--------+--------+--------+-----------+
         | 0      | 2      | 32 GB  | Yes       |
         | 1      | 2      | 32 GB  | Yes       |
         +--------+--------+--------+-----------+
      
      .. dropdown:: ⚠️ Troubleshooting: Module not found
         :color: warning
         :animate: fade-in
         
         If you see ``ModuleNotFoundError: No module named 'jax_neuronx'``:
         
         1. Verify virtual environment is activated:
            
            .. code-block:: bash
               
               which python
               # Should show: /opt/aws_neuronx_venv_jax/bin/python
         
         2. Check Python version:
            
            .. code-block:: bash
               
               python --version
               # Should be 3.11 or higher
         
         3. Reinstall jax-neuronx:
            
            .. code-block:: bash
               
               pip install --force-reinstall jax-neuronx
      
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

   .. tab-item:: Ubuntu 22.04
      :sync: ubuntu-22-04
      
      **Step 1: Find the Latest AMI**

      .. important::
         Ubuntu 22.04 has reached end-of-support on Neuron. Neuron no longer provides Ubuntu 22.04 DLAMIs or container images. New deployments should use Ubuntu 24.04. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc`.
      
      Get the latest JAX DLAMI for Ubuntu 22.04:
      
      .. code-block:: bash
         
         aws ec2 describe-images \
           --owners amazon \
           --filters "Name=name,Values=Deep Learning AMI Neuron JAX * (Ubuntu 22.04)*" \
           --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
           --output text
      
      **Step 2: Launch Instance**
      
      .. code-block:: bash
         
         aws ec2 run-instances \
           --image-id ami-xxxxxxxxxxxxxxxxx \
           --instance-type trn1.2xlarge \
           --key-name your-key-pair \
           --security-group-ids sg-xxxxxxxxx \
           --subnet-id subnet-xxxxxxxxx
      
      **Step 3: Connect to Instance**
      
      .. code-block:: bash
         
         ssh -i your-key-pair.pem ubuntu@<instance-public-ip>
      
      **Step 4: Activate Environment**
      
      .. code-block:: bash
         
         source /opt/aws_neuronx_venv_jax/bin/activate
      
      **Step 5: Verify Installation**
      
      .. code-block:: python
         
         python3 << EOF
         import jax
         import jax_neuronx
         
         print(f"JAX version: {jax.__version__}")
         print(f"Devices: {jax.devices()}")
         
         # Check Neuron devices
         import subprocess
         result = subprocess.run(['neuron-ls'], capture_output=True, text=True)
         print(result.stdout)
         EOF
      
      .. dropdown:: ⚠️ Troubleshooting: Module not found
         :color: warning
         :animate: fade-in
         
         If you see ``ModuleNotFoundError: No module named 'jax_neuronx'``:
         
         1. Verify virtual environment is activated
         2. Check Python version: ``python --version`` (should be 3.11+)
         3. Reinstall: ``pip install --force-reinstall jax-neuronx``
      
      .. dropdown:: ⚠️ Troubleshooting: No Neuron devices found
         :color: warning
         :animate: fade-in
         
         If ``neuron-ls`` shows no devices:
         
         1. Verify instance type
         2. Check Neuron driver: ``lsmod | grep neuron``
         3. Restart runtime: ``sudo systemctl restart neuron-monitor``

   .. tab-item:: Amazon Linux 2023
      :sync: al2023
      
      **Step 1: Find the Latest AMI**
      
      Get the latest JAX DLAMI for Amazon Linux 2023:
      
      .. code-block:: bash
         
         aws ec2 describe-images \
           --owners amazon \
           --filters "Name=name,Values=Deep Learning AMI Neuron JAX * (Amazon Linux 2023)*" \
           --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
           --output text
      
      **Step 2: Launch Instance**
      
      .. code-block:: bash
         
         aws ec2 run-instances \
           --image-id ami-xxxxxxxxxxxxxxxxx \
           --instance-type trn1.2xlarge \
           --key-name your-key-pair \
           --security-group-ids sg-xxxxxxxxx \
           --subnet-id subnet-xxxxxxxxx
      
      **Step 3: Connect to Instance**
      
      .. code-block:: bash
         
         ssh -i your-key-pair.pem ec2-user@<instance-public-ip>
      
      .. note::
         
         Amazon Linux 2023 uses ``ec2-user`` instead of ``ubuntu``.
      
      **Step 4: Activate Environment**
      
      .. code-block:: bash
         
         source /opt/aws_neuronx_venv_jax/bin/activate
      
      **Step 5: Verify Installation**
      
      .. code-block:: python
         
         python3 << EOF
         import jax
         import jax_neuronx
         
         print(f"JAX version: {jax.__version__}")
         print(f"Devices: {jax.devices()}")
         
         # Check Neuron devices
         import subprocess
         result = subprocess.run(['neuron-ls'], capture_output=True, text=True)
         print(result.stdout)
         EOF
      
      .. dropdown:: ⚠️ Troubleshooting: Module not found
         :color: warning
         :animate: fade-in
         
         If you see ``ModuleNotFoundError: No module named 'jax_neuronx'``:
         
         1. Verify virtual environment is activated
         2. Check Python version: ``python --version`` (should be 3.11+)
         3. Reinstall: ``pip install --force-reinstall jax-neuronx``
      
      .. dropdown:: ⚠️ Troubleshooting: No Neuron devices found
         :color: warning
         :animate: fade-in
         
         If ``neuron-ls`` shows no devices:
         
         1. Verify instance type
         2. Check Neuron driver: ``lsmod | grep neuron``
         3. Restart runtime: ``sudo systemctl restart neuron-monitor``

Next Steps
----------

Now that JAX is installed:

1. **Try a Quick Example**:
   
   .. code-block:: python
      
      import jax
      import jax.numpy as jnp
      
      # Simple operation on Neuron
      x = jnp.array([1.0, 2.0, 3.0])
      y = jnp.array([4.0, 5.0, 6.0])
      result = jax.numpy.multiply(x, y)
      print(result)

2. **Read Documentation**:
   
   - :doc:`/frameworks/jax/index`
   - :doc:`/frameworks/jax/api-reference-guide/index`

3. **Explore Setup Guide**:
   
   - :doc:`/frameworks/jax/setup/jax-setup`

Additional Resources
--------------------

- :doc:`/deploy/environments/dlami` - DLAMI documentation
- :doc:`/deploy/index` - Container-based deployment
- :doc:`../troubleshooting` - Common issues and solutions
- :doc:`/release-notes/index` - Version compatibility information
