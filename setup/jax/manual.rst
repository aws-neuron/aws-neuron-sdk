.. meta::
   :description: Manually install JAX Neuron on Inf2, Trn1, Trn2, Trn3 instances
   :keywords: jax, neuron, manual installation, pip
   :framework: jax
   :installation-method: manual
   :instance-types: inf2, trn1, trn2, trn3
   :os: ubuntu-24.04, ubuntu-22.04, al2023
   :python-versions: 3.11, 3.12
   :content-type: installation-guide
   :estimated-time: 15 minutes
   :date-modified: 2026-03-03

Install JAX Manually
=====================

Install JAX with Neuron support on existing systems using pip.

⏱️ **Estimated time**: 15 minutes

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Requirement
     - Details
   * - Instance Type
     - Inf2, Trn1, Trn2, or Trn3
   * - Operating System
     - Ubuntu 24.04, Ubuntu 22.04, or Amazon Linux 2023
   * - Python
     - Python 3.11, or 3.12
   * - Sudo Access
     - Required for driver installation
   * - Internet Access
     - For downloading packages

Installation Steps
------------------

.. tab-set::

   .. tab-item:: Ubuntu 24.04
      :sync: ubuntu-24-04
      
      **Step 1: Update System Packages**
      
      .. code-block:: bash
         
         sudo apt-get update
         sudo apt-get install -y python3-pip python3-venv
      
      **Step 2: Configure Neuron Repository**
      
      .. code-block:: bash
         
         # Add Neuron repository
         . /etc/os-release
         sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
         deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
         EOF
         
         # Add repository key
         wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
         
         # Update package list
         sudo apt-get update
      
      **Step 3: Install Neuron Driver and Runtime**
      
      .. code-block:: bash
         
         sudo apt-get install -y aws-neuronx-dkms
         sudo apt-get install -y aws-neuronx-runtime-lib
         sudo apt-get install -y aws-neuronx-collectives
      
      **Step 4: Create Virtual Environment**
      
      .. code-block:: bash
         
         python3.11 -m venv ~/neuron_venv_jax
         source ~/neuron_venv_jax/bin/activate
      
      **Step 5: Install JAX and Neuron Packages**
      
      .. code-block:: bash
         
         pip install -U pip
         pip install jax-neuronx[stable] --extra-index-url=https://pip.repos.neuron.amazonaws.com
      
      **Step 6: Verify Installation**
      
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
      
      .. dropdown:: ⚠️ Troubleshooting: GPG key error
         :color: warning
         :animate: fade-in
         
         If you see "EXPKEYSIG" error during apt-get update:
         
         .. code-block:: bash
            
            wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
            sudo apt-get update -y
      
      .. dropdown:: ⚠️ Troubleshooting: Driver installation failed
         :color: warning
         :animate: fade-in
         
         If driver installation fails:
         
         1. Check kernel headers are installed:
            
            .. code-block:: bash
               
               sudo apt-get install -y linux-headers-$(uname -r)
         
         2. Retry driver installation:
            
            .. code-block:: bash
               
               sudo apt-get install --reinstall aws-neuronx-dkms

   .. tab-item:: Ubuntu 22.04
      :sync: ubuntu-22-04
      
      **Step 1: Update System Packages**

      .. important::
         Ubuntu 22.04 has reached end-of-support on Neuron. Neuron no longer provides Ubuntu 22.04 DLAMIs or container images. New deployments should use Ubuntu 24.04. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc`.
      
      .. code-block:: bash
         
         sudo apt-get update
         sudo apt-get install -y python3-pip python3-venv
      
      **Step 2: Configure Neuron Repository**
      
      .. code-block:: bash
         
         # Add Neuron repository
         . /etc/os-release
         sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
         deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
         EOF
         
         # Add repository key
         wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
         
         # Update package list
         sudo apt-get update
      
      **Step 3: Install Neuron Driver and Runtime**
      
      .. code-block:: bash
         
         sudo apt-get install -y aws-neuronx-dkms
         sudo apt-get install -y aws-neuronx-runtime-lib
         sudo apt-get install -y aws-neuronx-collectives
      
      **Step 4: Create Virtual Environment**
      
      .. code-block:: bash
         
         python -m venv ~/neuron_venv_jax 
         source ~/neuron_venv_jax/bin/activate
      
      **Step 5: Install JAX and Neuron Packages**
      
      .. code-block:: bash
         
         pip install -U pip
         pip install jax-neuronx[stable] --extra-index-url=https://pip.repos.neuron.amazonaws.com
      
      **Step 6: Verify Installation**
      
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
      
      .. dropdown:: ⚠️ Troubleshooting: GPG key error
         :color: warning
         :animate: fade-in
         
         If you see "EXPKEYSIG" error, update the GPG key and retry.
      
      .. dropdown:: ⚠️ Troubleshooting: Driver installation failed
         :color: warning
         :animate: fade-in
         
         Ensure kernel headers are installed before retrying driver installation.

   .. tab-item:: Amazon Linux 2023
      :sync: al2023
      
      **Step 1: Update System Packages**
      
      .. code-block:: bash
         
         sudo yum update -y
         sudo yum install -y python3-pip python3-devel
      
      **Step 2: Configure Neuron Repository**
      
      .. code-block:: bash
         
         # Add Neuron repository
         sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
         [neuron]
         name=Neuron YUM Repository
         baseurl=https://yum.repos.neuron.amazonaws.com
         enabled=1
         metadata_expire=0
         EOF
         
         # Import GPG key
         sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
      
      **Step 3: Install Neuron Driver and Runtime**
      
      .. code-block:: bash
         
         sudo yum install -y aws-neuronx-dkms
         sudo yum install -y aws-neuronx-runtime-lib
         sudo yum install -y aws-neuronx-collectives
      
      **Step 4: Create Virtual Environment**
      
      .. code-block:: bash
         
         python -m venv ~/neuron_venv_jax
         source ~/neuron_venv_jax/bin/activate
      
      **Step 5: Install JAX and Neuron Packages**
      
      .. code-block:: bash
         
         pip install -U pip
         pip install jax-neuronx[stable] --extra-index-url=https://pip.repos.neuron.amazonaws.com
      
      **Step 6: Verify Installation**
      
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
      
      .. dropdown:: ⚠️ Troubleshooting: Repository access error
         :color: warning
         :animate: fade-in
         
         If you cannot access the Neuron repository:
         
         1. Verify network connectivity
         2. Check proxy settings if behind corporate firewall
         3. Ensure GPG key is imported correctly
      
      .. dropdown:: ⚠️ Troubleshooting: Driver installation failed
         :color: warning
         :animate: fade-in
         
         Ensure kernel-devel package is installed:
         
         .. code-block:: bash
            
            sudo yum install -y kernel-devel-$(uname -r)

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

- :doc:`dlami` - Use pre-configured DLAMI instead
- :doc:`dlc` - Use pre-configured Docker containers
- :doc:`/deploy/index` - Container-based deployment
- :doc:`../troubleshooting` - Common issues and solutions
- :doc:`/release-notes/index` - Version compatibility information
