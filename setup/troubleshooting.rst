.. meta::
   :description: Troubleshooting guide for AWS Neuron SDK installation issues
   :keywords: neuron, troubleshooting, installation, errors, debugging
   :content-type: troubleshooting
   :date-modified: 2026-03-03

Installation Troubleshooting
=============================

Common issues and solutions for Neuron SDK installation.

Module Import Errors
--------------------

ModuleNotFoundError: No module named 'torch_neuronx'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Python cannot find torch_neuronx module after installation.

**Causes**:

- Virtual environment not activated
- Wrong Python version
- Installation failed silently
- Multiple Python installations

**Solutions**:

1. **Verify virtual environment**:
   
   .. code-block:: bash
      
      which python
      # Should show virtual environment path, not system Python

2. **Check Python version**:
   
   .. code-block:: bash
      
      python --version
      # Should be 3.10, 3.11, or 3.12

3. **Reinstall torch-neuronx**:
   
   .. code-block:: bash
      
      pip install --force-reinstall torch-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com

4. **Verify installation**:
   
   .. code-block:: bash
      
      pip list | grep neuron

ImportError: cannot import name 'neuron' from 'torch'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Import error when trying to use Neuron features.

**Cause**: Using PyTorch/XLA syntax with Native PyTorch backend.

**Solution**: Update code to use Native PyTorch syntax:

.. code-block:: python
   
   # Old (PyTorch/XLA)
   import torch_xla.core.xla_model as xm
   device = xm.xla_device()
   
   # New (Native PyTorch)
   import torch
   device = torch.device('neuron')

See :doc:`/torch/index` for complete migration guide.

Device and Runtime Errors
--------------------------

No Neuron devices found
~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: ``neuron-ls`` shows no devices or returns error.

**Causes**:

- Wrong instance type
- Neuron driver not loaded
- Runtime not started

**Solutions**:

1. **Verify instance type**:
   
   .. code-block:: bash
      
      curl http://169.254.169.254/latest/meta-data/instance-type
      # Should show inf2.*, trn1.*, trn2.*, trn3.*, or inf1.*

2. **Check Neuron driver**:
   
   .. code-block:: bash
      
      lsmod | grep neuron
      # Should show neuron driver loaded

3. **Install/reload driver**:
   
   .. code-block:: bash
      
      # Ubuntu/Debian
      sudo apt-get install -y aws-neuronx-dkms
      
      # Amazon Linux
      sudo yum install -y aws-neuronx-dkms

4. **Restart runtime**:
   
   .. code-block:: bash
      
      sudo systemctl restart neuron-monitor
      neuron-ls

RuntimeError: Neuron runtime initialization failed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Runtime fails to initialize when running models.

**Causes**:

- Insufficient permissions
- Runtime version mismatch
- Corrupted runtime state

**Solutions**:

1. **Check runtime status**:
   
   .. code-block:: bash
      
      sudo systemctl status neuron-monitor

2. **Verify permissions**:
   
   .. code-block:: bash
      
      ls -l /dev/neuron*
      # Should be accessible by current user

3. **Reinstall runtime**:
   
   .. code-block:: bash
      
      sudo apt-get install --reinstall aws-neuronx-runtime-lib

Version Compatibility Issues
-----------------------------

Compiler version mismatch
~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: Error about incompatible compiler version.

**Cause**: neuronx-cc version incompatible with framework version.

**Solution**: Install compatible versions:

.. code-block:: bash
   
   # For PyTorch 2.9
   pip install neuronx-cc==2.15.* --extra-index-url=https://pip.repos.neuron.amazonaws.com

See :doc:`/release-notes/index` for version compatibility matrix.

Package dependency conflicts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: pip reports conflicting dependencies.

**Solution**: Use fresh virtual environment:

.. code-block:: bash
   
   python3 -m venv ~/fresh_neuron_venv
   source ~/fresh_neuron_venv/bin/activate
   pip install -U pip
   # Install packages in correct order
   pip install torch==2.9.0
   pip install torch-neuronx neuronx-cc --extra-index-url=https://pip.repos.neuron.amazonaws.com

Network and Repository Issues
------------------------------

Cannot connect to Neuron repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptoms**: apt-get or pip cannot reach Neuron repositories.

**Solutions**:

1. **Verify network connectivity**:
   
   .. code-block:: bash
      
      curl -I https://apt.repos.neuron.amazonaws.com
      curl -I https://pip.repos.neuron.amazonaws.com

2. **Check proxy settings** (if behind corporate proxy):
   
   .. code-block:: bash
      
      export https_proxy=http://proxy.example.com:8080
      export http_proxy=http://proxy.example.com:8080

3. **Use alternative index URL**:
   
   .. code-block:: bash
      
      pip install torch-neuronx --index-url=https://pip.repos.neuron.amazonaws.com

GPG key expired
~~~~~~~~~~~~~~~

**Symptoms**: "EXPKEYSIG" error during apt-get update.

**Solution**:

.. code-block:: bash
   
   sudo install -m 0755 -d /etc/apt/keyrings
   wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/neuron.gpg
   sudo apt-get update -y

Getting Help
------------

If issues persist:

1. **Check release notes**: :doc:`/release-notes/index`
2. **Review documentation**: :doc:`/torch/index`
3. **GitHub Issues**: `aws-neuron-sdk/aws-neuron-sdk <https://github.com/aws-neuron/aws-neuron-sdk/issues>`_
4. **AWS Support**: Open support case if you have AWS Support plan

Diagnostic Information
----------------------

When reporting issues, include:

.. code-block:: bash
   
   # System information
   uname -a
   cat /etc/os-release
   
   # Instance type
   curl http://169.254.169.254/latest/meta-data/instance-type
   
   # Neuron devices
   neuron-ls
   
   # Package versions
   pip list | grep -E "(torch|neuron)"
   
   # Driver status
   lsmod | grep neuron
   sudo systemctl status neuron-monitor
