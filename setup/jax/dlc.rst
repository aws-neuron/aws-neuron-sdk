.. meta::
   :description: Install JAX Neuron using Deep Learning Containers on Inf2, Trn1, Trn2, Trn3
   :keywords: jax, neuron, dlc, container, docker, installation
   :framework: jax
   :installation-method: container
   :instance-types: inf2, trn1, trn2, trn3
   :os: ubuntu-24.04, ubuntu-22.04, al2023
   :content-type: installation-guide
   :estimated-time: 10 minutes
   :date-modified: 2026-03-03

Install JAX via Deep Learning Container
=========================================

Install JAX with Neuron support using pre-configured AWS Deep Learning Containers (DLCs).

⏱️ **Estimated time**: ~10 minutes

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Requirement
     - Details
   * - Instance Type
     - Inf2, Trn1, Trn2, or Trn3
   * - Neuron Driver on Host
     - ``aws-neuronx-dkms`` installed on the host instance
   * - Docker Installed
     - Docker engine running on the host instance
   * - AWS Account
     - With EC2 permissions

Available container images
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70
   
   * - Image
     - ECR URI
   * - JAX Training
     - ``public.ecr.aws/neuron/jax-training-neuronx``

.. note::

   JAX DLCs are currently available for training workloads. For the full list of available images and tags, see `JAX Training Containers <https://github.com/aws-neuron/deep-learning-containers#jax-training-neuronx>`_.

For more information, see :doc:`/deploy/environments/dlc-images`.

Installation steps
------------------

.. tab-set::

   .. tab-item:: Ubuntu 24.04
      :sync: ubuntu-24-04
      
      **Step 1: Install Neuron driver on host**
      
      Configure the Neuron repository and install the driver:
      
      .. code-block:: bash
         
         . /etc/os-release
         sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
         deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
         EOF
         wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
         sudo apt-get update
         sudo apt-get install -y aws-neuronx-dkms
      
      **Step 2: Install and verify Docker**
      
      Install Docker and add your user to the ``docker`` group:
      
      .. code-block:: bash
         
         sudo apt-get install -y docker.io
         sudo usermod -aG docker $USER
      
      Log out and log back in to refresh group membership, then verify:
      
      .. code-block:: bash
         
         docker run hello-world
      
      **Step 3: Pull the DLC image from ECR**
      
      Pull the JAX Training DLC image:
      
      .. code-block:: bash
         
         docker pull public.ecr.aws/neuron/jax-training-neuronx:<image_tag>
      
      Replace ``<image_tag>`` with the desired tag from the `JAX Training Containers <https://github.com/aws-neuron/deep-learning-containers#jax-training-neuronx>`_ repository.
      
      **Step 4: Run the container**
      
      Launch the container with access to Neuron devices:
      
      .. code-block:: bash
         
         docker run -it \
           --device=/dev/neuron0 \
           --cap-add SYS_ADMIN \
           --cap-add IPC_LOCK \
           public.ecr.aws/neuron/jax-training-neuronx:<image_tag> \
           bash
      
      .. note::
         
         Adjust the ``--device`` flags based on your instance type. Use ``ls /dev/neuron*`` on the host to list available devices. For example, a ``trn1.32xlarge`` has 16 devices (``/dev/neuron0`` through ``/dev/neuron15``).
      
      **Step 5: Verify inside the container**
      
      Run the following commands inside the container to confirm Neuron devices are visible and JAX is installed:
      
      .. code-block:: bash
         
         neuron-ls
      
      .. code-block:: python
         
         python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
      
      **Expected output**:
      
      .. code-block:: text
         
         +--------+--------+--------+-----------+
         | DEVICE | CORES  | MEMORY | CONNECTED |
         +--------+--------+--------+-----------+
         | 0      | 2      | 32 GB  | Yes       |
         +--------+--------+--------+-----------+
         
         JAX version: 0.7.0
         Devices: [NeuronDevice(id=0), NeuronDevice(id=1)]
      
      .. dropdown:: ⚠️ Troubleshooting: Device not found in container
         :color: warning
         :animate: fade-in
         
         If ``neuron-ls`` shows no devices inside the container:
         
         1. Verify the Neuron driver is installed on the host:
            
            .. code-block:: bash
               
               # Run on the host (not inside the container)
               neuron-ls
         
         2. Confirm you passed the correct ``--device`` flag:
            
            .. code-block:: bash
               
               ls /dev/neuron*
         
         3. Restart the container with the correct device path:
            
            .. code-block:: bash
               
               docker run -it --device=/dev/neuron0 \
                 --cap-add SYS_ADMIN --cap-add IPC_LOCK \
                 public.ecr.aws/neuron/jax-training-neuronx:<image_tag> bash
      
      .. dropdown:: ⚠️ Troubleshooting: Permission denied
         :color: warning
         :animate: fade-in
         
         If you see ``permission denied`` errors when running Docker commands:
         
         1. Verify your user is in the ``docker`` group:
            
            .. code-block:: bash
               
               groups
               # Should include "docker"
         
         2. If not, add yourself and re-login:
            
            .. code-block:: bash
               
               sudo usermod -aG docker $USER
               # Log out and log back in
         
         3. Alternatively, run Docker with ``sudo``:
            
            .. code-block:: bash
               
               sudo docker run -it --device=/dev/neuron0 \
                 --cap-add SYS_ADMIN --cap-add IPC_LOCK \
                 public.ecr.aws/neuron/jax-training-neuronx:<image_tag> bash
      
      .. dropdown:: ⚠️ Troubleshooting: Image pull failure
         :color: warning
         :animate: fade-in
         
         If ``docker pull`` fails with a network or authentication error:
         
         1. Verify internet connectivity:
            
            .. code-block:: bash
               
               curl -s https://public.ecr.aws/v2/ | head -1
         
         2. Check that the image tag exists by browsing the `ECR Public Gallery <https://gallery.ecr.aws/neuron/jax-training-neuronx>`_.
         
         3. If you are behind a proxy, configure Docker proxy settings:
            
            .. code-block:: bash
               
               sudo mkdir -p /etc/systemd/system/docker.service.d
               sudo tee /etc/systemd/system/docker.service.d/proxy.conf > /dev/null <<EOF
               [Service]
               Environment="HTTP_PROXY=http://proxy:port"
               Environment="HTTPS_PROXY=http://proxy:port"
               EOF
               sudo systemctl daemon-reload
               sudo systemctl restart docker

   .. tab-item:: Ubuntu 22.04
      :sync: ubuntu-22-04
      
      **Step 1: Install Neuron driver on host**

      .. important::
         Ubuntu 22.04 has reached end-of-support on Neuron. Neuron no longer provides Ubuntu 22.04 DLAMIs or container images. New deployments should use Ubuntu 24.04. See :ref:`announce-eos-ubuntu-22-04-dlami-dlc`.
      
      Configure the Neuron repository and install the driver:
      
      .. code-block:: bash
         
         . /etc/os-release
         sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
         deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
         EOF
         wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -
         sudo apt-get update
         sudo apt-get install -y aws-neuronx-dkms
      
      **Step 2: Install and verify Docker**
      
      Install Docker and add your user to the ``docker`` group:
      
      .. code-block:: bash
         
         sudo apt-get install -y docker.io
         sudo usermod -aG docker $USER
      
      Log out and log back in to refresh group membership, then verify:
      
      .. code-block:: bash
         
         docker run hello-world
      
      **Step 3: Pull the DLC image from ECR**
      
      Pull the JAX Training DLC image:
      
      .. code-block:: bash
         
         docker pull public.ecr.aws/neuron/jax-training-neuronx:<image_tag>
      
      Replace ``<image_tag>`` with the desired tag from the `JAX Training Containers <https://github.com/aws-neuron/deep-learning-containers#jax-training-neuronx>`_ repository.
      
      **Step 4: Run the container**
      
      Launch the container with access to Neuron devices:
      
      .. code-block:: bash
         
         docker run -it \
           --device=/dev/neuron0 \
           --cap-add SYS_ADMIN \
           --cap-add IPC_LOCK \
           public.ecr.aws/neuron/jax-training-neuronx:<image_tag> \
           bash
      
      .. note::
         
         Adjust the ``--device`` flags based on your instance type. Use ``ls /dev/neuron*`` on the host to list available devices. For example, a ``trn1.32xlarge`` has 16 devices (``/dev/neuron0`` through ``/dev/neuron15``).
      
      **Step 5: Verify inside the container**
      
      Run the following commands inside the container to confirm Neuron devices are visible and JAX is installed:
      
      .. code-block:: bash
         
         neuron-ls
      
      .. code-block:: python
         
         python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
      
      .. dropdown:: ⚠️ Troubleshooting: Device not found in container
         :color: warning
         :animate: fade-in
         
         If ``neuron-ls`` shows no devices inside the container:
         
         1. Verify the Neuron driver is installed on the host
         2. Confirm you passed the correct ``--device`` flag: ``ls /dev/neuron*``
         3. Restart the container with the correct device path
      
      .. dropdown:: ⚠️ Troubleshooting: Permission denied
         :color: warning
         :animate: fade-in
         
         If you see ``permission denied`` errors when running Docker commands:
         
         1. Verify your user is in the ``docker`` group: ``groups``
         2. If not, add yourself: ``sudo usermod -aG docker $USER`` and re-login
         3. Alternatively, run Docker with ``sudo``
      
      .. dropdown:: ⚠️ Troubleshooting: Image pull failure
         :color: warning
         :animate: fade-in
         
         If ``docker pull`` fails with a network or authentication error:
         
         1. Verify internet connectivity: ``curl -s https://public.ecr.aws/v2/ | head -1``
         2. Check that the image tag exists in the `ECR Public Gallery <https://gallery.ecr.aws/neuron/jax-training-neuronx>`_
         3. If behind a proxy, configure Docker proxy settings

   .. tab-item:: Amazon Linux 2023
      :sync: al2023
      
      **Step 1: Install Neuron driver on host**
      
      Configure the Neuron repository and install the driver:
      
      .. code-block:: bash
         
         sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
         [neuron]
         name=Neuron YUM Repository
         baseurl=https://yum.repos.neuron.amazonaws.com
         enabled=1
         metadata_expire=0
         EOF
         sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
         sudo dnf update -y
         sudo dnf install -y "kernel-devel-uname-r == $(uname -r)"
         sudo dnf install -y aws-neuronx-dkms
      
      **Step 2: Install and verify Docker**
      
      Install Docker and add your user to the ``docker`` group:
      
      .. code-block:: bash
         
         sudo dnf install -y docker
         sudo usermod -aG docker $USER
      
      Log out and log back in to refresh group membership, then verify:
      
      .. code-block:: bash
         
         docker run hello-world
      
      **Step 3: Pull the DLC image from ECR**
      
      Pull the JAX Training DLC image:
      
      .. code-block:: bash
         
         docker pull public.ecr.aws/neuron/jax-training-neuronx:<image_tag>
      
      Replace ``<image_tag>`` with the desired tag from the `JAX Training Containers <https://github.com/aws-neuron/deep-learning-containers#jax-training-neuronx>`_ repository.
      
      **Step 4: Run the container**
      
      Launch the container with access to Neuron devices:
      
      .. code-block:: bash
         
         docker run -it \
           --device=/dev/neuron0 \
           --cap-add SYS_ADMIN \
           --cap-add IPC_LOCK \
           public.ecr.aws/neuron/jax-training-neuronx:<image_tag> \
           bash
      
      .. note::
         
         Adjust the ``--device`` flags based on your instance type. Use ``ls /dev/neuron*`` on the host to list available devices. For example, a ``trn1.32xlarge`` has 16 devices (``/dev/neuron0`` through ``/dev/neuron15``).
      
      **Step 5: Verify inside the container**
      
      Run the following commands inside the container to confirm Neuron devices are visible and JAX is installed:
      
      .. code-block:: bash
         
         neuron-ls
      
      .. code-block:: python
         
         python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
      
      .. dropdown:: ⚠️ Troubleshooting: Device not found in container
         :color: warning
         :animate: fade-in
         
         If ``neuron-ls`` shows no devices inside the container:
         
         1. Verify the Neuron driver is installed on the host
         2. Confirm you passed the correct ``--device`` flag: ``ls /dev/neuron*``
         3. Restart the container with the correct device path
      
      .. dropdown:: ⚠️ Troubleshooting: Permission denied
         :color: warning
         :animate: fade-in
         
         If you see ``permission denied`` errors when running Docker commands:
         
         1. Verify your user is in the ``docker`` group: ``groups``
         2. If not, add yourself: ``sudo usermod -aG docker $USER`` and re-login
         3. Alternatively, run Docker with ``sudo``
      
      .. dropdown:: ⚠️ Troubleshooting: Image pull failure
         :color: warning
         :animate: fade-in
         
         If ``docker pull`` fails with a network or authentication error:
         
         1. Verify internet connectivity: ``curl -s https://public.ecr.aws/v2/ | head -1``
         2. Check that the image tag exists in the `ECR Public Gallery <https://gallery.ecr.aws/neuron/jax-training-neuronx>`_
         3. If behind a proxy, configure Docker proxy settings

Next steps
----------

Now that JAX is running in a container:

1. **Find more container images**: Browse the full list of available Neuron DLC images at :doc:`/deploy/environments/dlc-images`.

2. **Customize your container**: Learn how to extend a DLC with additional packages at :ref:`containers-dlc-then-customize-devflow`.

3. **Read the JAX documentation**: Explore the :doc:`/frameworks/jax/index` for JAX framework documentation and tutorials.

Additional resources
--------------------

- :doc:`/deploy/environments/dlc-images` - Full DLC image list
- :doc:`/deploy/index` - Container documentation overview
- :doc:`../troubleshooting` - Common issues and solutions
- :doc:`/release-notes/index` - Version compatibility information
