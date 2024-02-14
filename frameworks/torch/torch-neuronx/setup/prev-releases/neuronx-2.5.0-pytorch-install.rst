.. _install-neuronx-2.5.0-pytorch:

Install PyTorch NeuronX (Neuron 2.5.0)
======================================

.. tab-set::

   .. tab-item:: PyTorch 1.11.0

      .. tab-set::

         .. tab-item:: Ubuntu 20 AMI 

            .. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst
            .. code:: bash
            
            	
            	# Configure Linux for Neuron repository updates
		. /etc/os-release

		sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
		deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
		EOF
		wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -		
            
            	# Update OS packages
		sudo apt-get update -y

		
		# Install OS headers
		sudo apt-get install linux-headers-$(uname -r) -y
 
		# Remove preinstalled packages and Install Neuron Driver and Runtime
		sudo apt-get remove aws-neuron-dkms  -y 
		sudo apt-get remove aws-neuronx-dkms  -y
		sudo apt-get remove aws-neuronx-oci-hook  -y		
		sudo apt-get remove aws-neuronx-runtime-lib -y
		sudo apt-get remove aws-neuronx-collectives -y
		sudo apt-get install aws-neuronx-dkms=2.6.5.0 -y
		sudo apt-get install aws-neuronx-oci-hook=2.1.2.0 -y
		sudo apt-get install aws-neuronx-runtime-lib=2.10.15.0 -y
		sudo apt-get install aws-neuronx-collectives=2.10.17.0 -y

		# Install EFA Driver(only required for multi-instance training)
		
		curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
		wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
		cat aws-efa-installer.key | gpg --fingerprint
		wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
  
		tar -xvf aws-efa-installer-latest.tar.gz
		cd aws-efa-installer && sudo bash efa_installer.sh --yes
		cd
		sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

		# Remove pre-installed package and Install Neuron Tools
		sudo apt-get remove aws-neuron-tools  -y
		sudo apt-get remove aws-neuronx-tools  -y
		sudo apt-get install aws-neuronx-tools=2.5.16.0 -y

		# Install Python venv and activate Python virtual environment to install
		# Neuron pip packages.
		sudo apt install python3.8-venv
		python3.8 -m venv aws_neuron_venv_pytorch_p38
		source aws_neuron_venv_pytorch_p38/bin/activate
		pip install -U pip

		# Install packages from beta repos
		python -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"
        
		# Install Python packages - Transformers package is needed for BERT
		python -m pip install torch-neuronx=="1.11.0.1.2.0" "neuronx-cc==2.2.0.73"


         .. tab-item:: Amazon Linux 2 AMI

            .. include :: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

            .. code:: bash

            
		# Configure Linux for Neuron repository updates

		sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
		[neuron]
		name=Neuron YUM Repository
		baseurl=https://yum.repos.neuron.amazonaws.com
		enabled=1
		metadata_expire=0
		EOF
		sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB 

		# Install OS headers
		sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y
		
		# Update OS packages
		sudo yum update -y


		# Remove preinstalled packages and Install Neuron Driver and Runtime
		sudo yum remove aws-neuron-dkms -y
		sudo yum remove aws-neuronx-dkms -y
		sudo yum remove aws-neuronx-oci-hook -y
		sudo yum remove aws-neuronx-runtime-lib -y
		sudo yum remove aws-neuronx-collectives -y
		sudo yum install aws-neuronx-dkms-2.6.5.0  -y
		sudo yum install aws-neuronx-oci-hook-2.1.2.0  -y
		sudo yum install aws-neuronx-runtime-lib-2.10.15.0  -y
		sudo yum install aws-neuronx-collectives-2.10.17.0  -y

		# Install EFA Driver(only required for multi-instance training)
		curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
		wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
		cat aws-efa-installer.key | gpg --fingerprint
		wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
		tar -xvf aws-efa-installer-latest.tar.gz
		cd aws-efa-installer && sudo bash efa_installer.sh --yes
		cd
		sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

             	# Remove pre-installed package and Install Neuron Tools
		sudo yum remove aws-neuron-tools  -y
		sudo yum remove aws-neuronx-tools  -y
		sudo yum install aws-neuronx-tools-2.5.16.0  -y

		# Install Python venv and activate Python virtual environment to install
		# Neuron pip packages.
		python3.7 -m venv aws_neuron_venv_pytorch_p37 
		source aws_neuron_venv_pytorch_p37/bin/activate
		python -m pip install -U pip

		# Install packages from beta repos
		python -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"
		
		# Install Python packages - Transformers package is needed for BERT
		python -m pip install torch-neuronx=="1.11.0.1.2.0" "neuronx-cc==2.2.0.73"
