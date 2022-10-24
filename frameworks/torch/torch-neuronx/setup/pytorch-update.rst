.. _pytorch-neuronx-update:

Update to latest PyTorch Neuron  (``torch-neuronx``)
====================================================

.. tab-set::

   .. tab-item:: PyTorch 1.11.0

      .. tab-set::

         .. tab-item:: Ubuntu 20 AMI 

            .. include :: note-setup-general.rst
            .. code:: bash
            
            		
               # Update OS packages
               sudo apt-get update -y

               
               # Update OS headers
               sudo apt-get install linux-headers-$(uname -r) -y
         
               # Update Neuron Driver and Runtime
               sudo apt-get install aws-neuronx-dkms=2.* -y
               sudo apt-get install aws-neuronx-oci-hook=2.* -y
               sudo apt-get install aws-neuronx-runtime-lib=2.* -y
               sudo apt-get install aws-neuronx-collectives=2.* -y

               # Update EFA Driver(only required for multiinstance training)
               
               curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
               wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
               cat aws-efa-installer.key | gpg --fingerprint
               wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
         
               tar -xvf aws-efa-installer-latest.tar.gz
               cd aws-efa-installer && sudo bash efa_installer.sh --yes
               cd
               sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

               # Update Neuron Tools
               sudo apt-get install aws-neuronx-tools=2.* -y

               # Activate a Python virtual environment where Neuron pip packages were installed.
               source aws_neuron_venv_pytorch_p38/bin/activate
                        

               # Install packages from repos
               python -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"
               
               # Update Python packages - Transformers package is needed for BERT
               python -m pip install torch-neuronx=="1.11.0.1.*" "neuronx-cc==2.*" transformers


         .. tab-item:: Amazon Linux 2 AMI

            .. include :: /note-setup-general.rst
            .. code:: bash

         
               # Update OS headers
               sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y
               
               # Update OS packages
               sudo yum update -y


               # Update Neuron Driver and Runtime
               sudo yum install aws-neuronx-dkms-2.*  -y
               sudo yum install aws-neuronx-oci-hook-2.*  -y
               sudo yum install aws-neuronx-runtime-lib-2.*  -y
               sudo yum install aws-neuronx-collectives-2.*  -y

               # Update EFA Driver(only required for multiinstance training)
               curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
               wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key
               cat aws-efa-installer.key | gpg --fingerprint
               wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig
               tar -xvf aws-efa-installer-latest.tar.gz
               cd aws-efa-installer && sudo bash efa_installer.sh --yes
               cd
               sudo rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer

               # Update Neuron Tools
               sudo yum install aws-neuronx-tools-2.*  -y

               # Activate a Python virtual environment where Neuron pip packages were installed.
               source aws_neuron_venv_pytorch_p37/bin/activate

               # Update packages from repos
               python -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"
               
               # Update Python packages - Transformers package is needed for BERT
               python -m pip install torch-neuronx=="1.11.0.1.*" "neuronx-cc==2.*" transformers

