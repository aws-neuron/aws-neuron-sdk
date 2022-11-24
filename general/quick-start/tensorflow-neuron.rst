.. _tensorflow_quick_start:

Get Started with TensorFlow Neuron
==================================

This page provide links that will assist you to quickly start with :ref:`tensorflow-neuron-main`.


.. _tensorflow_quick_start_inference:


.. tab-set::

    .. tab-item:: Get Started with Inference


        .. dropdown::  Launch Inf1 Instance
                :class-title: drop-down-class-title-small
                :class-body: drop-down-class-body-small
                :animate: fade-in

                .. include:: /general/setup/install-templates/launch-inf1.txt


        .. dropdown::  Install Drivers and Tools
                :class-title: drop-down-class-title-small
                :class-body: drop-down-class-body-small
                :animate: fade-in

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

			# Update OS packages
			sudo yum update -y
			
			################################################################################################################
			# Remove older versions of Neuron
			################################################################################################################
			sudo yum remove aws-neuron-dkms -y
			sudo yum remove aws-neuronx-dkms -y
			sudo yum remove aws-neuron-tools -y
			sudo yum remove aws-neuronx-tools -y

			################################################################################################################
			# To install or update to Neuron versions 2.5 and newer from previous releases:
			# - DO NOT skip 'aws-neuronx-dkms' install or upgrade step, you MUST install or upgrade to latest Neuron driver
			################################################################################################################

			# Install OS headers
			sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y

			# Install Neuron Driver
			sudo yum install aws-neuronx-dkms-2.* -y

			####################################################################################
			# Warning: If Linux kernel is updated as a result of OS package update
			#          Neuron driver (aws-neuron-dkms) should be re-installed after reboot
			####################################################################################

			# Install Neuron Tools
			sudo yum install aws-neuronx-tools=2.* -y

			export PATH=/opt/aws/neuron/bin:$PATH


        .. dropdown::  Install TensorFlow Neuron (``tensorflow-neuron``)
                :class-title: drop-down-class-title-small
                :class-body: drop-down-class-body-small
                :animate: fade-in                

                .. code:: bash
		
			# Install Python venv and activate Python virtual environment to install    
			# Neuron pip packages.
			sudo yum install -y python3.7-venv gcc-c++
			python3.7 -m venv tensorflow_venv
			source tensorflow_venv/bin/activate
			pip install -U pip


			# Install Jupyter notebook kernel 
			pip install ipykernel 
			python3.7 -m ipykernel install --user --name tensorflow_venv --display-name "Python (Neuron TensorFlow)"
			pip install jupyter notebook
			pip install environment_kernels


			# Set Pip repository  to point to the Neuron repository
			pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

			#Install Neuron TensorFlow
			pip install tensorflow-neuron[cc] "protobuf==3.20.1"

			# Optional: Install Neuron TensorFlow model server
			sudo yum install tensorflow-model-server-neuron -y

			# Install Neuron TensorBoard
			pip install tensorboard-plugin-neuron
                

        .. dropdown::  Run Tutorial
               :class-title: sphinx-design-class-title-small
               :class-body: sphinx-design-class-body-small
               :animate: fade-in
              
	       :ref:`ResNet-50 </src/examples/tensorflow/tensorflow_resnet50/resnet50.ipynb>` 

        .. card:: Visit TensorFlow Neuron section for more
                :class-body: sphinx-design-class-body-small
                :link: tensorflow-neuron-main
                :link-type: ref


  
  

    .. tab-item:: Get Started with Training

		.. note::
        
			TensorFlow Neuron support for training workloads is coming soon.



