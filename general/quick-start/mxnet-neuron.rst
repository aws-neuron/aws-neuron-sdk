.. _mxnet_quick_start:


Get Started with Apache MXNet (Incubating) Neuron
=================================================

This page provide links that will assist you to quickly start with :ref:`mxnet-neuron-main` (supporting inference only).


.. _mxnet_quick_start_inference:

.. tab-set::

   .. tab-item:: Get Started with Inference


        .. dropdown::  Launch Inf1 Instance
                :class-title: sphinx-design-class-title-small
                :class-body: sphinx-design-class-body-small
                :animate: fade-in

                .. include:: /general/setup/install-templates/launch-inf1.txt

        .. dropdown::  Install Drivers and Tools
                :class-title: sphinx-design-class-title-small
                :class-body: sphinx-design-class-body-small
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
			# To install or update to Neuron versions 1.19.1 and newer from previous releases:
			# - DO NOT skip 'aws-neuron-dkms' install or upgrade step, you MUST install or upgrade to latest Neuron driver
			################################################################################################################

			# Install OS headers
			sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y

			# Install Neuron Driver
			sudo yum install aws-neuron-dkms -y

			####################################################################################
			# Warning: If Linux kernel is updated as a result of OS package update
			#          Neuron driver (aws-neuron-dkms) should be re-installed after reboot
			####################################################################################

			# Install Neuron Tools
			sudo yum install aws-neuron-tools -y

			export PATH=/opt/aws/neuron/bin:$PATH


        .. dropdown::  Install MXNet Neuron
                :class-title: sphinx-design-class-title-small
                :class-body: sphinx-design-class-body-small
                :animate: fade-in                

                .. code:: bash

			# Install Python venv and activate Python virtual environment to install    
			# Neuron pip packages.
			sudo yum install -y python3.7-venv gcc-c++
			python3.7 -m venv mxnet_venv
			source mxnet_venv/bin/activate
			pip install -U pip


			# Install Jupyter notebook kernel 
			pip install ipykernel 
			python3.7 -m ipykernel install --user --name mxnet_venv --display-name "Python (Neuron MXNet)"
			pip install jupyter notebook
			pip install environment_kernels


			# Set Pip repository  to point to the Neuron repository
			pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

			#Install Neuron MXNet
			wget https://aws-mx-pypi.s3.us-west-2.amazonaws.com/1.8.0/aws_mx-1.8.0.2-py2.py3-none-manylinux2014_x86_64.whl
			pip install aws_mx-1.8.0.2-py2.py3-none-manylinux2014_x86_64.whl
			pip install mx_neuron neuron-cc

        .. dropdown::  Run Tutorial
               :class-title: sphinx-design-class-title-small
               :class-body: sphinx-design-class-body-small
               :animate: fade-in
        
               :ref:`ResNet-50 </src/examples/mxnet/resnet50/resnet50.ipynb>`


        .. card:: Visit MXNet Neuron section for more
                :class-body: sphinx-design-class-body-small
                :link: mxnet-neuron-main
                :link-type: ref




