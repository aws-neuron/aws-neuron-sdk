
.. dropdown::  Launch Trn1 Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. include:: /general/setup/install-templates/launch-trn1.txt


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

        # Install git
        sudo yum install git -y


        # Install OS headers
        sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y

        # Remove preinstalled packages and Install Neuron Driver and Runtime
        sudo yum remove aws-neuron-dkms -y
        sudo yum remove aws-neuronx-dkms -y
        sudo yum remove aws-neuronx-oci-hook -y
        sudo yum remove aws-neuronx-runtime-lib -y
        sudo yum remove aws-neuronx-collectives -y
        sudo yum install aws-neuronx-dkms-2.*  -y
        sudo yum install aws-neuronx-oci-hook-2.*  -y
        sudo yum install aws-neuronx-runtime-lib-2.*  -y
        sudo yum install aws-neuronx-collectives-2.*  -y

        # Install EFA Driver(only required for multiinstance training)
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
        sudo yum install aws-neuronx-tools-2.*  -y

        export PATH=/opt/aws/neuron/bin:$PATH

.. dropdown::  Install PyTorch Neuron (``torch-neuronx``)
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. code:: bash

        # Install Python venv and activate Python virtual environment to install
        # Neuron pip packages.
        python3.7 -m venv aws_neuron_venv_pytorch
        source aws_neuron_venv_pytorch/bin/activate
        pip install -U pip

        # Install packages from repos
        python -m pip config set global.extra-index-url "https://pip.repos.neuron.amazonaws.com"

        # Install wget, awscli
        pip install wget
        pip install awscli

        # Install Neuron packages
        pip install torch-neuronx==1.12.0.1.*
        pip install neuronx-cc==2.*


.. dropdown::  Run Tutorial
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    :ref:`neuronx-mlp-training-tutorial`


.. card:: Visit PyTorch Neuron section for more
    :class-body: sphinx-design-class-body-small
    :link: pytorch-neuronx-main
    :link-type: ref
