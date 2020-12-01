.. _neuron-install-guide:

Neuron Install Guide - Configuring Linux for repository updates
===============================================================

Neuron is using standard package managers (apt, yum, pip, and conda) to
install and keep updates current. Please refer to the applicable Linux
section for detailed configuration steps.

Neuron supports Python versions 3.5, 3.6, and 3.7.

Ubuntu 16
---------

.. code:: bash

   . /etc/os-release
   sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
   deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
   EOF

   wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

   sudo apt-get update
   sudo apt-get install linux-headers-$(uname -r)
   sudo apt-get install aws-neuron-dkms
   sudo apt-get install aws-neuron-runtime-base
   sudo apt-get install aws-neuron-runtime
   sudo apt-get install aws-neuron-tools

.. note::

   If you see the following errors during apt-get install, please
   wait a minute or so for background updates to finish and retry apt-get
   install:

   .. code:: bash

      E: Could not get lock /var/lib/dpkg/lock-frontend - open (11: Resource temporarily unavailable)
      E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), is another process using it?

Amazon Linux, Centos, RHEL
--------------------------

Verify the instance has kernel version 4.14 or latest and kernel headers
are installed.

.. code:: bash

   sudo tee /etc/yum.repos.d/neuron.repo > /dev/null <<EOF
   [neuron]
   name=Neuron YUM Repository
   baseurl=https://yum.repos.neuron.amazonaws.com
   enabled=1
   metadata_expire=0
   EOF

   sudo rpm --import https://yum.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
   sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
   sudo yum install aws-neuron-dkms
   sudo yum install aws-neuron-runtime-base
   sudo yum install aws-neuron-runtime
   sudo yum install aws-neuron-tools

Neuron Pip Packages
-------------------

It is recommended to use a virtual environment when installing Neuron
pip packages. The following steps show how to setup the virtual
environment on Ubuntu or Amazon Linux:

.. code:: bash

   # Ubuntu
   sudo apt-get update
   sudo apt-get install -y python3-venv g++

.. code:: bash

   # Amazon Linux
   sudo yum update
   sudo yum install -y python3 gcc-c++

Setup a new Python virtual environment:

.. code:: bash

   python3 -m venv test_venv
   source test_venv/bin/activate
   pip install -U pip

Modify Pip repository configurations to point to the Neuron repository:

.. code:: bash

   tee $VIRTUAL_ENV/pip.conf > /dev/null <<EOF
   [global]
   extra-index-url = https://pip.repos.neuron.amazonaws.com
   EOF

.. note::

   .. container:: toggle-header

      .. code:: bash

         curl https://pip.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --import
         pip download --no-deps neuron-cc
         # The above shows you the name of the package downloaded
         # Use it in the following command
         wget https://pip.repos.neuron.amazonaws.com/neuron-cc/neuron_cc-<VERSION FROM FILE>.whl.asc
         gpg --verify neuron_cc-<VERSION FROM FILE>.whl.asc neuron_cc-<VERSION FROM FILE>.whl

The following Pip installation commands assume you are using a virtual
Python environment (see above for instructions on how to setup a virtual
Python environment). If not using virtual Python environment, please
switch 'pip' with 'pip3' as appropriate for your Python environment.

TensorFlow
~~~~~~~~~~

.. code:: bash

   pip install neuron-cc
   pip install tensorflow-neuron

Please ignore the following error displayed during installation:

.. code:: bash

   ERROR: tensorflow-serving-api 1.15.0 requires tensorflow~=1.15.0, which is not installed.

TensorFlow Model Serving
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   sudo apt-get install tensorflow-model-server-neuron
   pip install tensorflow_serving_api

TensorBoard
~~~~~~~~~~~

.. code:: bash

   pip install tensorboard-neuron

-  Installing ``tensorflow-neuron`` will automatically install
   ``tensorboard-neuron`` as a dependency
-  To verify ``tensorboard-neuron`` is installed correctly, run
   ``tensorboard_neuron -h | grep run_neuron_profile``. If nothing is
   shown, please retry installation with the ``--force-reinstall``
   option.

MXNet
~~~~~

.. code:: bash

   pip install neuron-cc
   pip install mxnet-neuron

PyTorch
~~~~~~~

.. code:: bash

   #NOTE please make sure [tensorflow] option is provided during installation of neuron-cc for PyTorch-Neuron compilation; this is not necessary for PyTorch-Neuron inference.
   pip install neuron-cc[tensorflow]
   pip install torch-neuron

.. _neuron-install-conda-packages:

Neuron Conda Packages
---------------------

The following commands assumes you are using a Conda environment and
have already activated it. Please see
https://docs.conda.io/projects/conda/en/latest/user-guide/install/ for
installation instruction if Conda is not installed. The following steps
are example steps to install and activate Conda environment:

.. code:: bash

   curl -O https://repo.anaconda.com/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh
   echo "bfe34e1fa28d6d75a7ad05fd02fa5472275673d5f5621b77380898dee1be15d2 Miniconda3-4.7.12.1-Linux-x86_64.sh" | sha256sum --check
   bash Miniconda3-4.7.12.1-Linux-x86_64.sh
   source ~/.bashrc
   conda create -q -y -n test_conda_env python=3.6
   source activate test_conda_env

.. code:: bash

   # Add Neuron Conda channel to Conda environment
   conda config --env --add channels https://conda.repos.neuron.amazonaws.com

   # Install one of frameworks in the newly created conda environment

   # If you are installing MXNet-Neuron plus Neuron-Compiler
   conda install mxnet-neuron

   # If you are installing TensorFlow-Neuron plus Neuron-Compiler
   conda install tensorflow-neuron

   # If you are installing PyTorch-Neuron plus Neuron-Compiler
   conda install torch-neuron

.. note::

   The framework Conda packages already include ``neuron-cc`` packages
   for compilation so there's no need to install them separately.

.. note::

   The ``tensorflow-neuron`` Conda package comes with
   TensorBoard-Neuron. There is no standalone ``tensorboard-neuron``
   Conda package at this time.

.. note::

   .. container:: toggle-header

      .. code:: bash

         curl https://conda.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --import

         # This shows the version/build number of the package
         conda search tensorflow-neuron

         # Use the version/build number above to download the package and the signature
         wget https://conda.repos.neuron.amazonaws.com/linux-64/tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2
         wget https://conda.repos.neuron.amazonaws.com/linux-64/tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2.asc
         gpg --verify tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2.asc tensorflow-neuron-<VERSION FROM FILE>-py36_0.tar.bz2
