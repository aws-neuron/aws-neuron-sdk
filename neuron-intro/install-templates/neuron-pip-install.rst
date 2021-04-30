
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

.. include:: /neuron-intro/install-templates/neuron-pip-setup.rst

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
