
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

.. note::

   ``aws-neuron-dkms`` is a special kernel module package that has a dependency on the linux kernel version. This
   means that when the package is installed through apt, it will only be compatible with the linux kernel version
   that was running on the instance during the installation.

   You have to re-install this package if you change the linux kernel version. The current kernel version can be
   checked using ``uname -r``, and a list of the kernels that have dkms installed can be checked with
   ``dkms status | grep aws-neuron``. Refer to the :ref:`NRT Troubleshooting Guide <neuron-driver-installation-fails>`
   for steps on how to re-install aws-neuron-dkms on a new kernel.
