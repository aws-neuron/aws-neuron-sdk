
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

.. note::

   ``aws-neuron-dkms`` is a special kernel module package that has a dependency on the linux kernel version. This
   means that when the package is installed through yum, it will only be compatible with the linux kernel version
   that was running on the instance during the installation.

   You have to re-install this package if you change the linux kernel version. The current kernel version can be
   checked using ``uname -r``, and a list of the kernels that have dkms installed can be checked with
   ``dkms status | grep aws-neuron``. Refer to the :ref:`NRT Troubleshooting Guide <neuron-driver-installation-fails>`
   for steps on how to re-install aws-neuron-dkms on a new kernel.
