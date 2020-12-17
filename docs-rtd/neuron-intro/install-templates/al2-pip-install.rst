
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
   