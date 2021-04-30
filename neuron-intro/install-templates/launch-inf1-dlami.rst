* Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to Launch an Inf1 instance, when choosing the instance type at the EC2 console. Please make sure to select the correct instance type. To get more information about Inf1 instances sizes and pricing see `Inf1 web page <https://aws.amazon.com/ec2/instance-types/inf1/>`_.

* When choosing an Amazon Machine Image (AMI) make sure to select `Deep Learning AMI with Conda Options <https://docs.aws.amazon.com/dlami/latest/devguide/conda.html>`_. Please note that Neuron Conda packages are supported only in Ubuntu 16 DLAMI, Ubuntu 18 DLAMI and Amazon Linux2 DLAMI, Neuron Conda packages are not supported in Amazon Linux DLAMI.



* After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-connect-to-instance-linux>`_ to connect to the instance 

.. note::

  You can also launch the instance from AWS CLI, please see :ref:`AWS CLI commands to launch inf1 instances <launch-inf1-dlami-aws-cli>`.

