* Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to launch an Inf2 instance, when choosing the instance type at the EC2 console. Please make sure to select the correct instance type. To get more information about Inf2 instances sizes and pricing see `Inf2 web page <https://aws.amazon.com/ec2/instance-types/inf2/>`_.

* When choosing an Amazon Machine Image (AMI) make sure to select `Deep Learning AMI with Conda Options <https://docs.aws.amazon.com/dlami/latest/devguide/conda.html>`_. Please note that Neuron Conda environments are supported only in Ubuntu 18 DLAMI and Amazon Linux2 DLAMI, Neuron Conda environments are not supported in Amazon Linux DLAMI.



* After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-connect-to-instance-linux>`_ to connect to the instance 

.. note::

  You can also launch the instance from AWS CLI, please see :ref:`AWS CLI commands to launch inf2 instances <launch-inf1-dlami-aws-cli>`.

