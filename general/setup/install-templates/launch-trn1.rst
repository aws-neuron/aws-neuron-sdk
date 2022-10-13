* Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to Launch an Trn1 instance, when choosing the instance type at the EC2 console. Please make sure to select the correct instance type. To get more information about Trn1 instances sizes and pricing see `Trn1 web page <https://aws.amazon.com/ec2/instance-types/trn1/>`_.

* Select your Amazon Machine Image (AMI) of choice, please note that Neuron support Amazon Linux 2 AMI(HVM) - Kernel 5.10. 

* When launching a trn1, please adjust your primary EBS volume size to a minimum of 512GB.


* After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-connect-to-instance-linux>`_ to connect to the instance 


.. include:: /general/setup/install-templates/trn1-ga-warning.txt
